"""Pool - Resource management for skyward.

A Pool manages provisioned cloud resources (instances) and executes
computations on them. It separates environment configuration from
function definition, allowing multiple functions to share the same
resources.

Example:
    from skyward import compute, ComputePool, AWS, Image

    @compute
    def train(data):
        return model.fit(data)

    @compute
    def evaluate(model):
        return model.evaluate()

    # Create a pool with GPU and PyTorch
    pool = ComputePool(
        provider=AWS(region="us-east-1"),
        accelerator="A100",
        image=Image(pip=["torch", "transformers"]),
    )

    with pool:
        # Both functions share the same provisioned resources
        model = train(data) | pool
        score = evaluate(model) | pool
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import AbstractContextManager, ExitStack, suppress
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal

import rpyc
from loguru import logger

from skyward.accelerator import Accelerator
from skyward.callback import Callback, _callback, compose, emit, use_callback
from skyward.callbacks import panel
from skyward.conc import for_each_async, map_async
from skyward.events import (
    Error,
    FunctionCall,
    FunctionResult,
    LogLine,
    PoolStarted,
    PoolStopping,
    ProviderName,
    ProvisionedInstance,
)
from skyward.exceptions import ExecutionError, NotProvisionedError, ProvisioningError
from skyward.image import DEFAULT_IMAGE, Image
from skyward.logging import LogConfig, _setup_logging, _teardown_logging
from skyward.metrics import MetricsStreamer
from skyward.pending import PendingBatch, PendingCompute
from skyward.selection import (
    AllProvidersFailedError,
    normalize_providers,
    normalize_selector,
)
from skyward.spec import AllocationLike
from skyward.task import PooledConnection, TaskPool
from skyward.types import (
    Architecture,
    Auto,
    Instance,
    Memory,
    Provider,
    ProviderLike,
    SelectionLike,
)
from skyward.volume import Volume, parse_volume_uri


def _parse_volumes(
    volume: dict[str, str] | Sequence[Volume] | None
) -> tuple[Volume, ...]:
    """Parse volume specification into Volume objects."""
    match volume:
        case None:
            return ()
        case dict() as d:
            return tuple(parse_volume_uri(mount_path, uri) for mount_path, uri in d.items())
        case [*items]:
            return tuple(items)
        case _:
            raise TypeError(
                f"volume must be dict or Sequence[Volume], got {type(volume).__name__}"
            )


@dataclass
class ComputePool:
    """Resource pool for executing computations.

    A Pool provisions cloud resources (instances) and executes
    PendingCompute objects on them. Multiple functions can share
    the same pool, avoiding redundant provisioning.

    Args:
        provider: Cloud provider (AWS, DigitalOcean). Required.
        image: Image specification (python, pip, apt, env).
        nodes: Number of nodes to provision. Default 1.
        accelerator: Accelerator specification. Formats:
            - "H100" → 1 accelerator
            - Accelerator.NVIDIA.H100() → 1 GPU full
            - Accelerator.NVIDIA.H100(count=8) → 8 GPUs
            - Accelerator.NVIDIA.H100(mig="3g.40gb") → 1 MIG partition
        architecture: CPU architecture preference. Formats:
            - Auto() → cheapest architecture (default)
            - "arm64" → ARM instances only
            - "x86_64" → x86_64 instances only
        allocation: Instance allocation strategy. Formats:
            - "spot-if-available" → try spot, fallback on-demand (default)
            - "always-spot" → 100% spot, fail if unavailable
            - "on-demand" → 100% on-demand
            - "cheapest" → compare all options, pick cheapest
            - 0.8 → minimum 80% spot
        volume: Volumes to mount.
        cpu: CPU cores per worker (for cgroups limits).
        memory: Memory per worker (e.g., "32GB", for cgroups limits).
        env: Environment variables (merged with image.env, pool overrides).

    Example:
        pool = ComputePool(
            provider=AWS(),
            accelerator=Accelerator.NVIDIA.A100(),
            image=Image(pip=["torch", "transformers"]),
        )

        with pool:
            result = train(data) | pool
    """

    # Required
    provider: ProviderLike

    # Provider selection (when multiple providers given)
    selection: SelectionLike = "first"

    # Environment specification
    image: Image = field(default_factory=lambda: DEFAULT_IMAGE)

    # Resource specification
    nodes: int = 1
    machine: str | None = None  # Direct instance type override (e.g., "p5.48xlarge")
    accelerator: Accelerator | str | None = None
    architecture: Architecture = field(default_factory=Auto)
    cpu: int | None = None
    memory: Memory | None = None
    volume: dict[str, str] | Sequence[Volume] | None = None
    allocation: AllocationLike = "spot-if-available"
    timeout: int = 3600
    env: dict[str, str] | None = None

    # Concurrency
    concurrency: int = 1  # Number of concurrent tasks per instance

    # Display settings
    display: Literal["panel", "quiet"] = "quiet"
    on_event: Callback | None = None
    collect_metrics: bool = True

    # Logging configuration
    logging: LogConfig | bool = True

    # Internal state
    _active: bool = field(default=False, init=False, repr=False)
    _built_provider: Provider | None = field(default=None, init=False, repr=False)
    _log_config: LogConfig | None = field(default=None, init=False, repr=False)
    _log_handler_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _instances: tuple[Instance, ...] | None = field(default=None, init=False, repr=False)
    _callback_ctx: AbstractContextManager[None] | None = field(
        default=None, init=False, repr=False
    )
    _job_id: str = field(default="", init=False, repr=False)

    # Task pool state
    _accelerator_cfg: Accelerator | None = field(default=None, init=False, repr=False)
    _task_pool: TaskPool | None = field(default=None, init=False, repr=False)
    _metrics_streamers: list[MetricsStreamer] = field(default_factory=list, init=False, repr=False)
    _metrics_connections: list[rpyc.Connection] = field(
        default_factory=list, init=False, repr=False
    )
    _event_callback: Callback | None = field(default=None, init=False, repr=False)
    _exit_stack: ExitStack | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Parse accelerator and logging configuration."""
        if self.accelerator is not None:
            self._accelerator_cfg = Accelerator.from_value(self.accelerator)

        # Parse logging configuration
        match self.logging:
            case True:
                self._log_config = LogConfig()
            case LogConfig() as cfg:
                self._log_config = cfg
            case False:
                self._log_config = None

    def _parse_accelerator_for_provider(self) -> str | None:
        """Get the accelerator type for the provider."""
        if self._accelerator_cfg is not None:
            return self._accelerator_cfg.accelerator
        return None

    @property
    def workers_per_instance(self) -> int:
        """Number of workers per instance.

        Always returns 1 - one RPyC server per instance.
        Concurrency is controlled by the `concurrency` parameter.
        """
        return 1

    @property
    def total_slots(self) -> int:
        """Total number of execution slots (nodes * concurrency)."""
        return self.nodes * self.concurrency

    def _get_worker_isolation_data(
        self, device_count: int
    ) -> tuple[None, None, str]:
        """Get worker isolation configuration.

        Always returns (None, None, "") - no worker isolation.
        We use a single RPyC server per instance with concurrent slots.
        """
        return None, None, ""

    def __enter__(self) -> ComputePool:
        """Enter pool context and provision resources."""
        import uuid

        # Setup logging first (before any log statements)
        if self._log_config:
            self._log_handler_ids = _setup_logging(self._log_config)

        with ExitStack() as stack:
            self._active = True
            self._job_id = str(uuid.uuid4())[:8]

            # Normalize and build all providers
            configs = normalize_providers(self.provider)
            built_providers = tuple(config.build() for config in configs)
            selector = normalize_selector(self.selection)

            provider_names = [p.name for p in built_providers]
            logger.info(f"Starting pool with {self.nodes} nodes, providers: {provider_names}")
            logger.debug(
                f"Pool config: accelerator={self.accelerator}, "
                f"allocation={self.allocation}, concurrency={self.concurrency}"
            )

            # 1. Callback context (cleanup: exits context)
            self._event_callback = self._build_callback()
            self._callback_ctx = use_callback(self._event_callback)
            stack.enter_context(self._callback_ctx)

            # 2. Shutdown (cleanup: terminates instances)
            #    Registered BEFORE provisioning to capture partial failures
            stack.callback(self._shutdown)

            # 3. Spinner (cleanup: stops spinner)
            emit(PoolStarted(nodes=self.nodes))
            stack.callback(lambda: emit(PoolStopping()))

            # 4. Select provider and provision with fallback
            compute = self._build_compute_spec()
            selected = selector(built_providers, compute)

            # Order: selected first, then remaining for fallback
            ordered = (selected,) + tuple(p for p in built_providers if p is not selected)

            errors: list[tuple[Provider, Exception]] = []
            for provider in ordered:
                try:
                    logger.info(f"Trying provider: {provider.name}")
                    self._built_provider = provider
                    self._provision()
                    break
                except ProvisioningError as e:
                    logger.warning(f"Provider {provider.name} failed: {e}")
                    errors.append((provider, e))
                    self._instances = None  # Reset for next attempt
                    continue
            else:
                raise AllProvidersFailedError(errors)

            logger.info(f"Pool ready with {len(self._instances or [])} instances")

            # Success - transfer ownership
            self._exit_stack = stack.pop_all()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit pool context and release resources."""
        logger.info("Stopping pool...")
        try:
            if self._exit_stack is not None:
                self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
                self._exit_stack = None
        finally:
            self._active = False

            # Teardown logging last
            if self._log_handler_ids:
                logger.debug("Pool shutdown complete")
                _teardown_logging(self._log_handler_ids)
                self._log_handler_ids = []

    def run[R](self, pending: PendingCompute[R]) -> R:
        """Execute a single computation on the pool.

        Acquires the first available connection, executes, and releases.

        Args:
            pending: PendingCompute to execute.

        Returns:
            The result of the computation.

        Raises:
            NotProvisionedError: If pool is not active.
        """
        if not self._active or self._task_pool is None:
            raise NotProvisionedError()

        logger.debug(f"Executing task: {pending.fn.__name__}")
        conn = self._task_pool.acquire()
        try:
            result = self._execute_on_connection(conn, pending)
            logger.debug(f"Task {pending.fn.__name__} completed successfully")
            return result
        finally:
            self._task_pool.release(conn)

    def run_batch(self, batch: PendingBatch) -> tuple[Any, ...]:
        """Execute a batch of computations in parallel.

        Distributes computations across available connections (round-robin).

        Args:
            batch: PendingBatch containing computations to execute.

        Returns:
            Tuple of results in the same order as the computations.

        Raises:
            NotProvisionedError: If pool is not active.
        """
        if not self._active or self._task_pool is None:
            raise NotProvisionedError()

        computations = list(batch.computations)
        n_computations = len(computations)
        logger.debug(f"Executing batch of {n_computations} tasks")

        # Acquire as many connections as we need (up to available)
        n_conns = min(n_computations, self._task_pool.total_slots)
        conns = [self._task_pool.acquire() for _ in range(n_conns)]

        try:
            # Assign computations to connections (round-robin)
            def execute(item: tuple[int, PendingCompute[Any]]) -> Any:
                idx, pending = item
                conn = conns[idx % n_conns]
                return self._execute_on_connection(conn, pending)

            results = list(
                map_async(
                    execute,
                    list(enumerate(computations)),
                    concurrency=n_conns,
                )
            )
            return tuple(results)
        finally:
            for conn in conns:
                self._task_pool.release(conn)

    def broadcast[R](self, pending: PendingCompute[R]) -> tuple[R, ...]:
        """Broadcast: execute computation on ALL connections in parallel.

        Unlike run() which executes on one connection, broadcast() executes the
        same computation on every connection in the pool simultaneously.

        Args:
            pending: PendingCompute to broadcast.

        Returns:
            Tuple of results, one per connection.

        Raises:
            NotProvisionedError: If pool is not active.

        Example:
            # Initialize model on all connections
            results = load_model(path) @ pool  # tuple of N results
        """
        if not self._active or self._task_pool is None:
            raise NotProvisionedError()

        # Acquire all connections
        total = self._task_pool.total_slots
        conns = [self._task_pool.acquire() for _ in range(total)]

        try:
            results = list(
                map_async(
                    lambda c: self._execute_on_connection(c, pending),
                    conns,
                )
            )
            return tuple(results)
        finally:
            for conn in conns:
                self._task_pool.release(conn)

    def _execute_on_connection[R](
        self,
        pc: PooledConnection,
        pending: PendingCompute[R],
    ) -> R:
        """Execute a computation on a pooled connection."""
        assert self._task_pool is not None
        assert self._instances is not None

        conn = pc.conn

        from skyward.serialization import deserialize, serialize

        fn_bytes = serialize(pending.fn)
        args_bytes = serialize(pending.args)
        kwargs_bytes = serialize(pending.kwargs_dict)

        event_callback = self._event_callback

        # Create ProvisionedInstance for events
        provisioned = self._make_provisioned(pc.instance)

        def stdout_callback(line: str) -> None:
            if event_callback is not None:
                event_callback(
                    LogLine(
                        instance=provisioned,
                        line=line,
                        timestamp=time.time(),
                    )
                )

        # Emit FunctionCall event
        fn_name = getattr(pending.fn, "__name__", str(pending.fn))
        start_time = time.time()
        if event_callback is not None:
            event_callback(
                FunctionCall(
                    function_name=fn_name,
                    instance=provisioned,
                    timestamp=start_time,
                )
            )

        try:
            result_bytes = conn.root.execute(fn_bytes, args_bytes, kwargs_bytes, stdout_callback)
            response = deserialize(result_bytes)
            if response.get("error"):
                raise ExecutionError(response["error"])

            # Emit FunctionResult on success
            if event_callback is not None:
                event_callback(
                    FunctionResult(
                        function_name=fn_name,
                        instance=provisioned,
                        timestamp=time.time(),
                        duration_seconds=time.time() - start_time,
                        success=True,
                    )
                )
            return response["result"]
        except ExecutionError:
            # Emit FunctionResult on execution error
            if event_callback is not None:
                event_callback(
                    FunctionResult(
                        function_name=fn_name,
                        instance=provisioned,
                        timestamp=time.time(),
                        duration_seconds=time.time() - start_time,
                        success=False,
                        error="ExecutionError",
                    )
                )
            raise
        except Exception as e:
            # Emit FunctionResult on unexpected error
            if event_callback is not None:
                event_callback(
                    FunctionResult(
                        function_name=fn_name,
                        instance=provisioned,
                        timestamp=time.time(),
                        duration_seconds=time.time() - start_time,
                        success=False,
                        error=str(e),
                    )
                )
            emit(Error(message=f"Execution failed on {pc.instance_id}: {e}", instance=provisioned))
            raise

    def _build_compute_spec(self) -> Any:
        """Build compute spec for provider selection and provisioning."""
        from skyward.pool_compute import _PoolCompute

        return _PoolCompute(
            pool=self,
            fn=lambda: None,  # Placeholder
            nodes=self.nodes,
            machine=self.machine,
            accelerator=self.accelerator,
            architecture=self.architecture,
            image=self.image,
            cpu=self.cpu,
            memory=self.memory,
            timeout=self.timeout,
            allocation=self.allocation,
            volumes=list(_parse_volumes(self.volume)),
        )

    def _provision(self) -> None:
        """Provision resources for the pool."""
        logger.debug(f"Creating compute spec: nodes={self.nodes}, machine={self.machine}")
        compute = self._build_compute_spec()

        # Provision and setup (providers use emit() directly)
        logger.info(f"Provisioning {self.nodes} instances via {self._built_provider.name}...")
        self._instances = self._built_provider.provision(compute)
        logger.debug(f"Provisioned {len(self._instances)} instances")

        logger.debug("Running instance setup/bootstrap...")
        self._built_provider.setup(self._instances, compute)

        # Initialize TaskPool (creates tunnels and connections in parallel)
        logger.debug("Initializing task pool and connections...")
        self._task_pool = TaskPool(
            provider=self._built_provider,
            instances=self._instances,
            concurrency=self.concurrency,
        )

        # Setup cluster on all instances in parallel (once per instance)
        logger.debug("Setting up cluster environment on instances...")
        self._setup_all_instances()

        # Start metrics streaming if enabled (dedicated RPyC connections)
        if self.collect_metrics:
            logger.debug("Starting metrics streaming...")
            provider_name = self._get_provider_name()

            # Create dedicated connections for metrics (one per instance)
            for instance in self._instances:
                tunnel = self._task_pool.get_tunnel(instance)
                conn = rpyc.connect(
                    "127.0.0.1",
                    tunnel.local_port,
                    config={
                        "allow_pickle": True,
                        "sync_request_timeout": 3600,
                    },
                )
                self._metrics_connections.append(conn)

                streamer = MetricsStreamer(
                    instance=instance,
                    conn=conn,
                    callback=self._event_callback,
                    interval=0.2,
                    provider_name=provider_name,
                )
                streamer.start()
                self._metrics_streamers.append(streamer)

    def _shutdown(self) -> None:
        """Shutdown and release pool resources."""
        if self._instances is None:
            return

        from skyward.pool_compute import _PoolCompute

        instance_count = len(self._instances)
        logger.debug(f"Shutting down {instance_count} instances...")

        # Stop metrics streaming
        if self._metrics_streamers:
            logger.debug("Stopping metrics streaming...")
            for streamer in self._metrics_streamers:
                streamer.stop()
            self._metrics_streamers.clear()

        # Close dedicated metrics connections
        if self._metrics_connections:
            logger.debug("Closing metrics connections...")
            for conn in self._metrics_connections:
                with suppress(Exception):
                    conn.close()
            self._metrics_connections.clear()

        # Close TaskPool if active
        if self._task_pool is not None:
            logger.debug("Closing task pool connections...")
            with suppress(Exception):
                self._task_pool.close_all()
            self._task_pool = None

        # Get accelerator type for provider
        accelerator_for_provider = self._parse_accelerator_for_provider()

        compute = _PoolCompute(
            pool=self,
            fn=lambda: None,
            nodes=self.nodes,
            machine=self.machine,
            accelerator=accelerator_for_provider,
            architecture=self.architecture,
            image=self.image,
            cpu=self.cpu,
            memory=self.memory,
            timeout=self.timeout,
            allocation=self.allocation,
            volumes=list(_parse_volumes(self.volume)),
        )

        logger.debug("Terminating instances...")
        with suppress(Exception):
            self._built_provider.shutdown(self._instances, compute)

        logger.info(f"Terminated {instance_count} instances")
        self._instances = None

    def _get_provider_name(self) -> ProviderName:
        """Get ProviderName enum from the built provider."""
        provider_name_map = {
            "aws": ProviderName.AWS,
            "digitalocean": ProviderName.DigitalOcean,
            "verda": ProviderName.Verda,
        }
        return provider_name_map.get(
            getattr(self._built_provider, "name", "aws").lower(),
            ProviderName.AWS,
        )

    def _make_provisioned(self, inst: Instance) -> ProvisionedInstance:
        """Create ProvisionedInstance from Instance for events."""
        return ProvisionedInstance(
            instance_id=inst.id,
            node=inst.node,
            provider=self._get_provider_name(),
            spot=inst.spot,
            spec=None,  # Spec not available at this point
            ip=inst.private_ip,
        )

    def _setup_all_instances(self) -> None:
        """Setup cluster environment on all instances in parallel."""
        import time as _time

        from skyward.events import InstanceReady, PoolReady

        assert self._task_pool is not None
        assert self._instances is not None

        t0 = _time.perf_counter()

        # Acquire one connection per instance (first N connections are interleaved)
        conns = [self._task_pool.acquire() for _ in range(len(self._instances))]

        # Build provisioned instances for events
        provisioned_list = [self._make_provisioned(inst) for inst in self._instances]

        try:
            # Setup in parallel and emit InstanceReady for each
            def setup(pc: PooledConnection) -> None:
                self._setup_cluster(pc)
                provisioned = self._make_provisioned(pc.instance)
                emit(InstanceReady(instance=provisioned))

            for_each_async(setup, conns)
        finally:
            for conn in conns:
                self._task_pool.release(conn)

        cluster_setup_duration = _time.perf_counter() - t0

        # Emit consolidated PoolReady event
        emit(PoolReady(
            instances=tuple(provisioned_list),
            tunnels=self._task_pool.tunnel_count,
            connections=self._task_pool.connection_count,
            total_duration_seconds=self._task_pool.init_duration_seconds + cluster_setup_duration,
        ))

    def _setup_cluster(self, pc: PooledConnection) -> None:
        """Setup cluster environment on instance (called once per instance)."""
        from skyward.providers.pool_info import build_pool_info
        from skyward.serialization import serialize

        # Build peer info (instances is guaranteed non-None when called)
        instances = self._instances
        assert instances is not None
        peers = [
            {"node": i, "private_ip": inst.private_ip, "addr": inst.private_ip}
            for i, inst in enumerate(instances)
        ]

        # Resolve accelerator by probing instance
        from skyward.providers.accelerator_detection import resolve_accelerator

        accelerator_type, accelerator_count = resolve_accelerator(pc.instance)

        pool_info = build_pool_info(
            node=pc.node_idx,
            total_nodes=len(instances),
            accelerator_count=accelerator_count,
            total_accelerators=len(instances) * accelerator_count,
            head_addr=instances[0].private_ip,
            head_port=29500,
            job_id=self._job_id,
            peers=peers,
            accelerator_type=accelerator_type,
            worker=0,  # Always 0 since we have single RPyC server per instance
            workers_per_node=1,  # Always 1 since we use concurrency, not workers
        )

        # Merge env from image and pool (pool overrides image)
        merged_env = {**self.image.env, **(self.env or {})}
        env_bytes = serialize(merged_env)
        pc.conn.root.setup_cluster(pool_info.model_dump_json(), env_bytes)

    def _build_callback(self) -> Callback:
        """Build the composite callback for this pool."""
        callbacks: list[Callback] = []

        # Parent callback from context (e.g., from @sky.app)
        parent_cb = _callback.get()
        if parent_cb is not None:
            callbacks.append(parent_cb)

        # Display callback (panel has built-in cost tracking)
        match self.display:
            case "panel":
                callbacks.append(panel())
            case "quiet":
                pass

        # User callback
        if self.on_event is not None:
            callbacks.append(self.on_event)

        return compose(*callbacks)

    @property
    def is_active(self) -> bool:
        """True if pool is provisioned and ready."""
        return self._active and self._instances is not None

    @property
    def instance_count(self) -> int:
        """Number of provisioned instances."""
        return len(self._instances) if self._instances else 0

    def __repr__(self) -> str:
        status = "active" if self.is_active else "inactive"
        provider_name = (
            self._built_provider.name if self._built_provider
            else type(self.provider).__name__
        )
        return (
            f"Pool(provider={provider_name}, nodes={self.nodes}, "
            f"accelerator={self.accelerator}, {status})"
        )
