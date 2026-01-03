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
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal

from skyward.accelerator import Accelerator
from skyward.callback import Callback, compose, emit, use_callback
from skyward.callbacks import cost_tracker, log, spinner
from skyward.conc import for_each_async, map_async
from skyward.events import Error, LogLine, PoolStarted, PoolStopping
from skyward.exceptions import ExecutionError, NotProvisionedError
from skyward.image import DEFAULT_IMAGE, Image
from skyward.metrics import MetricsPoller
from skyward.pending import PendingBatch, PendingCompute
from skyward.spec import AllocationLike
from skyward.task import PooledConnection, TaskPool
from skyward.types import Architecture, Auto, Instance, Provider
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
    provider: Provider

    # Environment specification
    image: Image = field(default_factory=lambda: DEFAULT_IMAGE)

    # Resource specification
    nodes: int = 1
    machine: str | None = None  # Direct instance type override (e.g., "p5.48xlarge")
    accelerator: Accelerator | str | None = None
    architecture: Architecture = field(default_factory=Auto)
    cpu: int | None = None
    memory: str | None = None
    volume: dict[str, str] | Sequence[Volume] | None = None
    allocation: AllocationLike = "spot-if-available"
    timeout: int = 3600
    env: dict[str, str] | None = None

    # Concurrency
    concurrency: int = 1  # Number of concurrent tasks per instance

    # Display settings
    display: Literal["spinner", "log", "quiet"] = "log"
    on_event: Callback | None = None
    collect_metrics: bool = True

    # Internal state
    _active: bool = field(default=False, init=False, repr=False)
    _instances: tuple[Instance, ...] | None = field(default=None, init=False, repr=False)
    _callback_ctx: AbstractContextManager[None] | None = field(
        default=None, init=False, repr=False
    )
    _job_id: str = field(default="", init=False, repr=False)

    # Task pool state
    _accelerator_cfg: Accelerator | None = field(default=None, init=False, repr=False)
    _task_pool: TaskPool | None = field(default=None, init=False, repr=False)
    _metrics_pollers: list[MetricsPoller] = field(default_factory=list, init=False, repr=False)
    _event_callback: Callback | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Parse accelerator and store configuration."""
        if self.accelerator is not None:
            self._accelerator_cfg = Accelerator.from_value(self.accelerator)

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

        self._active = True
        self._job_id = str(uuid.uuid4())[:8]

        # Build and store callback for direct access (bypasses ContextVar issues with joblib)
        self._event_callback = self._build_callback()

        # Setup callback context for emit() usage
        self._callback_ctx = use_callback(self._event_callback)
        self._callback_ctx.__enter__()

        emit(PoolStarted())
        self._provision()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit pool context and release resources."""
        try:
            emit(PoolStopping())
            self._shutdown()
        finally:
            self._active = False
            if self._callback_ctx is not None:
                self._callback_ctx.__exit__(exc_type, exc_val, exc_tb)
                self._callback_ctx = None

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

        conn = self._task_pool.acquire()
        try:
            return self._execute_on_connection(conn, pending)
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

        def stdout_callback(line: str) -> None:
            if event_callback is not None:
                event_callback(
                    LogLine(
                        node=pc.node_idx,
                        instance_id=pc.instance_id,
                        line=line,
                        timestamp=time.time(),
                    )
                )

        try:
            result_bytes = conn.root.execute(fn_bytes, args_bytes, kwargs_bytes, stdout_callback)
            response = deserialize(result_bytes)
            if response.get("error"):
                raise ExecutionError(response["error"])
            return response["result"]
        except ExecutionError:
            raise
        except Exception as e:
            emit(Error(message=f"Execution failed on {pc.instance_id}: {e}"))
            raise

    def _provision(self) -> None:
        """Provision resources for the pool."""
        from skyward.pool_compute import _PoolCompute

        # Create a Compute-like object that the provider expects
        compute = _PoolCompute(
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

        # Provision and setup (providers use emit() directly)
        instances = self.provider.provision(compute)
        self.provider.setup(instances, compute)
        self._instances = instances

        # Initialize TaskPool (creates tunnels and connections in parallel)
        self._task_pool = TaskPool(
            provider=self.provider,
            instances=instances,
            concurrency=self.concurrency,
        )

        # Setup cluster on all instances in parallel (once per instance)
        self._setup_all_instances()

        # Start metrics polling if enabled
        if self.collect_metrics:
            for instance in instances:
                poller = MetricsPoller(instance, callback=self._event_callback)
                poller.start()
                self._metrics_pollers.append(poller)

    def _shutdown(self) -> None:
        """Shutdown and release pool resources."""
        if self._instances is None:
            return

        from skyward.pool_compute import _PoolCompute

        # Stop metrics polling
        for poller in self._metrics_pollers:
            poller.stop()
        self._metrics_pollers.clear()

        # Close TaskPool first if active
        if self._task_pool is not None:
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

        with suppress(Exception):
            self.provider.shutdown(self._instances, compute)

        self._instances = None

    def _setup_all_instances(self) -> None:
        """Setup cluster environment on all instances in parallel."""
        import time as _time

        from skyward.events import ClusterSetupCompleted

        assert self._task_pool is not None
        assert self._instances is not None

        t0 = _time.perf_counter()

        # Acquire one connection per instance (first N connections are interleaved)
        conns = [self._task_pool.acquire() for _ in range(len(self._instances))]

        try:
            # Setup in parallel
            def setup(pc: PooledConnection) -> None:
                self._setup_cluster(pc)

            for_each_async(setup, conns)
        finally:
            for conn in conns:
                self._task_pool.release(conn)

        emit(ClusterSetupCompleted(
            instance_count=len(self._instances),
            duration_seconds=_time.perf_counter() - t0,
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

        # Cost tracking (always active)
        callbacks.append(
            cost_tracker(
                region=getattr(self.provider, "region", "us-east-1"),
                provider=getattr(self.provider, "name", "aws"),
            )
        )

        # Display callback
        match self.display:
            case "log":
                callbacks.append(log)
            case "spinner":
                callbacks.append(spinner())
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
        return (
            f"Pool(provider={self.provider.name}, nodes={self.nodes}, "
            f"accelerator={self.accelerator}, {status})"
        )
