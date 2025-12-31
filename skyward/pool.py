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
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Callable

from skyward.accelerator import Accelerator, MIG_MAX_INSTANCES
from skyward.callback import Callback, compose, emit, use_callback
from skyward.callbacks import cost_tracker, log, spinner
from skyward.conc import map_async
from skyward.events import Error, LogLine, PoolStarted, PoolStopping
from skyward.exceptions import ExecutionError, NotProvisionedError
from skyward.metrics import MetricsPoller
from skyward.pending import PendingBatch, PendingCompute
from skyward.image import DEFAULT_IMAGE, Image
from skyward.spec import SpotLike
from skyward.types import Instance, Provider
from skyward.volume import Volume, parse_volume_uri
from skyward.worker import (
    ResourceLimits,
    Worker,
    WorkerConfig,
    WorkerPool,
    create_partition,
    generate_worker_configs,
)

if TYPE_CHECKING:
    import rpyc


def _parse_volumes(
    volume: dict[str, str] | Sequence[Volume] | None
) -> tuple[Volume, ...]:
    """Parse volume specification into Volume objects."""
    if volume is None:
        return ()

    if isinstance(volume, Sequence) and not isinstance(volume, (str, dict)):
        return tuple(volume)

    if isinstance(volume, dict):
        return tuple(parse_volume_uri(mount_path, uri) for mount_path, uri in volume.items())

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
        volume: Volumes to mount.
        spot: Spot strategy.
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
    accelerator: Accelerator | str | None = None
    cpu: int | None = None
    memory: str | None = None
    volume: dict[str, str] | Sequence[Volume] | None = None
    spot: SpotLike = "always"
    timeout: int = 3600
    env: dict[str, str] | None = None

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
    _cluster_setup: set[str] = field(default_factory=set, init=False, repr=False)
    _job_id: str = field(default="", init=False, repr=False)

    # Worker isolation state
    _accelerator_cfg: Accelerator | None = field(default=None, init=False, repr=False)
    _worker_pool: WorkerPool | None = field(default=None, init=False, repr=False)
    _worker_configs: tuple[WorkerConfig, ...] | None = field(default=None, init=False, repr=False)
    _worker_limits: ResourceLimits | None = field(default=None, init=False, repr=False)
    _worker_partition_script: str = field(default="", init=False, repr=False)
    _metrics_pollers: list[MetricsPoller] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Parse accelerator and store configuration."""
        if self.accelerator is None:
            return

        # Convert to Accelerator if it's a string
        cfg = Accelerator.from_value(self.accelerator)
        if cfg is not None:
            object.__setattr__(self, "_accelerator_cfg", cfg)

    def _parse_accelerator_for_provider(self) -> str | None:
        """Get the accelerator type for the provider."""
        if self._accelerator_cfg is not None:
            return self._accelerator_cfg.accelerator
        return None

    @property
    def workers_per_instance(self) -> int:
        """Number of workers per instance."""
        if self._accelerator_cfg is None:
            return 1
        mig = self._accelerator_cfg.multiple_instance
        if mig is None:
            return 1
        if isinstance(mig, list):
            return self._accelerator_cfg.count * len(mig)
        return self._accelerator_cfg.count * MIG_MAX_INSTANCES.get(mig, 1)

    def _get_worker_isolation_data(
        self, device_count: int
    ) -> tuple[tuple[WorkerConfig, ...] | None, ResourceLimits | None, str]:
        """Get worker isolation configuration.

        Returns:
            Tuple of (worker_configs, resource_limits, partition_script).
            Returns (None, None, "") if no worker isolation needed.
        """
        workers = self.workers_per_instance
        if self._accelerator_cfg is None or workers <= 1:
            return None, None, ""

        # Create resource limits from cpu/memory
        limits = None
        if self.cpu is not None or self.memory is not None:
            limits = ResourceLimits.from_params(memory=self.memory, cpu=self.cpu)

        # Get partition strategy for this accelerator
        strategy = create_partition(
            accelerator=self._accelerator_cfg.accelerator,
            device_count=device_count,
            mig=self._accelerator_cfg.multiple_instance,
        )

        # Generate worker configs using partition strategy's env function
        def device_env_fn(worker_id: int) -> dict[str, str]:
            return strategy.get_worker_env(worker_id, device_count)

        configs = generate_worker_configs(
            worker_count=workers,
            device_env_fn=device_env_fn,
            limits=limits,
        )

        return configs, limits, strategy.setup_script

    def __enter__(self) -> ComputePool:
        """Enter pool context and provision resources."""
        import uuid

        self._active = True
        self._job_id = str(uuid.uuid4())[:8]

        # Setup callback context
        self._callback_ctx = use_callback(self._build_callback())
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

        Acquires the first available worker, executes, and releases.

        Args:
            pending: PendingCompute to execute.

        Returns:
            The result of the computation.

        Raises:
            NotProvisionedError: If pool is not active.
        """
        if not self._active or self._worker_pool is None:
            raise NotProvisionedError()

        workers = self._worker_pool.acquire(1)
        try:
            return self._execute_on_worker(workers[0], pending)
        finally:
            self._worker_pool.release(*workers)

    def run_batch(self, batch: PendingBatch) -> tuple[Any, ...]:
        """Execute a batch of computations in parallel.

        Distributes computations across available workers (round-robin).

        Args:
            batch: PendingBatch containing computations to execute.

        Returns:
            Tuple of results in the same order as the computations.

        Raises:
            NotProvisionedError: If pool is not active.
        """
        if not self._active or self._worker_pool is None:
            raise NotProvisionedError()

        computations = list(batch.computations)
        n_computations = len(computations)

        # Acquire as many workers as we need (up to available)
        n_workers = min(n_computations, self._worker_pool.total_workers)
        workers = self._worker_pool.acquire(n_workers)

        try:
            # Assign computations to workers (round-robin)
            def execute_on_worker(item: tuple[int, PendingCompute[Any]]) -> Any:
                idx, pending = item
                worker = workers[idx % n_workers]
                return self._execute_on_worker(worker, pending)

            results = list(
                map_async(
                    execute_on_worker,
                    list(enumerate(computations)),
                )
            )
            return tuple(results)
        finally:
            self._worker_pool.release(*workers)

    def broadcast[R](self, pending: PendingCompute[R]) -> tuple[R, ...]:
        """Broadcast: execute computation on ALL workers in parallel.

        Unlike run() which executes on one worker, broadcast() executes the
        same computation on every worker in the pool simultaneously.

        With MIG or multi-worker configurations, this broadcasts to all
        workers (not just instances). For example, with 1 instance and
        MIG="3g.40gb" (2 partitions), this broadcasts to 2 workers.

        Args:
            pending: PendingCompute to broadcast.

        Returns:
            Tuple of results, one per worker.

        Raises:
            NotProvisionedError: If pool is not active.

        Example:
            # Initialize model on all workers
            results = load_model(path) @ pool  # tuple of N results
        """
        if not self._active or self._worker_pool is None:
            raise NotProvisionedError()

        # Acquire all workers
        total_workers = self._worker_pool.total_workers
        workers = self._worker_pool.acquire(total_workers)

        try:
            results = list(
                map_async(
                    lambda w: self._execute_on_worker(w, pending),
                    workers,
                )
            )
            return tuple(results)
        finally:
            self._worker_pool.release(*workers)

    def _execute_on_worker[R](
        self,
        worker: Worker,
        pending: PendingCompute[R],
    ) -> R:
        """Execute a computation on a specific worker (via WorkerPool)."""
        assert self._worker_pool is not None
        assert self._instances is not None

        # Get connection from WorkerPool
        conn = self._worker_pool.get_connection(worker)

        # Get instance for this worker (for cluster setup and logging)
        instance = self._worker_pool._get_instance_for_worker(worker)
        if instance is None:
            raise RuntimeError(f"Instance not found for worker {worker.key}")

        # Calculate node index (instance position in cluster)
        node = next(
            (i for i, inst in enumerate(self._instances) if inst.id == instance.id),
            0,
        )

        # Setup cluster environment on first connection to this worker
        worker_key = f"{instance.id}:{worker.worker_id}"
        if worker_key not in self._cluster_setup:
            self._setup_cluster_on_worker(node, worker.worker_id, instance, conn)

        # Serialize and send to remote
        from skyward.serialization import deserialize, serialize

        fn_bytes = serialize(pending.fn)
        args_bytes = serialize(pending.args)
        kwargs_bytes = serialize(pending.kwargs_dict)

        # Create stdout callback for this worker
        def stdout_callback(line: str) -> None:
            emit(
                LogLine(
                    node=node,
                    instance_id=instance.id,
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
            emit(Error(message=f"Execution failed on worker {worker.key}: {e}"))
            raise

    def _provision(self) -> None:
        """Provision resources for the pool."""
        from skyward.pool_compute import _PoolCompute

        # Get accelerator type for provider (without worker count info)
        accelerator_for_provider = self._parse_accelerator_for_provider()

        # Determine device count (will be updated after provisioning)
        device_count = 1
        if self._accelerator_cfg is not None:
            device_count = self._accelerator_cfg.count

        # Get worker isolation data if needed
        configs, limits, partition_script = self._get_worker_isolation_data(device_count)
        object.__setattr__(self, "_worker_configs", configs)
        object.__setattr__(self, "_worker_limits", limits)
        object.__setattr__(self, "_worker_partition_script", partition_script)

        # Create a Compute-like object that the provider expects
        compute = _PoolCompute(
            pool=self,
            fn=lambda: None,  # Placeholder
            nodes=self.nodes,
            accelerator=self.accelerator,  # Keep full spec (e.g., 'L40S:2') for provider to parse
            image=self.image,
            cpu=self.cpu,
            memory=self.memory,
            timeout=self.timeout,
            spot=self.spot,
            volumes=list(_parse_volumes(self.volume)),
            _workers_per_instance=self.workers_per_instance,
            _worker_configs=configs,
            _worker_limits=limits,
            _worker_partition_script=partition_script,
        )

        # Provision and setup (providers use emit() directly)
        instances = self.provider.provision(compute)
        self.provider.setup(instances, compute)
        self._instances = instances

        # Always initialize WorkerPool (even with 1 worker per instance)
        self._worker_pool = WorkerPool(provider=self.provider)
        for instance in instances:
            self._worker_pool.register_instance(
                instance=instance,
                worker_count=self.workers_per_instance,
            )

        # Start metrics polling if enabled
        if self.collect_metrics:
            for instance in instances:
                poller = MetricsPoller(instance)
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

        # Close WorkerPool first if active
        if self._worker_pool is not None:
            try:
                self._worker_pool.close_all()
            except Exception:
                pass
            self._worker_pool = None

        # Get accelerator type for provider
        accelerator_for_provider = self._parse_accelerator_for_provider()

        compute = _PoolCompute(
            pool=self,
            fn=lambda: None,
            nodes=self.nodes,
            accelerator=accelerator_for_provider,
            image=self.image,
            cpu=self.cpu,
            memory=self.memory,
            timeout=self.timeout,
            spot=self.spot,
            volumes=list(_parse_volumes(self.volume)),
            _workers_per_instance=self.workers_per_instance,
            _worker_configs=self._worker_configs,
            _worker_limits=self._worker_limits,
            _worker_partition_script=self._worker_partition_script,
        )

        try:
            self.provider.shutdown(self._instances, compute)
        except Exception:
            pass  # Shutdown errors are not critical

        self._instances = None

    def _setup_cluster_on_worker(
        self,
        node: int,
        worker_id: int,
        instance: Instance,
        conn: rpyc.Connection,
    ) -> None:
        """Setup cluster environment on worker (called once per worker)."""
        worker_key = f"{instance.id}:{worker_id}"
        if worker_key in self._cluster_setup:
            return

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

        accelerator_type, accelerator_count = resolve_accelerator(instance)

        pool_info = build_pool_info(
            node=node,
            total_nodes=len(instances),
            accelerator_count=accelerator_count,
            total_accelerators=len(instances) * accelerator_count,
            head_addr=instances[0].private_ip,
            head_port=29500,
            job_id=self._job_id,
            peers=peers,
            accelerator_type=accelerator_type,
            worker=worker_id,
            workers_per_node=self.workers_per_instance,
        )

        # Merge env from image and pool (pool overrides image)
        merged_env = {**self.image.env, **(self.env or {})}
        env_bytes = serialize(merged_env)
        conn.root.setup_cluster(pool_info.model_dump_json(), env_bytes)
        self._cluster_setup.add(worker_key)

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
