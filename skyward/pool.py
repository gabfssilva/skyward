"""Pool - Resource management for skyward.

A Pool manages provisioned cloud resources (instances) and executes
computations on them. It separates environment configuration from
function definition, allowing multiple functions to share the same
resources.

Example:
    from skyward import compute, Pool, AWS

    @compute
    def train(data):
        return model.fit(data)

    @compute
    def evaluate(model):
        return model.evaluate()

    # Create a pool with GPU and PyTorch
    pool = Pool(
        provider=AWS(region="us-east-1"),
        accelerator="A100",
        pip=["torch", "transformers"],
    )

    with pool:
        # Both functions share the same provisioned resources
        model = train(data) | pool
        score = evaluate(model) | pool
"""

from __future__ import annotations

import contextlib
import contextvars
import subprocess
import threading
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Callable

import rpyc
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_fixed,
)

from skyward.accelerator import Accelerator
from skyward.callback import Callback, compose, emit, use_callback
from skyward.callbacks import cost_tracker, log, spinner
from skyward.conc import map_async
from skyward.events import Error, LogLine, PoolStarted, PoolStopping, SkywardEvent
from skyward.exceptions import ConnectionLostError, ExecutionError, NotProvisionedError
from skyward.pending import PendingBatch, PendingCompute
from skyward.spec import SpotLike
from skyward.types import Instance, Provider
from skyward.volume import Volume, parse_volume_uri

if TYPE_CHECKING:
    pass


class _RPyCNotReadyError(Exception):
    """RPyC not connected - retry."""


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
        nodes: Number of nodes to provision. Default 1.
        accelerator: Accelerator type ("A100", "H100", etc.) or None for CPU-only.
        python: Python version to use.
        pip: pip packages to install.
        pip_extra_index_url: Extra pip index URL.
        apt: apt packages to install.
        volume: Volumes to mount.
        spot: Spot strategy.

    Example:
        pool = Pool(
            provider=AWS(),
            accelerator="A100",
            pip=["torch", "transformers"],
        )

        with pool:
            result = train(data) | pool
    """

    # Required
    provider: Provider

    # Resource specification
    nodes: int = 1
    accelerator: Accelerator | list[Accelerator] = None
    cpu: int | None = None
    memory: str | None = None
    python: str = "3.13"
    pip: Sequence[str] = ()
    pip_extra_index_url: str | None = None
    apt: Sequence[str] = ()
    volume: dict[str, str] | Sequence[Volume] | None = None
    spot: SpotLike = "if-available"
    timeout: int = 3600
    env: dict[str, str] | None = None

    # Display settings
    display: Literal["spinner", "log", "quiet"] = "log"
    on_event: Callback | None = None

    # Internal state
    _active: bool = field(default=False, init=False, repr=False)
    _instances: tuple[Instance, ...] | None = field(default=None, init=False, repr=False)
    _callback_ctx: AbstractContextManager[None] | None = field(
        default=None, init=False, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _conn_locks: dict[str, threading.Lock] = field(default_factory=dict, init=False, repr=False)
    _tunnels: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict, init=False, repr=False)
    _rpyc_conns: dict[str, rpyc.Connection] = field(default_factory=dict, init=False, repr=False)
    _cluster_setup: set[str] = field(default_factory=set, init=False, repr=False)
    _job_id: str = field(default="", init=False, repr=False)

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

        Args:
            pending: PendingCompute to execute.

        Returns:
            The result of the computation.

        Raises:
            NotProvisionedError: If pool is not active.
        """
        if not self._active or self._instances is None:
            raise NotProvisionedError()

        # Use first available instance (simple scheduling)
        instance = self._instances[0]
        return self._execute_on_instance(instance, pending, self._stdout_cb(emit))

    def run_batch(self, batch: PendingBatch) -> tuple[Any, ...]:
        """Execute a batch of computations in parallel.

        Args:
            batch: PendingBatch containing computations to execute.

        Returns:
            Tuple of results in the same order as the computations.

        Raises:
            NotProvisionedError: If pool is not active.
        """
        if not self._active or self._instances is None:
            raise NotProvisionedError()

        # Distribute computations across instances (round-robin)
        n_instances = len(self._instances)

        stdout_cb = self._stdout_cb(emit)

        def execute_indexed(idx: int, pending: PendingCompute[Any]) -> Any:
            instance = self._instances[idx % n_instances]  # type: ignore
            return self._execute_on_instance(instance, pending, stdout_cb)

        # Execute in parallel
        results = list(
            map_async(
                lambda item: execute_indexed(item[0], item[1]),
                list(enumerate(batch.computations)),
            )
        )
        return tuple(results)

    def broadcast[R](self, pending: PendingCompute[R]) -> tuple[R, ...]:
        """Broadcast: execute computation on ALL nodes in parallel.

        Unlike run() which executes on one node, broadcast() executes the
        same computation on every node in the pool simultaneously.

        Args:
            pending: PendingCompute to broadcast.

        Returns:
            Tuple of results, one per node.

        Raises:
            NotProvisionedError: If pool is not active.

        Example:
            # Initialize model on all 4 nodes
            results = load_model(path) @ pool  # tuple of 4 results
        """
        if not self._active or self._instances is None:
            raise NotProvisionedError()

        stdout_cb = self._stdout_cb(emit)

        results = list(
            map_async(
                lambda inst: self._execute_on_instance(inst, pending, stdout_cb),
                self._instances,
            )
        )
        return tuple(results)

    def _execute_on_instance[R](
        self,
        instance: Instance,
        pending: PendingCompute[R],
        stdout_cb: Callable[[int, Instance], Callable[[str], None]],
    ) -> R:
        """Execute a computation on a specific instance."""
        conn = self._get_rpyc_connection(instance)
        node = self._instances.index(instance) if self._instances else 0

        # Serialize and send to remote
        from skyward.serialization import deserialize, serialize

        fn_bytes = serialize(pending.fn)
        args_bytes = serialize(pending.args)
        kwargs_bytes = serialize(pending.kwargs_dict)

        try:
            result_bytes = conn.root.execute(fn_bytes, args_bytes, kwargs_bytes, stdout_cb(node, instance))
            response = deserialize(result_bytes)
            if response.get("error"):
                raise ExecutionError(response["error"])
            return response["result"]
        except ExecutionError:
            raise
        except Exception as e:
            emit(Error(message=f"Execution failed on {instance.id}: {e}"))
            raise

    def _provision(self) -> None:
        """Provision resources for the pool."""
        from skyward.pool_compute import _PoolCompute

        # Create a Compute-like object that the provider expects
        compute = _PoolCompute(
            pool=self,
            fn=lambda: None,  # Placeholder
            nodes=self.nodes,
            accelerator=self.accelerator,
            cpu=self.cpu,
            memory=self.memory,
            python=self.python,
            pip=tuple(self.pip),
            pip_extra_index_url=self.pip_extra_index_url,
            apt=tuple(self.apt),
            env=frozenset(self.env.items()) if self.env else frozenset(),
            timeout=self.timeout,
            spot=self.spot,
            volumes=_parse_volumes(self.volume),
        )

        # Provision and setup (providers use emit() directly)
        instances = self.provider.provision(compute)
        self.provider.setup(instances, compute)
        self._instances = instances

    def _stdout_cb(
        self,
        do_emit: Callable[[Any], None],
    ):
        return lambda node, instance: lambda line: do_emit(
            LogLine(
                node=node,
                instance_id=instance.id,
                line=line,
                timestamp=time.time(),
            ),
        )

    def _shutdown(self) -> None:
        """Shutdown and release pool resources."""
        if self._instances is None:
            return

        from skyward.pool_compute import _PoolCompute

        compute = _PoolCompute(
            pool=self,
            fn=lambda: None,
            nodes=self.nodes,
            accelerator=self.accelerator,
            cpu=self.cpu,
            memory=self.memory,
            python=self.python,
            pip=tuple(self.pip),
            pip_extra_index_url=self.pip_extra_index_url,
            apt=tuple(self.apt),
            env=frozenset(),
            timeout=self.timeout,
            spot=self.spot,
            volumes=_parse_volumes(self.volume),
        )

        try:
            self.provider.shutdown(self._instances, compute)
        except Exception:
            pass  # Shutdown errors are not critical

        # Close connections
        self._close_connections()
        self._instances = None

    def _get_instance_lock(self, key: str) -> threading.Lock:
        """Get or create lock for specific instance."""
        if key not in self._conn_locks:
            with self._lock:
                if key not in self._conn_locks:
                    self._conn_locks[key] = threading.Lock()
        return self._conn_locks[key]

    def _get_rpyc_connection(self, instance: Instance) -> rpyc.Connection:
        """Get or create RPyC connection for instance."""
        key = instance.id
        if key in self._rpyc_conns:
            return self._rpyc_conns[key]

        with self._get_instance_lock(key):
            if key in self._rpyc_conns:
                return self._rpyc_conns[key]

            local_port, proc, conn = self._connect_with_retry(instance)
            self._tunnels[key] = proc
            self._rpyc_conns[key] = conn

            # Setup cluster environment on first connection
            node = self._instances.index(instance)  # type: ignore[union-attr]
            self._setup_cluster_on_instance(node, instance, conn)

            return conn

    def _setup_cluster_on_instance(
        self, node: int, instance: Instance, conn: rpyc.Connection
    ) -> None:
        """Setup cluster environment on instance (called once per node)."""
        if instance.id in self._cluster_setup:
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
        from skyward.providers.common.accelerator_detection import resolve_accelerator

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
        )

        env_bytes = serialize(self.env or {})
        conn.root.setup_cluster(pool_info.model_dump_json(), env_bytes)
        self._cluster_setup.add(instance.id)

    def _connect_with_retry(
        self, instance: Instance
    ) -> tuple[int, subprocess.Popen[bytes], rpyc.Connection]:
        """Create tunnel and verify RPyC connection with retry."""
        # Create tunnel
        local_port, proc = self.provider.create_tunnel(instance)

        try:
            @retry(
                stop=stop_after_delay(60),
                wait=wait_fixed(0.5),
                retry=retry_if_exception_type(_RPyCNotReadyError),
                reraise=True,
            )
            def _connect_rpyc() -> rpyc.Connection:
                try:
                    conn = rpyc.connect(
                        "127.0.0.1",
                        local_port,
                        config={"allow_pickle": True, "sync_request_timeout": 3600},
                    )
                    if conn.root.ping() == "pong":
                        return conn
                except Exception:
                    pass
                raise _RPyCNotReadyError()

            conn = _connect_rpyc()
            return local_port, proc, conn

        except RetryError as e:
            proc.terminate()
            raise ConnectionLostError(
                instance.id,
                f"Failed to establish RPyC connection after 60s: {e.last_attempt.exception()}",
            ) from e
        except Exception:
            proc.terminate()
            raise

    def _close_connections(self) -> None:
        """Close all RPyC connections and tunnels."""
        for conn in self._rpyc_conns.values():
            with contextlib.suppress(Exception):
                conn.close()

        for proc in self._tunnels.values():
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=5)

        self._rpyc_conns.clear()
        self._tunnels.clear()
        self._conn_locks.clear()

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
