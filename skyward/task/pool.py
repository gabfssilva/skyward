"""Task pool for managing RPyC connections across instances."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import rpyc
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_fixed,
)

from skyward.constants import RPYC_PORT
from skyward.task.object_pool import ObjectPool

if TYPE_CHECKING:
    from skyward.types import Instance, Provider


# =============================================================================
# Tunnel
# =============================================================================


@dataclass
class Tunnel:
    """SSH/SSM tunnel to an instance."""

    proc: subprocess.Popen[bytes]
    local_port: int
    instance_id: str

    def close(self) -> None:
        """Close the tunnel."""
        import signal

        if self.proc.poll() is not None:
            return  # Already dead

        logger.debug(f"Closing tunnel for {self.instance_id} (port {self.local_port})")

        # Try graceful termination first
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass

        # SSH -N ignores SIGTERM - force kill
        logger.debug(f"Force killing tunnel process for {self.instance_id}")
        self.proc.send_signal(signal.SIGKILL)
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass  # Best effort - process is orphaned


# =============================================================================
# Pooled Connection
# =============================================================================


class _RPyCNotReadyError(Exception):
    """RPyC not connected yet - retry."""


@dataclass
class PooledConnection:
    """Pooled connection with its instance and tunnel.

    Connection is established eagerly at construction time.
    """

    instance: Instance
    node_idx: int
    tunnel: Tunnel
    _conn: rpyc.Connection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        logger.debug(f"Connecting to instance {self.instance.id} via port {self.tunnel.local_port}")
        self._conn = self._connect_rpyc()
        logger.info(f"Connected to instance {self.instance.id}")

    @property
    def instance_id(self) -> str:
        return self.instance.id

    @property
    def conn(self) -> rpyc.Connection:
        """The RPyC connection."""
        return self._conn

    def _connect_rpyc(self) -> rpyc.Connection:
        """Connect to RPyC server with retry."""
        attempt = 0

        @retry(
            stop=stop_after_delay(60),
            wait=wait_fixed(0.5),
            retry=retry_if_exception_type(_RPyCNotReadyError),
            reraise=True,
        )
        def _connect() -> rpyc.Connection:
            nonlocal attempt
            attempt += 1
            try:
                conn = rpyc.connect(
                    "127.0.0.1",
                    self.tunnel.local_port,
                    config={"allow_pickle": True, "sync_request_timeout": 3600},
                )
                if conn.root.ping() == "pong":
                    logger.debug(f"RPyC connected after {attempt} attempts")
                    return conn
            except Exception:
                pass
            if attempt % 10 == 0:
                logger.debug(f"RPyC connection attempt {attempt} for {self.instance.id}")
            raise _RPyCNotReadyError()

        try:
            return _connect()
        except RetryError as e:
            logger.error(f"Failed to connect to {self.instance.id} after {attempt} attempts")
            raise ConnectionError(f"Failed to connect: {e}") from e

    def is_alive(self) -> bool:
        """Check if connection is still alive."""
        try:
            return self._conn.root.ping() == "pong"
        except Exception:
            return False

    def close(self) -> None:
        """Close the connection."""
        logger.debug(f"Closing connection to {self.instance.id}")
        try:
            self._conn.close()
        except Exception:
            pass

# =============================================================================
# Connection Spec (for ObjectPool creation)
# =============================================================================


@dataclass(frozen=True)
class _ConnectionSpec:
    """Specification for creating a PooledConnection."""

    instance: Instance
    node_idx: int
    tunnel: Tunnel


# =============================================================================
# Task Pool
# =============================================================================

@dataclass
class TaskPool:
    """Pool of connections for task execution.

    Uses ObjectPool for thread-safe acquire/release with parallel creation.
    Tunnels and connections are all created in parallel.

    Example:
        pool = TaskPool(
            provider=provider,
            instances=instances,
            concurrency=4,
        )

        conn = pool.acquire()
        try:
            result = conn.conn.root.execute(...)
        finally:
            pool.release(conn)
    """

    provider: Provider
    instances: tuple[Instance, ...]
    concurrency: int
    _pool: ObjectPool[PooledConnection] = field(init=False, repr=False)
    # Timing data for PoolReady event (consolidated in pool.py)
    tunnel_count: int = field(init=False, repr=False)
    connection_count: int = field(init=False, repr=False)
    init_duration_seconds: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from skyward.conc import map_async

        t0 = time.perf_counter()
        logger.info(f"Initializing task pool: {len(self.instances)} instances, concurrency={self.concurrency}")

        # 1. Create all tunnels in parallel (one per instance)
        logger.debug(f"Creating {len(self.instances)} tunnels...")

        def create_tunnel(instance: Instance) -> Tunnel:
            logger.debug(f"Creating tunnel for {instance.id}")
            local_port, proc = self.provider.create_tunnel(instance, RPYC_PORT)
            logger.debug(f"Tunnel created for {instance.id} -> localhost:{local_port}")
            return Tunnel(proc, local_port, instance.id)

        tunnels = dict(zip(
            [inst.id for inst in self.instances],
            map_async(create_tunnel, self.instances),
        ))
        logger.debug(f"All {len(tunnels)} tunnels created")

        # 2. Build specs interleaved: (node0, conn0), (node1, conn0), ..., (nodeN, connM)
        # This distributes load across nodes when acquiring connections
        specs: list[_ConnectionSpec] = []
        for _ in range(self.concurrency):
            for node_idx, instance in enumerate(self.instances):
                specs.append(_ConnectionSpec(instance, node_idx, tunnels[instance.id]))

        # 3. Create all connections in parallel via ObjectPool
        logger.debug(f"Creating {len(specs)} connections...")

        def create(i: int) -> PooledConnection:
            spec = specs[i]
            return PooledConnection(spec.instance, spec.node_idx, spec.tunnel)

        def close(pc: PooledConnection) -> None:
            pc.close()

        def check(pc: PooledConnection) -> bool:
            return pc.is_alive()

        self._pool = ObjectPool(
            size=len(specs),
            create=create,
            close=close,
            check=check,
            interval=10.0,
        )

        # Store tunnels for cleanup
        self._tunnels = tunnels

        duration = time.perf_counter() - t0
        logger.info(f"Task pool ready: {len(specs)} connections in {duration:.1f}s")

        # Store timing data for PoolReady event (emitted by pool.py)
        self.tunnel_count = len(tunnels)
        self.connection_count = len(specs)
        self.init_duration_seconds = duration

    def acquire(self) -> PooledConnection:
        """Acquire connection from pool (blocks if empty)."""
        return self._pool.acquire()

    def release(self, conn: PooledConnection) -> None:
        """Return connection to the pool."""
        self._pool.release(conn)

    def close_all(self) -> None:
        """Close all connections and tunnels."""
        logger.debug(f"Closing all connections and tunnels...")
        self._pool.close_all()
        for tunnel in self._tunnels.values():
            tunnel.close()
        logger.debug("All connections and tunnels closed")

    @property
    def total_slots(self) -> int:
        """Total number of slots (available + in use)."""
        return self._pool.total

    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return self._pool.available

    def __repr__(self) -> str:
        return f"TaskPool(total={self.total_slots}, available={self.available_slots})"
