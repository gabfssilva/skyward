"""Task pool for managing RPyC connections across instances."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import rpyc
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
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            pass


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
        self._conn = self._connect_rpyc()

    @property
    def instance_id(self) -> str:
        return self.instance.id

    @property
    def conn(self) -> rpyc.Connection:
        """The RPyC connection."""
        return self._conn

    def _connect_rpyc(self) -> rpyc.Connection:
        """Connect to RPyC server with retry."""

        @retry(
            stop=stop_after_delay(60),
            wait=wait_fixed(0.5),
            retry=retry_if_exception_type(_RPyCNotReadyError),
            reraise=True,
        )
        def _connect() -> rpyc.Connection:
            try:
                conn = rpyc.connect(
                    "127.0.0.1",
                    self.tunnel.local_port,
                    config={"allow_pickle": True, "sync_request_timeout": 3600},
                )
                if conn.root.ping() == "pong":
                    return conn
            except Exception:
                pass
            raise _RPyCNotReadyError()

        try:
            return _connect()
        except RetryError as e:
            raise ConnectionError(f"Failed to connect: {e}") from e

    def is_alive(self) -> bool:
        """Check if connection is still alive."""
        try:
            return self._conn.root.ping() == "pong"
        except Exception:
            return False

    def close(self) -> None:
        """Close the connection."""
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

    def __post_init__(self) -> None:
        from skyward.conc import map_async
        from skyward.events import TaskPoolInitCompleted, emit

        t0 = time.perf_counter()

        # 1. Create all tunnels in parallel (one per instance)
        def create_tunnel(instance: Instance) -> Tunnel:
            local_port, proc = self.provider.create_tunnel(instance, RPYC_PORT)
            return Tunnel(proc, local_port, instance.id)

        tunnels = dict(zip(
            [inst.id for inst in self.instances],
            map_async(create_tunnel, self.instances),
        ))

        # 2. Build specs interleaved: (node0, conn0), (node1, conn0), ..., (nodeN, connM)
        # This distributes load across nodes when acquiring connections
        specs: list[_ConnectionSpec] = []
        for _ in range(self.concurrency):
            for node_idx, instance in enumerate(self.instances):
                specs.append(_ConnectionSpec(instance, node_idx, tunnels[instance.id]))

        # 3. Create all connections in parallel via ObjectPool
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

        emit(TaskPoolInitCompleted(
            tunnels=len(tunnels),
            connections=len(specs),
            duration_seconds=time.perf_counter() - t0,
        ))

    def acquire(self) -> PooledConnection:
        """Acquire connection from pool (blocks if empty)."""
        return self._pool.acquire()

    def release(self, conn: PooledConnection) -> None:
        """Return connection to the pool."""
        self._pool.release(conn)

    def close_all(self) -> None:
        """Close all connections and tunnels."""
        self._pool.close_all()
        for tunnel in self._tunnels.values():
            tunnel.close()

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
