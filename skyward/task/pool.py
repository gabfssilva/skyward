"""Task pool for managing RPyC connections across instances."""

from __future__ import annotations

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

from skyward.core.constants import RPYC_PORT
from skyward.internal.object_pool import ObjectPool

if TYPE_CHECKING:
    from skyward.providers.ssh import ChannelStream, SSHConnection
    from skyward.types import Instance, Instances


# =============================================================================
# Pooled Connection
# =============================================================================


class _RPyCNotReadyError(Exception):
    """RPyC not connected yet - retry."""


@dataclass
class PooledConnection:
    """Pooled connection with its instance and SSH channel.

    Uses Paramiko direct-tcpip channel (no subprocess tunnel).
    Connection is established eagerly at construction time.
    """

    instance: Instance
    node_idx: int
    _ssh_conn: SSHConnection = field(init=False, repr=False)
    _channel: ChannelStream = field(init=False, repr=False)
    _conn: rpyc.Connection = field(init=False, repr=False)
    _bg_thread: rpyc.BgServingThread = field(init=False, repr=False)

    def __post_init__(self) -> None:
        logger.debug(f"Connecting to instance {self.instance.id} via SSH channel")
        # Acquire SSH connection and open channel
        self._ssh_conn = self.instance.pool.acquire()
        self._channel = self._ssh_conn.open_tunnel(RPYC_PORT)
        self._conn = self._connect_rpyc()
        # Start background thread to handle server callbacks (e.g., stdout)
        self._bg_thread = rpyc.BgServingThread(self._conn)
        logger.info(f"Connected to instance {self.instance.id}")

    @property
    def instance_id(self) -> str:
        return self.instance.id

    @property
    def conn(self) -> rpyc.Connection:
        """The RPyC connection."""
        return self._conn

    def _connect_rpyc(self) -> rpyc.Connection:
        """Connect to RPyC server via channel with retry."""
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
                conn = rpyc.connect_stream(
                    self._channel,
                    config={
                        "allow_pickle": True,
                        "allow_public_attrs": True,
                        "sync_request_timeout": 3600,
                    },
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
        """Close the connection and release SSH."""
        from contextlib import suppress

        logger.debug(f"Closing connection to {self.instance.id}")
        for cleanup in [
            lambda: self._bg_thread.stop(),
            lambda: setattr(self._conn, "_closed", True),
            lambda: self._channel.close(),
            lambda: self.instance.pool.release(self._ssh_conn),
        ]:
            with suppress(Exception):
                cleanup()
        logger.debug(f"Connection closed for {self.instance.id}")

# =============================================================================
# Connection Spec (for ObjectPool creation)
# =============================================================================


@dataclass(frozen=True)
class _ConnectionSpec:
    """Specification for creating a PooledConnection."""

    instance: Instance
    node_idx: int


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
            instances=instance_pool,  # Implements Instances protocol
            concurrency=4,
        )

        conn = pool.acquire()
        try:
            result = conn.conn.root.execute(...)
        finally:
            pool.release(conn)
    """

    instances: Instances
    concurrency: int
    _pool: ObjectPool[PooledConnection] = field(init=False, repr=False)
    # Timing data for PoolReady event (consolidated in pool.py)
    connection_count: int = field(init=False, repr=False)
    init_duration_seconds: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        t0 = time.perf_counter()
        n = len(self.instances)
        logger.info(f"Initializing task pool: {n} instances, concurrency={self.concurrency}")

        # Build specs interleaved: (node0, conn0), (node1, conn0), ..., (nodeN, connM)
        # This distributes load across nodes when acquiring connections
        specs: list[_ConnectionSpec] = []
        for _ in range(self.concurrency):
            for node_idx, instance in enumerate(self.instances):
                specs.append(_ConnectionSpec(instance, node_idx))

        # Create all connections in parallel via ObjectPool
        # Each PooledConnection creates its own SSH channel internally
        logger.debug(f"Creating {len(specs)} connections...")

        def create(i: int) -> PooledConnection:
            spec = specs[i]
            return PooledConnection(spec.instance, spec.node_idx)

        def close(pc: PooledConnection) -> None:
            pc.close()

        def check(pc: PooledConnection) -> bool:
            return pc.is_alive()

        self._pool = ObjectPool(
            create=create,
            close=close,
            check=check,
            max_size=len(specs),
            min_size=len(specs),  # Pre-warm all RPyC connections
            health_interval=10.0,
        )

        duration = time.perf_counter() - t0
        logger.info(f"Task pool ready: {len(specs)} connections in {duration:.1f}s")

        # Store timing data for PoolReady event (emitted by pool.py)
        self.connection_count = len(specs)
        self.init_duration_seconds = duration

    def acquire(self) -> PooledConnection:
        """Acquire connection from pool (blocks if empty)."""
        return self._pool.acquire()

    def release(self, conn: PooledConnection) -> None:
        """Return connection to the pool."""
        self._pool.release(conn)

    def close_all(self) -> None:
        """Close all connections."""
        logger.debug("Closing all connections...")
        self._pool.close_all()
        logger.debug("All connections closed")

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
