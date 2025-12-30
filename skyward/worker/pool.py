"""Worker pool for managing multiple RPyC connections per instance.

Provides an acquire/release pattern for distributing work across
isolated workers within instances.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
from collections import deque
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

if TYPE_CHECKING:
    from skyward.types import Instance, Provider


class _RPyCNotReadyError(Exception):
    """RPyC not connected - retry."""


@dataclass(frozen=True, slots=True)
class Worker:
    """Represents a single worker on an instance.

    Attributes:
        instance_id: Parent instance ID.
        worker_id: Worker ID within instance (0-indexed).
        port: RPyC port for this worker.
    """

    instance_id: str
    worker_id: int
    port: int

    @property
    def key(self) -> str:
        """Unique key for this worker."""
        return f"{self.instance_id}:{self.worker_id}"


@dataclass
class WorkerPool:
    """Pool of workers with acquire/release pattern.

    Manages RPyC connections to multiple workers across instances.
    Workers are acquired for execution and released when done.

    Example:
        pool = WorkerPool(provider)
        pool.register_instance(instance, worker_count=4)

        worker = pool.acquire()
        try:
            conn = pool.get_connection(worker)
            result = conn.root.execute(fn_bytes, ...)
        finally:
            pool.release(worker)
    """

    provider: Provider

    # Worker tracking
    _available: dict[str, deque[Worker]] = field(default_factory=dict, repr=False)
    _in_use: set[str] = field(default_factory=set, repr=False)

    # Instance storage (for tunnel creation)
    _instances: dict[str, Instance] = field(default_factory=dict, repr=False)

    # Connections
    _conns: dict[str, rpyc.Connection] = field(default_factory=dict, repr=False)
    _tunnels: dict[str, subprocess.Popen] = field(default_factory=dict, repr=False)

    # Locks
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _conn_locks: dict[str, threading.Lock] = field(default_factory=dict, repr=False)

    def register_instance(
        self,
        instance: Instance,
        worker_count: int,
        base_port: int = RPYC_PORT,
    ) -> None:
        """Register workers for an instance.

        Args:
            instance: Instance to register.
            worker_count: Number of workers on this instance.
            base_port: Base port for workers (port = base_port + worker_id).
        """
        with self._lock:
            workers = deque(
                Worker(
                    instance_id=instance.id,
                    worker_id=i,
                    port=base_port + i,
                )
                for i in range(worker_count)
            )
            self._available[instance.id] = workers
            self._instances[instance.id] = instance

    def acquire(self, n: int = 1) -> tuple[Worker, ...]:
        """Acquire workers for execution.

        Blocks until n workers are available.

        Args:
            n: Number of workers to acquire.

        Returns:
            Tuple of acquired Workers.

        Raises:
            RuntimeError: If pool is empty (no instances registered).
        """
        workers: list[Worker] = []

        with self._lock:
            # Check if we have any workers at all
            total_available = sum(len(q) for q in self._available.values())
            if total_available == 0 and not self._in_use:
                raise RuntimeError("No workers available - no instances registered")

        # Acquire workers one at a time
        for _ in range(n):
            worker = self._acquire_one()
            workers.append(worker)

        return tuple(workers)

    def _acquire_one(self) -> Worker:
        """Acquire a single worker, blocking until available."""
        while True:
            with self._lock:
                # Try each instance's queue
                for instance_id, queue in self._available.items():
                    if queue:
                        worker = queue.popleft()
                        self._in_use.add(worker.key)
                        return worker

            # No workers available, wait and retry
            # In a real implementation, use a condition variable
            import time
            time.sleep(0.01)

    def release(self, *workers: Worker) -> None:
        """Release workers back to the pool.

        Args:
            workers: Workers to release.
        """
        with self._lock:
            for worker in workers:
                if worker.key in self._in_use:
                    self._in_use.discard(worker.key)
                    if worker.instance_id in self._available:
                        self._available[worker.instance_id].append(worker)

    def get_connection(self, worker: Worker) -> rpyc.Connection:
        """Get or create RPyC connection for worker.

        Args:
            worker: Worker to connect to.

        Returns:
            RPyC connection.

        Raises:
            ConnectionError: If connection fails.
        """
        key = worker.key

        # Fast path: existing connection
        if key in self._conns:
            conn = self._conns[key]
            # Check if connection is still alive
            try:
                if conn.root.ping() == "pong":
                    return conn
            except Exception:
                pass
            # Connection dead, clean up
            self._cleanup_connection(key)

        # Slow path: create new connection
        lock = self._get_conn_lock(key)
        with lock:
            # Double-check after acquiring lock
            if key in self._conns:
                return self._conns[key]

            # Get parent instance
            instance = self._get_instance_for_worker(worker)
            if instance is None:
                raise RuntimeError(f"Instance not found for worker {worker.key}")

            # Create tunnel and connection
            local_port, proc, conn = self._connect_with_retry(instance, worker.port)
            self._tunnels[key] = proc
            self._conns[key] = conn

            return conn

    def _get_instance_for_worker(self, worker: Worker) -> Instance | None:
        """Get Instance object for a worker."""
        return self._instances.get(worker.instance_id)

    def _get_conn_lock(self, key: str) -> threading.Lock:
        """Get or create lock for connection key."""
        if key not in self._conn_locks:
            with self._lock:
                if key not in self._conn_locks:
                    self._conn_locks[key] = threading.Lock()
        return self._conn_locks[key]

    def _connect_with_retry(
        self,
        instance: Instance,
        remote_port: int,
    ) -> tuple[int, subprocess.Popen, rpyc.Connection]:
        """Create tunnel and connect with retry.

        Args:
            instance: Instance to connect to.
            remote_port: Remote RPyC port.

        Returns:
            Tuple of (local_port, tunnel_process, rpyc_connection).
        """
        # Create tunnel to specific port
        local_port, proc = self.provider.create_tunnel(instance, remote_port)

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
            raise ConnectionError(
                f"Failed to connect to worker at port {remote_port}: {e}"
            ) from e
        except Exception:
            proc.terminate()
            raise

    def _cleanup_connection(self, key: str) -> None:
        """Clean up a dead connection."""
        with contextlib.suppress(Exception):
            if key in self._conns:
                self._conns[key].close()
                del self._conns[key]
        with contextlib.suppress(Exception):
            if key in self._tunnels:
                self._tunnels[key].terminate()
                self._tunnels[key].wait(timeout=5)
                del self._tunnels[key]

    def close_all(self) -> None:
        """Close all connections and tunnels."""
        with self._lock:
            for conn in self._conns.values():
                with contextlib.suppress(Exception):
                    conn.close()

            for proc in self._tunnels.values():
                with contextlib.suppress(Exception):
                    proc.terminate()
                    proc.wait(timeout=5)

            self._conns.clear()
            self._tunnels.clear()
            self._available.clear()
            self._in_use.clear()
            self._conn_locks.clear()
            self._instances.clear()

    @property
    def total_workers(self) -> int:
        """Total number of workers (available + in use)."""
        with self._lock:
            available = sum(len(q) for q in self._available.values())
            return available + len(self._in_use)

    @property
    def available_workers(self) -> int:
        """Number of available workers."""
        with self._lock:
            return sum(len(q) for q in self._available.values())

    def __repr__(self) -> str:
        return (
            f"WorkerPool(total={self.total_workers}, "
            f"available={self.available_workers}, "
            f"in_use={len(self._in_use)})"
        )
