"""RPyC-based remote function executor for v2.

Provides async execution of functions on remote instances via RPyC.
Uses SSH tunnel (local port forwarding) to access RPyC on localhost.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import asyncssh
import rpyc
from loguru import logger

if TYPE_CHECKING:
    pass


# RPyC server port (same as v1)
RPYC_PORT = 18861

# Events log path on remote (same as v1)
EVENTS_LOG = "/opt/skyward/events.jsonl"


class _RPyCNotReady(Exception):
    """RPyC not connected yet - retry."""


def _find_free_port() -> int:
    """Find an available local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class AsyncRPyCExecutor:
    """Async RPyC executor for remote function execution.

    Uses SSH tunnel (local port forwarding) to access RPyC server.
    The RPyC server runs on localhost:18861 on the remote machine,
    and we forward a local port to it via SSH.

    Example:
        executor = AsyncRPyCExecutor(
            host="10.0.0.1",
            user="ubuntu",
            key_path="~/.ssh/id_rsa",
        )

        async with executor:
            result = await executor.execute(fn, *args, **kwargs)
    """

    host: str
    user: str
    key_path: str
    ssh_port: int = 22
    remote_port: int = RPYC_PORT
    connect_timeout: float = 60.0

    _ssh_conn: asyncssh.SSHClientConnection | None = field(default=None, repr=False)
    _local_port: int = field(default=0, repr=False)
    _listener: asyncssh.SSHListener | None = field(default=None, repr=False)
    _conn: rpyc.Connection | None = field(default=None, repr=False)
    _bg_thread: rpyc.BgServingThread | None = field(default=None, repr=False)
    _log_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _log_callback: Callable[[str, str], None] | None = field(default=None, repr=False)

    async def connect(self) -> None:
        """Establish SSH tunnel and RPyC connection."""
        if self._conn is not None:
            return

        # Connect SSH
        self._ssh_conn = await asyncssh.connect(
            self.host,
            port=self.ssh_port,
            username=self.user,
            client_keys=[self.key_path],
            known_hosts=None,
            connect_timeout=self.connect_timeout,
        )

        # Create local port forwarding
        self._local_port = _find_free_port()
        self._listener = await self._ssh_conn.forward_local_port(
            "",  # Listen on all interfaces
            self._local_port,
            "127.0.0.1",  # Forward to localhost on remote
            self.remote_port,
        )

        logger.debug(f"SSH tunnel: localhost:{self._local_port} -> {self.host}:localhost:{self.remote_port}")

        # Wait for RPyC to be ready
        self._conn = await self._connect_rpyc()

        # Start background thread for callbacks
        self._bg_thread = rpyc.BgServingThread(self._conn)

        logger.debug(f"Connected to {self.host}")

    async def _connect_rpyc(self) -> rpyc.Connection:
        """Connect to RPyC server via local tunnel with retry."""
        start = asyncio.get_event_loop().time()
        attempt = 0

        while asyncio.get_event_loop().time() - start < self.connect_timeout:
            attempt += 1
            try:
                conn = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._connect_rpyc_sync,
                )
                return conn
            except _RPyCNotReady:
                if attempt % 10 == 0:
                    logger.debug(f"RPyC connection attempt {attempt} for {self.host}")
                await asyncio.sleep(0.5)

        raise TimeoutError(f"Could not connect to RPyC on {self.host}")

    def _connect_rpyc_sync(self) -> rpyc.Connection:
        """Sync RPyC connection via local tunnel (runs in executor)."""
        try:
            # Connect to local forwarded port
            conn = rpyc.connect(
                "127.0.0.1",
                port=self._local_port,
                config={
                    "allow_pickle": True,
                    "allow_public_attrs": True,
                    "sync_request_timeout": 3600,
                },
            )
            if conn.root.ping() == "pong":
                return conn
        except Exception:
            pass
        raise _RPyCNotReady()

    async def close(self) -> None:
        """Close RPyC connection and SSH tunnel."""
        logger.debug(f"Executor close starting for {self.host}")

        # Stop log streaming first
        self.stop_log_streaming()

        if self._bg_thread is not None:
            logger.debug("Stopping BgServingThread...")
            with suppress(Exception):
                self._bg_thread.stop()
            self._bg_thread = None
            logger.debug("BgServingThread stopped")

        if self._conn is not None:
            logger.debug("Closing RPyC connection...")
            conn = self._conn
            self._conn = None
            # RPyC close can hang waiting for remote ack, run with timeout
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, conn.close),
                    timeout=5.0,
                )
            except (TimeoutError, asyncio.TimeoutError):
                logger.debug("RPyC close timed out, continuing anyway")
            except Exception:
                pass
            logger.debug("RPyC connection closed")

        if self._listener is not None:
            logger.debug("Closing SSH listener...")
            with suppress(Exception):
                self._listener.close()
            self._listener = None
            logger.debug("SSH listener closed")

        if self._ssh_conn is not None:
            logger.debug("Closing SSH connection...")
            with suppress(Exception):
                self._ssh_conn.close()
            # Wait for close with timeout to avoid hanging
            logger.debug("Waiting for SSH connection to close (5s timeout)...")
            with suppress(Exception):
                await asyncio.wait_for(self._ssh_conn.wait_closed(), timeout=5.0)
            self._ssh_conn = None
            logger.debug("SSH connection closed")

        logger.debug(f"Executor close completed for {self.host}")

    async def __aenter__(self) -> AsyncRPyCExecutor:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    @property
    def is_connected(self) -> bool:
        """Whether connection is established."""
        return self._conn is not None

    def ping(self) -> bool:
        """Check if RPyC connection is alive."""
        if self._conn is None:
            return False
        try:
            return self._conn.root.ping() == "pong"
        except Exception:
            return False

    async def execute[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function remotely and return result.

        Args:
            fn: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of function execution.

        Raises:
            RuntimeError: If not connected.
            ExecutionError: If remote execution fails.
        """
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        from skyward.utils.serialization import deserialize, serialize

        # Serialize function and arguments (with compression)
        fn_bytes = serialize(fn)
        args_bytes = serialize(args)
        kwargs_bytes = serialize(kwargs)

        # Execute remotely (in executor since rpyc is sync)
        result_bytes = await asyncio.get_event_loop().run_in_executor(
            None,
            self._execute_sync,
            fn_bytes,
            args_bytes,
            kwargs_bytes,
        )

        # Deserialize result (handles compression automatically)
        response = deserialize(result_bytes)
        if response.get("error"):
            raise RuntimeError(f"Remote execution failed: {response['error']}")

        return response["result"]

    def _execute_sync(
        self,
        fn_bytes: bytes,
        args_bytes: bytes,
        kwargs_bytes: bytes,
    ) -> bytes:
        """Sync execution (runs in executor)."""
        return self._conn.root.execute(fn_bytes, args_bytes, kwargs_bytes)

    async def setup_cluster(
        self,
        pool_info_json: str,
        env_vars: dict[str, str],
    ) -> None:
        """Setup cluster environment on remote instance.

        Args:
            pool_info_json: JSON string for COMPUTE_POOL env var.
            env_vars: Additional environment variables.
        """
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        from skyward.utils.serialization import serialize

        env_bytes = serialize(env_vars)

        await asyncio.get_event_loop().run_in_executor(
            None,
            self._setup_cluster_sync,
            pool_info_json,
            env_bytes,
        )

    def _setup_cluster_sync(
        self,
        pool_info_json: str,
        env_bytes: bytes,
    ) -> str:
        """Sync setup_cluster (runs in executor)."""
        return self._conn.root.setup_cluster(pool_info_json, env_bytes)

    def start_log_streaming(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """Start streaming logs from remote events.jsonl.

        Args:
            callback: Function called with (content, stream) for each log line.
                      stream is "stdout" or "stderr".
        """
        if self._ssh_conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        if self._log_task is not None:
            return  # Already streaming

        self._log_callback = callback
        self._log_task = asyncio.create_task(self._stream_logs())

    async def _stream_logs(self) -> None:
        """Background task that streams logs from remote."""
        if self._ssh_conn is None:
            return

        try:
            # Run tail -F on events.jsonl
            process = await self._ssh_conn.create_process(
                f"tail -F {EVENTS_LOG} 2>/dev/null",
                encoding="utf-8",
            )

            assert process.stdout is not None

            async for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                    # Only emit log events (type="log")
                    if event.get("type") == "log":
                        content = event.get("content", "")
                        stream = event.get("stream", "stdout")
                        if self._log_callback:
                            self._log_callback(content, stream)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Log streaming error for {self.host}: {e}")

    def stop_log_streaming(self) -> None:
        """Stop log streaming."""
        if self._log_task is not None:
            self._log_task.cancel()
            self._log_task = None
            self._log_callback = None


__all__ = [
    "AsyncRPyCExecutor",
    "RPYC_PORT",
]
