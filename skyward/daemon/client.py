"""Daemon client -- async connection to the daemon over Unix socket."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from types import TracebackType

from .protocol import (
    BroadcastSucceeded,
    DaemonError,
    DaemonRequest,
    DaemonResponse,
    DaemonStopped,
    Disconnect,
    EnsurePool,
    GetNodeCount,
    NodeCount,
    Ping,
    Pong,
    PoolFailed,
    PoolLogLine,
    PoolProvisioning,
    PoolReady,
    PoolShutdown,
    ShutdownDaemon,
    ShutdownPool,
    StreamEnd,
    SubmitBroadcast,
    SubmitTask,
    SubscribeEvents,
    TaskFailed,
    TaskSucceeded,
)
from .wire import async_recv, async_send

_DEFAULT_SOCKET = Path.home() / ".skyward" / "daemon.sock"


class DaemonClient:
    """Async client for communicating with the daemon."""

    def __init__(
        self, socket_path: Path = _DEFAULT_SOCKET, default_timeout: float = 600.0,
    ) -> None:
        self._socket_path = socket_path
        self._default_timeout = default_timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        try:
            self._reader, self._writer = await asyncio.open_unix_connection(
                str(self._socket_path),
            )
        except (FileNotFoundError, ConnectionRefusedError):
            raise ConnectionError(
                "Daemon is not running. Start it with: sky daemon start",
            ) from None

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None

    async def __aenter__(self) -> DaemonClient:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def _request(
        self, msg: DaemonRequest, timeout: float | None = None,
    ) -> DaemonResponse:
        assert self._reader is not None and self._writer is not None
        await async_send(self._writer, msg)
        resp = await asyncio.wait_for(
            async_recv(self._reader),
            timeout=timeout or self._default_timeout,
        )
        if isinstance(resp, DaemonError):
            raise RuntimeError(f"Daemon error: {resp.error}")
        return resp  # type: ignore[return-value]

    async def request(
        self, msg: DaemonRequest, timeout: float | None = None,
    ) -> DaemonResponse:
        """Send a request and return the raw response.

        Unlike ``_request``, this does **not** raise on ``DaemonError``
        — the caller is responsible for inspecting the response type.
        """
        assert self._reader is not None and self._writer is not None
        await async_send(self._writer, msg)
        resp = await asyncio.wait_for(
            async_recv(self._reader),
            timeout=timeout or self._default_timeout,
        )
        return resp  # type: ignore[return-value]

    async def ping(self) -> Pong:
        return await self._request(Ping())  # type: ignore[return-value]

    async def ensure_pool(
        self, name: str, *, spec_bytes: bytes = b"",
    ) -> PoolReady:
        assert self._reader is not None and self._writer is not None
        await async_send(self._writer, EnsurePool(name=name, spec_bytes=spec_bytes))
        while True:
            resp = await asyncio.wait_for(
                async_recv(self._reader),
                timeout=self._default_timeout,
            )
            match resp:
                case PoolReady():
                    return resp
                case PoolFailed(reason=reason):
                    raise RuntimeError(f"Pool '{name}' failed: {reason}")
                case PoolProvisioning() | PoolLogLine():
                    continue
                case DaemonError(error=error):
                    raise RuntimeError(f"Daemon error: {error}")
        raise RuntimeError("Unexpected end of stream")

    async def ensure_pool_stream(
        self, name: str, *, spec_bytes: bytes = b"",
    ) -> AsyncIterator[object]:
        assert self._reader is not None and self._writer is not None
        await async_send(self._writer, EnsurePool(name=name, spec_bytes=spec_bytes))
        while True:
            resp = await asyncio.wait_for(
                async_recv(self._reader),
                timeout=self._default_timeout,
            )
            match resp:
                case DaemonError(error=error):
                    raise RuntimeError(f"Daemon error: {error}")
                case _:
                    yield resp
            match resp:
                case PoolReady() | PoolFailed():
                    return

    async def submit_task(
        self, pool_name: str, payload: bytes, timeout: float = 300.0,
    ) -> TaskSucceeded:
        resp = await self._request(
            SubmitTask(pool_name=pool_name, payload=payload, timeout=timeout),
        )
        match resp:
            case TaskFailed(error=error, traceback=tb):
                raise RuntimeError(f"Remote task failed: {error}\n{tb}")
            case TaskSucceeded():
                return resp
        raise RuntimeError(f"Unexpected response: {resp}")

    async def submit_broadcast(
        self, pool_name: str, payload: bytes, timeout: float = 300.0,
    ) -> BroadcastSucceeded:
        resp = await self._request(
            SubmitBroadcast(pool_name=pool_name, payload=payload, timeout=timeout),
        )
        match resp:
            case TaskFailed(error=error, traceback=tb):
                raise RuntimeError(f"Remote broadcast failed: {error}\n{tb}")
            case BroadcastSucceeded():
                return resp
        raise RuntimeError(f"Unexpected response: {resp}")

    async def get_node_count(self, pool_name: str) -> int:
        resp = await self._request(GetNodeCount(pool_name=pool_name))
        match resp:
            case NodeCount(ready=n):
                return n
        raise RuntimeError(f"Unexpected response: {resp}")

    async def disconnect(self, pool_name: str) -> None:
        assert self._writer is not None
        await async_send(self._writer, Disconnect(pool_name=pool_name))

    async def shutdown_pool(self, pool_name: str) -> None:
        resp = await self._request(ShutdownPool(pool_name=pool_name))
        match resp:
            case PoolShutdown():
                return
        raise RuntimeError(f"Unexpected response: {resp}")

    async def shutdown_daemon(self) -> None:
        resp = await self._request(ShutdownDaemon())
        match resp:
            case DaemonStopped():
                return
        raise RuntimeError(f"Unexpected response: {resp}")

    async def subscribe(
        self, pool_name: str,
    ) -> AsyncIterator[object]:
        """Subscribe to live events for a pool.

        Yields ``SessionView`` (state snapshots), ``Log.Emitted`` (log lines),
        and ``SessionEvent`` domain events (node ready, task completed, etc.).
        Stream ends when ``StreamEnd`` is received or connection closes.
        """
        assert self._reader is not None and self._writer is not None
        await async_send(self._writer, SubscribeEvents(pool_name=pool_name))

        while True:
            try:
                msg = await async_recv(self._reader)
            except (asyncio.IncompleteReadError, ConnectionError, EOFError):
                break
            match msg:
                case StreamEnd():
                    break
                case DaemonError(error=err):
                    raise RuntimeError(f"Subscribe failed: {err}")
                case _:
                    yield msg
