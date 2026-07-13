"""SSH transport as a plain asyncio class.

Owns the asyncssh connection end to end — no raw connection leaks.
Reconnection and port re-forwarding are internal: operations issued
while the link is down wait for the reconnect to finish, and forwards
are re-established on the same local ports. Connection state changes
are pushed to an optional ``on_event`` callback.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from skyward.infra.ssh import RawStreamEvent
from skyward.observability.logger import logger

log = logger.bind(component="ssh_transport")

EVENTS_FILE = "/opt/skyward/events.jsonl"


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class StreamEvent:
    lines_read: int
    event: RawStreamEvent


@dataclass(frozen=True, slots=True)
class ConnectionLost:
    error: str


@dataclass(frozen=True, slots=True)
class ConnectionRestored:
    pass


@dataclass(frozen=True, slots=True)
class ConnectionFailed:
    error: str


type ConnectionEvent = ConnectionLost | ConnectionRestored | ConnectionFailed


class TransportUnavailableError(RuntimeError):
    """Transport closed or permanently failed — no further operations possible."""


@dataclass(frozen=True, slots=True)
class _Forward:
    """Forward already allocated — local_port is stable across reconnects."""

    remote_host: str
    remote_port: int
    local_port: int


async def _default_connect(
    host: str, port: int, user: str, key_path: str, connect_timeout: float,
    password: str | None = None,
) -> Any:
    import asyncssh
    return await asyncssh.connect(
        host, port=port, username=user,
        client_keys=[key_path] if key_path else [],
        password=password,
        known_hosts=None,
        connect_timeout=connect_timeout,
        keepalive_interval=10,
        keepalive_count_max=5,
        tcp_keepalive=True,
    )


def _is_permanent(e: Exception) -> bool:
    import asyncssh
    return isinstance(e, (asyncssh.PermissionDenied, asyncssh.PasswordChangeRequired))


class SshTransport:
    """Async SSH transport with internal retry, reconnect, and re-forward."""

    def __init__(
        self,
        host: str,
        user: str,
        key_path: str,
        port: int = 22,
        *,
        retry_max_attempts: int = 150,
        retry_delay: float = 2.0,
        connect_timeout: float = 30.0,
        reconnect_max_attempts: int | None = None,
        password: str | None = None,
        connect_fn: Callable[[], Awaitable[Any]] | None = None,
        on_event: Callable[[ConnectionEvent], None] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._retry_max_attempts = retry_max_attempts
        self._retry_delay = retry_delay
        self._reconnect_max_attempts = (
            reconnect_max_attempts if reconnect_max_attempts is not None else retry_max_attempts
        )
        self._connect = connect_fn or (
            lambda: _default_connect(host, port, user, key_path, connect_timeout, password)
        )
        self._on_event = on_event or (lambda _e: None)
        self._conn: Any = None
        self._connected = asyncio.Event()
        self._closed = False
        self._failed: str | None = None
        self._forwards: tuple[_Forward, ...] = ()
        self._listeners: tuple[Any, ...] = ()
        self._watcher: asyncio.Task[None] | None = None

    # ── lifecycle ─────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish the connection, retrying up to ``retry_max_attempts``."""
        last_error = ""
        for attempt in range(self._retry_max_attempts):
            if self._closed:
                raise TransportUnavailableError("transport closed")
            try:
                conn = await self._connect()
            except Exception as e:
                if _is_permanent(e):
                    log.error("SSH connection failed permanently (auth): {err}", err=str(e))
                    self._fail(str(e))
                    raise TransportUnavailableError(str(e)) from e
                last_error = str(e)
                log.debug(
                    "SSH connect attempt {n} failed: {err}, retrying in {d}s",
                    n=attempt + 1, err=last_error, d=self._retry_delay,
                )
                await asyncio.sleep(self._retry_delay)
                continue
            log.info("SSH connected to {host}:{port}", host=self._host, port=self._port)
            self._attach(conn)
            return
        self._fail(last_error)
        raise TransportUnavailableError(
            f"SSH connection failed after {self._retry_max_attempts} attempts: {last_error}"
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
        self._close_listeners()
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
                await asyncio.wait_for(self._conn.wait_closed(), timeout=5.0)
        self._connected.set()

    # ── operations ────────────────────────────────────────────────────

    async def run(
        self, *command: str, timeout: float | None = None, check: bool = False,
    ) -> CommandResult:
        conn = await self._ready()
        result = await conn.run(" ".join(command), timeout=timeout, check=False)
        code = result.exit_status or 0
        stdout = str(result.stdout or "")
        stderr = str(result.stderr or "")
        if check and code != 0:
            raise RuntimeError(f"Command failed ({code}): {stderr}")
        return CommandResult(exit_code=code, stdout=stdout, stderr=stderr)

    async def write_file(self, remote: str, content: str) -> None:
        conn = await self._ready()
        async with conn.create_process(f"cat > {remote}") as proc:
            proc.stdin.write(content)
            proc.stdin.write_eof()
            await proc.wait()

    async def write_bytes(self, remote: str, content: bytes) -> None:
        conn = await self._ready()
        async with conn.create_process(f"cat > {remote}", encoding=None) as proc:
            proc.stdin.write(content)
            proc.stdin.write_eof()
            await proc.wait()

    async def upload(self, local: str, remote: str) -> None:
        import asyncssh
        conn = await self._ready()
        await asyncssh.scp(local, (conn, remote))

    async def download(self, remote: str) -> bytes:
        conn = await self._ready()
        async with conn.start_sftp_client() as sftp, sftp.open(remote, "rb") as f:
            return await f.read()

    async def forward_port(self, remote_host: str, remote_port: int) -> int:
        conn = await self._ready()
        local_port, listener = await self._do_forward(conn, remote_host, remote_port)
        self._forwards = (*self._forwards, _Forward(remote_host, remote_port, local_port))
        self._listeners = (*self._listeners, listener)
        return local_port

    async def open_channel(self, remote_host: str, remote_port: int) -> tuple[Any, Any]:
        conn = await self._ready()
        reader, writer = await conn.open_connection(remote_host, remote_port)
        return reader, writer

    async def events(self, start_line: int = 0) -> AsyncIterator[StreamEvent]:
        """Stream JSONL events from the node, transparently surviving reconnects.

        Ends when the transport is closed or permanently fails.
        """
        offset = start_line
        while True:
            try:
                conn = await self._ready()
            except TransportUnavailableError:
                return
            try:
                async for ev in self._stream(conn, offset):
                    offset = ev.lines_read
                    yield ev
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._closed or self._failed:
                    return
                log.warning("Event stream interrupted: {err}", err=str(e))
            if self._closed or self._failed:
                return
            await asyncio.sleep(0.5)

    # ── internals ─────────────────────────────────────────────────────

    def _attach(self, conn: Any) -> None:
        self._conn = conn
        self._connected.set()
        self._watcher = asyncio.create_task(self._watch(conn))

    def _fail(self, error: str) -> None:
        self._failed = error
        self._connected.set()

    async def _ready(self) -> Any:
        while True:
            await self._connected.wait()
            if self._closed:
                raise TransportUnavailableError("transport closed")
            if self._failed:
                raise TransportUnavailableError(f"transport failed: {self._failed}")
            return self._conn

    async def _watch(self, conn: Any) -> None:
        await conn.wait_closed()
        if self._closed:
            return
        self._connected.clear()
        log.warning("Connection to {host} dropped, reconnecting", host=self._host)
        self._close_listeners()
        self._on_event(ConnectionLost(error="SSH connection closed by remote"))
        await self._reconnect()

    async def _reconnect(self) -> None:
        for attempt in range(self._reconnect_max_attempts):
            log.info("Reconnecting in {d}s (attempt {n})", d=self._retry_delay, n=attempt + 1)
            await asyncio.sleep(self._retry_delay)
            if self._closed:
                return
            try:
                conn = await self._connect()
                listeners = [
                    (await self._do_forward(conn, f.remote_host, f.remote_port, f.local_port))[1]
                    for f in self._forwards
                ]
            except Exception as e:
                log.debug("Reconnect attempt {n} failed: {err}", n=attempt + 1, err=str(e))
                continue
            log.info("Reconnected to {host}:{port}", host=self._host, port=self._port)
            self._listeners = tuple(listeners)
            self._attach(conn)
            self._on_event(ConnectionRestored())
            return
        log.error("Reconnection permanently failed after {n} attempts", n=self._reconnect_max_attempts)
        self._fail("max reconnection attempts")
        self._on_event(ConnectionFailed(error="max reconnection attempts"))

    async def _do_forward(
        self, conn: Any, remote_host: str, remote_port: int, local_port: int = 0,
    ) -> tuple[int, Any]:
        listener = await conn.forward_local_port("", local_port, remote_host, remote_port)
        return listener.get_port(), listener

    def _close_listeners(self) -> None:
        for listener in self._listeners:
            with contextlib.suppress(Exception):
                listener.close()
        self._listeners = ()

    async def _stream(self, conn: Any, start_line: int) -> AsyncIterator[StreamEvent]:
        from skyward.infra.ssh import _parse_jsonl_line

        for _ in range(120):
            result = await conn.run(f"test -f {EVENTS_FILE}", check=False)
            if (result.exit_status or 0) == 0:
                break
            await asyncio.sleep(1.0)

        tail_offset = start_line + 1
        async with conn.create_process(
            f"stdbuf -oL tail -s 0.1 -n +{tail_offset} -F {EVENTS_FILE}"
        ) as proc:
            buffer = ""
            lines_read = start_line
            while True:
                chunk = await asyncio.wait_for(proc.stdout.read(4096), timeout=600.0)
                if not chunk:
                    return
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    lines_read += 1
                    line = line.strip()
                    if not line:
                        continue
                    event = _parse_jsonl_line(line)
                    if event is not None:
                        yield StreamEvent(lines_read=lines_read, event=event)
