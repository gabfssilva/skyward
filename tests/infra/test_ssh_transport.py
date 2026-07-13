"""Tests for SshTransport — connection, retry, operations, reconnection."""
from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import pytest

from skyward.infra.ssh import RawLogEvent
from skyward.infra.ssh_transport import (
    ConnectionFailed,
    ConnectionLost,
    ConnectionRestored,
    SshTransport,
    StreamEvent,
    TransportUnavailableError,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class FakeListener:
    def __init__(self, port: int) -> None:
        self._port = port
        self.closed = False

    def get_port(self) -> int:
        return self._port

    def close(self) -> None:
        self.closed = True


class FakeResult:
    def __init__(self, exit_status: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_status = exit_status
        self.stdout = stdout
        self.stderr = stderr


class FakeConn:
    def __init__(self, stream_lines: list[str] | None = None) -> None:
        self._closed = asyncio.Event()
        self.run_calls: list[str] = []
        self.forwards: list[tuple[int, str, int]] = []
        self._next_local_port = 50000
        self._stream_lines = stream_lines or []

    async def wait_closed(self) -> None:
        await self._closed.wait()

    def is_closed(self) -> bool:
        return self._closed.is_set()

    def close(self) -> None:
        self._closed.set()

    def drop(self) -> None:
        self._closed.set()

    async def run(self, cmd: str, timeout: float | None = None, check: bool = False) -> FakeResult:
        self.run_calls.append(cmd)
        return FakeResult(exit_status=0, stdout="out", stderr="")

    async def forward_local_port(
        self, host: str, local_port: int, remote_host: str, remote_port: int,
    ) -> FakeListener:
        port = local_port or self._next_local_port
        self.forwards.append((port, remote_host, remote_port))
        return FakeListener(port)

    async def open_connection(self, host: str, port: int) -> tuple[str, str]:
        return "reader", "writer"

    def create_process(self, cmd: str, encoding: Any = "utf-8") -> _FakeProcessCM:
        return _FakeProcessCM(self._stream_lines, self._closed)


class _FakeProcessCM:
    def __init__(self, lines: list[str], closed: asyncio.Event) -> None:
        self._chunks = ["".join(f"{line}\n" for line in lines)] if lines else []
        self._closed = closed

    async def __aenter__(self) -> _FakeProcessCM:
        self.stdout = self
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    async def read(self, n: int) -> str:
        if self._chunks:
            return self._chunks.pop(0)
        await self._closed.wait()
        raise ConnectionResetError("connection closed")


def _transport(connect_fn: Any, **kwargs: Any) -> SshTransport:
    return SshTransport(
        "10.0.0.1", "ubuntu", "/tmp/key",
        retry_delay=0.02, connect_fn=connect_fn, **kwargs,
    )


async def test_connects_successfully() -> None:
    conn = FakeConn()
    calls = 0

    async def connect() -> FakeConn:
        nonlocal calls
        calls += 1
        return conn

    t = _transport(connect)
    await t.connect()
    assert calls == 1
    await t.close()


async def test_retries_then_succeeds() -> None:
    conn = FakeConn()
    attempts = 0

    async def connect() -> FakeConn:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError(f"attempt {attempts}")
        return conn

    t = _transport(connect)
    await t.connect()
    assert attempts == 3
    await t.close()


async def test_connect_exhaustion_raises() -> None:
    async def connect() -> FakeConn:
        raise ConnectionError("nope")

    t = _transport(connect, retry_max_attempts=3)
    with pytest.raises(TransportUnavailableError, match="after 3 attempts"):
        await t.connect()


async def test_permanent_auth_failure_does_not_retry() -> None:
    import asyncssh

    attempts = 0

    async def connect() -> FakeConn:
        nonlocal attempts
        attempts += 1
        raise asyncssh.PermissionDenied("denied")

    t = _transport(connect)
    with pytest.raises(TransportUnavailableError):
        await t.connect()
    assert attempts == 1


async def test_run_returns_command_result() -> None:
    conn = FakeConn()

    async def connect() -> FakeConn:
        return conn

    t = _transport(connect)
    await t.connect()
    result = await t.run("echo", "hi", timeout=5.0)
    assert result.exit_code == 0
    assert result.stdout == "out"
    assert conn.run_calls[-1] == "echo hi"
    await t.close()


async def test_run_check_raises_on_nonzero() -> None:
    conn = FakeConn()

    async def failing_run(cmd: str, timeout: float | None = None, check: bool = False) -> FakeResult:
        return FakeResult(exit_status=2, stderr="boom")

    conn.run = failing_run  # type: ignore[method-assign]

    async def connect() -> FakeConn:
        return conn

    t = _transport(connect)
    await t.connect()
    with pytest.raises(RuntimeError, match="boom"):
        await t.run("false", check=True)
    await t.close()


async def test_forward_port_and_open_channel() -> None:
    conn = FakeConn()

    async def connect() -> FakeConn:
        return conn

    t = _transport(connect)
    await t.connect()
    port = await t.forward_port("127.0.0.1", 25520)
    assert port == 50000
    assert conn.forwards == [(50000, "127.0.0.1", 25520)]
    reader, writer = await t.open_channel("127.0.0.1", 8080)
    assert (reader, writer) == ("reader", "writer")
    await t.close()


async def test_reconnects_and_reforwards_on_same_local_port() -> None:
    conns = [FakeConn(), FakeConn()]
    events: list[object] = []

    async def connect() -> FakeConn:
        return conns.pop(0)

    t = _transport(connect, on_event=events.append)
    first = conns[0]
    second = conns[1]
    await t.connect()
    port = await t.forward_port("127.0.0.1", 25520)

    first.drop()
    await asyncio.sleep(0.2)

    assert any(isinstance(e, ConnectionLost) for e in events)
    assert any(isinstance(e, ConnectionRestored) for e in events)
    assert second.forwards == [(port, "127.0.0.1", 25520)]
    result = await t.run("echo hi")
    assert result.exit_code == 0
    await t.close()


async def test_ops_wait_during_reconnect() -> None:
    conns = [FakeConn(), FakeConn()]

    async def connect() -> FakeConn:
        return conns.pop(0)

    t = _transport(connect)
    first = conns[0]
    second = conns[1]
    await t.connect()
    first.drop()
    await asyncio.sleep(0.01)

    result = await asyncio.wait_for(t.run("late"), timeout=2.0)
    assert result.exit_code == 0
    assert second.run_calls == ["late"]
    await t.close()


async def test_reconnect_exhaustion_fails_pending_ops() -> None:
    first = FakeConn()
    connected_once = False

    async def connect() -> FakeConn:
        nonlocal connected_once
        if not connected_once:
            connected_once = True
            return first
        raise ConnectionError("gone")

    events: list[object] = []
    t = _transport(connect, reconnect_max_attempts=2, on_event=events.append)
    await t.connect()
    first.drop()
    await asyncio.sleep(0.01)

    with pytest.raises(TransportUnavailableError):
        await asyncio.wait_for(t.run("late"), timeout=2.0)
    assert any(isinstance(e, ConnectionFailed) for e in events)


async def test_close_rejects_further_ops() -> None:
    conn = FakeConn()

    async def connect() -> FakeConn:
        return conn

    t = _transport(connect)
    await t.connect()
    await t.close()
    with pytest.raises(TransportUnavailableError):
        await t.run("echo")


async def test_events_yields_parsed_stream_and_ends_on_close() -> None:
    lines = [
        json.dumps({"type": "log", "content": "hello"}),
        json.dumps({"type": "log", "content": "world"}),
        "not-json-at-all {",
    ]
    conn = FakeConn(stream_lines=lines)

    async def connect() -> FakeConn:
        return conn

    t = _transport(connect)
    await t.connect()

    received: list[StreamEvent] = []

    async def consume() -> None:
        async for ev in t.events():
            received.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.2)
    assert [ev.event for ev in received] == [
        RawLogEvent(content="hello"),
        RawLogEvent(content="world"),
    ]
    assert [ev.lines_read for ev in received] == [1, 2]

    await t.close()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(task, timeout=2.0)
    assert task.done()
