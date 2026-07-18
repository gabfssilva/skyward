"""An interactive session, both halves, without a machine.

The daemon half — which node a shell lands on, and the up/down rendezvous — is
driven against a fake channel; the endpoint half against a fake shell, so the
framing can be read off without a compute being live. Raw mode is the one thing
left out: it needs a real tty, and a test process does not have one.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from litestar.testing import AsyncTestClient

from skyward.application import ports
from skyward.application.runtimes import Runtime, Runtimes, Terminal
from skyward.runtime.source import Source
from skyward.server.app import create_app, with_real

pytestmark = pytest.mark.unit


def _noop(*_: object) -> None:
    pass


def _runtime() -> tuple[Runtimes, Runtime]:
    runtimes = Runtimes(_noop, _noop, _noop, _noop)
    return runtimes, runtimes.open("cmp", Source(argument="skyward"), "key")


class EchoChannel:
    """A loopback channel: whatever is written surfaces on read, in order."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._buffer = b""
        self._eof = False

    def write(self, data: bytes) -> None:
        self._queue.put_nowait(data)

    async def drain(self) -> None:
        pass

    def write_eof(self) -> None:
        self._queue.put_nowait(None)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def read(self, n: int) -> bytes:
        while not self._buffer and not self._eof:
            if (item := await self._queue.get()) is None:
                self._eof = True
            else:
                self._buffer += item
        data, self._buffer = self._buffer[:n], self._buffer[n:]
        return data


class _FakeSsh:
    def __init__(self) -> None:
        self.shells: list[tuple[str | None, str, tuple[int, int]]] = []

    async def open_shell(self, command: str | None, term: str, size: tuple[int, int]) -> tuple[EchoChannel, EchoChannel]:
        self.shells.append((command, term, size))
        return EchoChannel(), EchoChannel()


class _FakeNode:
    def __init__(self, ready: bool = True) -> None:
        self.tunnel = 1 if ready else None
        self._ssh = _FakeSsh()


async def test_open_shell_without_a_ready_node_raises() -> None:
    _, runtime = _runtime()
    with pytest.raises(RuntimeError):
        await runtime.open_shell()


async def test_open_shell_stays_on_one_node_instead_of_rotating(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    a, b = _FakeNode(), _FakeNode()
    monkeypatch.setattr(runtime, "nodes", {"a": a, "b": b})

    for _ in range(3):
        await runtime.open_shell()

    assert len(a._ssh.shells) == 3
    assert b._ssh.shells == []


async def test_open_shell_honours_the_node_it_was_given(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    a, b = _FakeNode(), _FakeNode()
    monkeypatch.setattr(runtime, "nodes", {"a": a, "b": b})

    await runtime.open_shell("b", "/opt/skyward/.venv/bin/python", "xterm", (120, 40))

    assert a._ssh.shells == []
    assert b._ssh.shells == [("/opt/skyward/.venv/bin/python", "xterm", (120, 40))]


async def test_open_shell_refuses_a_node_that_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    monkeypatch.setattr(runtime, "nodes", {"a": _FakeNode(), "b": _FakeNode(ready=False)})

    with pytest.raises(RuntimeError, match="node b is not ready"):
        await runtime.open_shell("b")


async def test_terminal_bridges_up_and_down_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    runtimes, runtime = _runtime()
    echo = EchoChannel()

    async def open_shell(node_id: str | None, command: str | None, term: str, size: tuple[int, int]) -> tuple[EchoChannel, EchoChannel]:
        return echo, echo

    monkeypatch.setattr(runtime, "open_shell", open_shell)
    terminal = Terminal(runtimes)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"ls"
        yield b"\r"

    painted = bytearray()

    async def collect() -> None:
        async for data in terminal.down("c1"):
            painted.extend(data)

    async with asyncio.TaskGroup() as group:
        group.create_task(terminal.up("cmp", "c1", None, None, "xterm", (80, 24), chunks()))
        group.create_task(collect())

    assert bytes(painted) == b"ls\r"


async def test_terminal_up_on_a_compute_that_is_not_live_raises() -> None:
    runtimes = Runtimes(_noop, _noop, _noop, _noop)
    terminal = Terminal(runtimes)

    async def nothing() -> AsyncIterator[bytes]:
        return
        yield b""

    with pytest.raises(RuntimeError, match="not live"):
        await terminal.up("cmp", "c1", None, None, "xterm", (80, 24), nothing())


class _FakeShell:
    def __init__(self) -> None:
        self.opened: list[tuple[str, str, str | None, str | None, str, tuple[int, int]]] = []
        self.typed = bytearray()

    async def up(
        self,
        compute_id: str,
        cid: str,
        node_id: str | None,
        command: str | None,
        term: str,
        size: tuple[int, int],
        chunks: AsyncIterator[bytes],
    ) -> None:
        self.opened.append((compute_id, cid, node_id, command, term, size))
        async for chunk in chunks:
            self.typed.extend(chunk)

    async def down(self, cid: str) -> AsyncIterator[bytes]:
        yield f"{cid}$ ".encode()


@pytest.fixture
async def shell() -> _FakeShell:
    return _FakeShell()


@pytest.fixture
async def client(shell: _FakeShell) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(with_real(shell=shell))) as client:
        yield client


async def test_the_fake_shell_satisfies_the_port(shell: _FakeShell) -> None:
    assert isinstance(shell, ports.Shell)


async def test_up_carries_the_body_and_the_session_it_belongs_to(client: AsyncTestClient, shell: _FakeShell) -> None:
    response = await client.post(
        "/v1/computes/cmp/shell/up",
        content=b"whoami\r",
        params={"cid": "c1", "node": "nod_1", "command": "python", "term": "xterm", "columns": 120, "rows": 40},
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    assert shell.opened == [("cmp", "c1", "nod_1", "python", "xterm", (120, 40))]
    assert bytes(shell.typed) == b"whoami\r"


async def test_up_defaults_to_no_node_and_the_login_shell(client: AsyncTestClient, shell: _FakeShell) -> None:
    await client.post(
        "/v1/computes/cmp/shell/up",
        content=b"",
        params={"cid": "c1"},
        headers={"Content-Type": "application/octet-stream"},
    )

    assert shell.opened == [("cmp", "c1", None, None, "xterm-256color", (80, 24))]


async def test_down_streams_raw_bytes_for_the_session(client: AsyncTestClient) -> None:
    response = await client.get("/v1/computes/cmp/shell/down", params={"cid": "c1"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.content == b"c1$ "
