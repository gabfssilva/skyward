"""Port forwarding, both ends, without a machine.

The client half — the loopback listener and the byte pump — is driven against a
fake daemon that round-robins a node per connection and echoes the bytes back.
The daemon half — the round-robin selection and the up/down rendezvous — is
driven against a fake channel, so a real SSH connection and a real compute are
never needed to prove either does what it says.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

import pytest

from skyward.server.application.runtimes import Forward, Runtime, Runtimes
from skyward.server.application.source import Source
from skyward.core.forward import TcpProxy
from skyward.core.spec import Port

pytestmark = pytest.mark.unit


def _noop(*_: object) -> None:
    pass


def _runtime() -> tuple[Runtimes, Runtime]:
    runtimes = Runtimes(_noop, _noop, _noop, _noop)
    return runtimes, runtimes.open("cmp", Source(arguments=("skyward",)), "key")


class EchoChannel:
    """A loopback TCP channel: whatever is written surfaces on read, in order."""

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


class FakeDaemon:
    """The daemon half, faked: a node round-robined per connection, bytes echoed.

    ``forward_down`` answers with the node the connection was routed to, a colon,
    then the bytes ``forward_up`` fed in — enough to prove the pump carried them
    and to read off which node each connection landed on.
    """

    def __init__(self, nodes: tuple[str, ...]) -> None:
        self._nodes = nodes
        self._cursor = 0
        self._conns: dict[str, asyncio.Queue[bytes | None]] = {}
        self.routed: list[str] = []

    def _queue(self, cid: str) -> asyncio.Queue[bytes | None]:
        return self._conns.setdefault(cid, asyncio.Queue())

    async def forward_up(self, compute: str, cid: str, port: int, route: str, chunks: AsyncIterator[bytes]) -> None:
        node = self._nodes[self._cursor % len(self._nodes)]
        self._cursor += 1
        self.routed.append(node)
        queue = self._queue(cid)
        await queue.put(f"{node}:".encode())
        async for chunk in chunks:
            await queue.put(chunk)
        await queue.put(None)

    async def forward_down(self, compute: str, cid: str) -> AsyncIterator[bytes]:
        queue = self._queue(cid)
        while (item := await queue.get()) is not None:
            yield item


async def _roundtrip(local_port: int, payload: bytes) -> bytes:
    """Open one connection to the proxy, send ``payload``, read the answer to EOF."""
    reader, writer = await asyncio.open_connection("127.0.0.1", local_port)
    writer.write(payload)
    await writer.drain()
    writer.write_eof()
    answer = await reader.read()
    writer.close()
    with suppress(Exception):
        await writer.wait_closed()
    return answer


async def test_tcp_proxy_pumps_bytes_both_ways() -> None:
    proxy = TcpProxy(FakeDaemon(("node-0",)), "cmp", Port(remote=80, local=0))
    await proxy.start()
    try:
        answer = await _roundtrip(proxy.local, b"ping")
    finally:
        await proxy.stop()

    assert answer == b"node-0:ping"


async def test_tcp_proxy_round_robins_across_connections() -> None:
    daemon = FakeDaemon(("node-0", "node-1"))
    proxy = TcpProxy(daemon, "cmp", Port(remote=80))
    await proxy.start()
    try:
        tags = [(await _roundtrip(proxy.local, b"x")).split(b":", 1)[0].decode() for _ in range(4)]
    finally:
        await proxy.stop()

    assert tags == ["node-0", "node-1", "node-0", "node-1"]
    assert daemon.routed == ["node-0", "node-1", "node-0", "node-1"]


async def test_tcp_proxy_stop_closes_the_listener() -> None:
    proxy = TcpProxy(FakeDaemon(("node-0",)), "cmp", Port(remote=80))
    await proxy.start()
    port = proxy.local
    await proxy.stop()

    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", port)


class _FakeSsh:
    def __init__(self) -> None:
        self.opened: list[tuple[str, int]] = []

    async def open_connection(self, host: str, port: int) -> tuple[EchoChannel, EchoChannel]:
        self.opened.append((host, port))
        return EchoChannel(), EchoChannel()


class _FakeNode:
    def __init__(self) -> None:
        self.tunnel = 1
        self._ssh = _FakeSsh()


async def test_open_channel_without_a_ready_node_raises() -> None:
    _, runtime = _runtime()
    with pytest.raises(RuntimeError):
        await runtime.open_channel(80)


async def test_open_channel_round_robins_over_ready_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    a, b = _FakeNode(), _FakeNode()
    monkeypatch.setattr(runtime, "nodes", {"a": a, "b": b})

    for _ in range(3):
        await runtime.open_channel(80)

    assert len(a._ssh.opened) == 2
    assert len(b._ssh.opened) == 1
    assert a._ssh.opened[0] == ("127.0.0.1", 80)


async def test_forward_bridges_up_and_down_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    runtimes, runtime = _runtime()
    echo = EchoChannel()

    async def open_channel(remote_port: int, route: str = "round_robin") -> tuple[EchoChannel, EchoChannel]:
        return echo, echo

    monkeypatch.setattr(runtime, "open_channel", open_channel)
    forward = Forward(runtimes)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"hel"
        yield b"lo"

    received = bytearray()

    async def collect() -> None:
        async for data in forward.down("c1"):
            received.extend(data)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(forward.up("cmp", "c1", 80, "round_robin", chunks()))
        tg.create_task(collect())

    assert bytes(received) == b"hello"


async def test_forward_down_waits_for_a_later_up(monkeypatch: pytest.MonkeyPatch) -> None:
    runtimes, runtime = _runtime()
    echo = EchoChannel()

    async def open_channel(remote_port: int, route: str = "round_robin") -> tuple[EchoChannel, EchoChannel]:
        return echo, echo

    monkeypatch.setattr(runtime, "open_channel", open_channel)
    forward = Forward(runtimes)

    received = bytearray()

    async def collect() -> None:
        async for data in forward.down("c2"):
            received.extend(data)

    down = asyncio.create_task(collect())
    await asyncio.sleep(0)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"world"

    await forward.up("cmp", "c2", 80, "round_robin", chunks())
    await down

    assert bytes(received) == b"world"
