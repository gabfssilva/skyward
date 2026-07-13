"""Tests for the round-robin TCP proxy actor.

Uses real in-process echo servers and a stub transport actor whose
``OpenChannel`` opens a loopback connection to its echo server, so the
proxy's byte-pump and rotation are exercised end to end.
"""
import asyncio
import socket
from contextlib import suppress

import pytest
from casty import ActorContext, ActorSystem, Behavior, Behaviors

from skyward.actors.tcp_proxy import NodeDown, NodeUp, tcp_proxy
from skyward.actors.tcp_proxy.messages import ProxyMsg
from skyward.infra.ssh_actor import ChannelOpened, OpenChannel


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


async def _start_echo(tag: bytes) -> tuple[asyncio.Server, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        data = await reader.read(1024)
        writer.write(tag + data)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, int(server.sockets[0].getsockname()[1])


def _echo_transport(echo_port: int) -> Behavior[object]:
    async def receive(ctx: ActorContext[object], msg: object) -> Behavior[object]:
        match msg:
            case OpenChannel(reply_to=rt):
                reader, writer = await asyncio.open_connection("127.0.0.1", echo_port)
                rt.tell(ChannelOpened(reader=reader, writer=writer))
        return Behaviors.same()

    return Behaviors.receive(receive)


async def _roundtrip(local_port: int, payload: bytes = b"ping") -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", local_port)
    writer.write(payload)
    await writer.drain()
    data = await asyncio.wait_for(reader.read(1024), timeout=2.0)
    writer.close()
    with suppress(Exception):
        await writer.wait_closed()
    return data


@pytest.fixture
async def system():
    s = ActorSystem("test-tcp-proxy")
    yield s
    await s.shutdown()


@pytest.mark.asyncio
async def test_bridges_bytes_through_one_node(system: ActorSystem) -> None:
    server, echo_port = await _start_echo(b"n0:")
    try:
        t0 = system.spawn(_echo_transport(echo_port), "t0")
        local = _free_port()
        system.spawn(
            tcp_proxy(remote_port=9999, local_port=local, initial_nodes=((0, t0),)),
            "proxy",
        )
        await asyncio.sleep(0.2)

        assert await _roundtrip(local, b"hello") == b"n0:hello"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_round_robin_alternates_across_nodes(system: ActorSystem) -> None:
    s0, p0 = await _start_echo(b"0:")
    s1, p1 = await _start_echo(b"1:")
    try:
        t0 = system.spawn(_echo_transport(p0), "t0")
        t1 = system.spawn(_echo_transport(p1), "t1")
        local = _free_port()
        system.spawn(
            tcp_proxy(remote_port=9999, local_port=local, initial_nodes=((0, t0), (1, t1))),
            "proxy",
        )
        await asyncio.sleep(0.2)

        tags = [(await _roundtrip(local))[:2] for _ in range(4)]
        assert tags == [b"0:", b"1:", b"0:", b"1:"]
    finally:
        for srv in (s0, s1):
            srv.close()
            await srv.wait_closed()


@pytest.mark.asyncio
async def test_node_down_removes_from_rotation(system: ActorSystem) -> None:
    s0, p0 = await _start_echo(b"0:")
    s1, p1 = await _start_echo(b"1:")
    try:
        t0 = system.spawn(_echo_transport(p0), "t0")
        t1 = system.spawn(_echo_transport(p1), "t1")
        local = _free_port()
        proxy: object = system.spawn(
            tcp_proxy(remote_port=9999, local_port=local, initial_nodes=((0, t0), (1, t1))),
            "proxy",
        )
        await asyncio.sleep(0.2)

        proxy.tell(NodeDown(node_id=1))  # type: ignore[attr-defined]
        await asyncio.sleep(0.1)

        tags = [(await _roundtrip(local))[:2] for _ in range(3)]
        assert tags == [b"0:", b"0:", b"0:"]
    finally:
        for srv in (s0, s1):
            srv.close()
            await srv.wait_closed()


@pytest.mark.asyncio
async def test_node_up_adds_to_rotation(system: ActorSystem) -> None:
    s0, p0 = await _start_echo(b"0:")
    s1, p1 = await _start_echo(b"1:")
    try:
        t0 = system.spawn(_echo_transport(p0), "t0")
        t1 = system.spawn(_echo_transport(p1), "t1")
        local = _free_port()
        proxy: object = system.spawn(
            tcp_proxy(remote_port=9999, local_port=local, initial_nodes=((0, t0),)),
            "proxy",
        )
        await asyncio.sleep(0.2)

        assert (await _roundtrip(local))[:2] == b"0:"

        proxy.tell(NodeUp(node_id=1, transport_ref=t1))  # type: ignore[attr-defined]
        await asyncio.sleep(0.1)

        # cursor is now 1 -> next picks node 1, then alternates
        tags = [(await _roundtrip(local))[:2] for _ in range(2)]
        assert tags == [b"1:", b"0:"]
    finally:
        for srv in (s0, s1):
            srv.close()
            await srv.wait_closed()


@pytest.mark.asyncio
async def test_empty_rotation_closes_connection(system: ActorSystem) -> None:
    local = _free_port()
    system.spawn(
        tcp_proxy(remote_port=9999, local_port=local, initial_nodes=()),
        "proxy",
    )
    await asyncio.sleep(0.2)

    # No ready nodes: the proxy accepts then immediately closes -> clean EOF.
    reader, writer = await asyncio.open_connection("127.0.0.1", local)
    data = await asyncio.wait_for(reader.read(1024), timeout=2.0)
    writer.close()
    with suppress(Exception):
        await writer.wait_closed()
    assert data == b""


@pytest.mark.asyncio
async def test_stop_proxy_releases_port(system: ActorSystem) -> None:
    from skyward.actors.tcp_proxy import StopProxy

    local = _free_port()
    proxy: object = system.spawn(
        tcp_proxy(remote_port=9999, local_port=local, initial_nodes=()),
        "proxy",
    )
    await asyncio.sleep(0.2)
    proxy.tell(StopProxy())  # type: ignore[attr-defined]
    await asyncio.sleep(0.2)

    with pytest.raises((ConnectionRefusedError, OSError)):
        await asyncio.open_connection("127.0.0.1", local)
