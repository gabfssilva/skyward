"""Tests for TcpProxy — bridging, rotation, and tracked-bridge shutdown.

Uses real in-process echo servers and a fake transport whose
``open_channel`` opens a loopback connection to its echo server, so the
proxy's byte-pump and rotation are exercised end to end.
"""
from __future__ import annotations

import asyncio
import socket
from contextlib import suppress
from typing import Any

import pytest

from skyward.infra.tcp_proxy import TcpProxy

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


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


class _EchoTransport:
    def __init__(self, echo_port: int) -> None:
        self._echo_port = echo_port

    async def open_channel(self, host: str, port: int) -> tuple[Any, Any]:
        return await asyncio.open_connection("127.0.0.1", self._echo_port)


async def _roundtrip(local_port: int, payload: bytes = b"ping") -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", local_port)
    writer.write(payload)
    await writer.drain()
    data = await asyncio.wait_for(reader.read(1024), timeout=2.0)
    writer.close()
    with suppress(Exception):
        await writer.wait_closed()
    return data


async def test_bridges_bytes_through_one_node() -> None:
    server, echo_port = await _start_echo(b"n0:")
    proxy = TcpProxy(remote_port=9999, local_port=_free_port(), initial_nodes=((0, _EchoTransport(echo_port)),))  # type: ignore[arg-type]
    try:
        await proxy.start()
        assert await _roundtrip(proxy._local_port, b"hello") == b"n0:hello"
    finally:
        await proxy.stop()
        server.close()
        await server.wait_closed()


async def test_round_robin_alternates_across_nodes() -> None:
    s0, p0 = await _start_echo(b"0:")
    s1, p1 = await _start_echo(b"1:")
    local = _free_port()
    proxy = TcpProxy(
        remote_port=9999, local_port=local,
        initial_nodes=((0, _EchoTransport(p0)), (1, _EchoTransport(p1))),  # type: ignore[arg-type]
    )
    try:
        await proxy.start()
        tags = [(await _roundtrip(local))[:2] for _ in range(4)]
        assert tags == [b"0:", b"1:", b"0:", b"1:"]
    finally:
        await proxy.stop()
        for srv in (s0, s1):
            srv.close()
            await srv.wait_closed()


async def test_remove_node_drops_from_rotation() -> None:
    s0, p0 = await _start_echo(b"0:")
    s1, p1 = await _start_echo(b"1:")
    local = _free_port()
    proxy = TcpProxy(
        remote_port=9999, local_port=local,
        initial_nodes=((0, _EchoTransport(p0)), (1, _EchoTransport(p1))),  # type: ignore[arg-type]
    )
    try:
        await proxy.start()
        proxy.remove_node(1)
        tags = [(await _roundtrip(local))[:2] for _ in range(3)]
        assert tags == [b"0:", b"0:", b"0:"]
    finally:
        await proxy.stop()
        for srv in (s0, s1):
            srv.close()
            await srv.wait_closed()


async def test_add_node_joins_rotation_and_is_idempotent() -> None:
    s0, p0 = await _start_echo(b"0:")
    s1, p1 = await _start_echo(b"1:")
    local = _free_port()
    proxy = TcpProxy(remote_port=9999, local_port=local, initial_nodes=((0, _EchoTransport(p0)),))  # type: ignore[arg-type]
    try:
        await proxy.start()
        assert (await _roundtrip(local))[:2] == b"0:"

        t1 = _EchoTransport(p1)
        proxy.add_node(1, t1)  # type: ignore[arg-type]
        proxy.add_node(1, t1)  # type: ignore[arg-type]
        tags = [(await _roundtrip(local))[:2] for _ in range(2)]
        assert tags == [b"1:", b"0:"]
    finally:
        await proxy.stop()
        for srv in (s0, s1):
            srv.close()
            await srv.wait_closed()


async def test_empty_rotation_closes_connection() -> None:
    local = _free_port()
    proxy = TcpProxy(remote_port=9999, local_port=local)
    try:
        await proxy.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", local)
        data = await asyncio.wait_for(reader.read(1024), timeout=2.0)
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()
        assert data == b""
    finally:
        await proxy.stop()


async def test_stop_releases_port_and_cancels_live_bridges() -> None:
    hold = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await hold.wait()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    echo_port = int(server.sockets[0].getsockname()[1])
    local = _free_port()
    proxy = TcpProxy(remote_port=9999, local_port=local, initial_nodes=((0, _EchoTransport(echo_port)),))  # type: ignore[arg-type]
    await proxy.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", local)
    await asyncio.sleep(0.1)
    assert len(proxy._bridges) == 1

    await proxy.stop()
    assert not proxy._bridges

    with pytest.raises((ConnectionRefusedError, OSError)):
        await asyncio.open_connection("127.0.0.1", local)

    writer.close()
    hold.set()
    server.close()
    await server.wait_closed()
