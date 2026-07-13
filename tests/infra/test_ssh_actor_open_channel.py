"""Tests for the OpenChannel capability of the SSH transport actor."""
import asyncio
from unittest.mock import MagicMock

import pytest
from casty import ActorSystem

from skyward.infra.ssh_actor import (
    ChannelFailed,
    ChannelOpened,
    OpenChannel,
    ssh_transport,
)


async def _as_coro(val: MagicMock) -> MagicMock:
    return val


def _mock_conn() -> MagicMock:
    conn = MagicMock()

    async def _never_close() -> None:
        await asyncio.Future()

    conn.wait_closed = _never_close
    conn.close = MagicMock()
    return conn


@pytest.fixture
async def system():
    s = ActorSystem("test-open-channel")
    yield s
    await s.shutdown()


@pytest.mark.asyncio
async def test_open_channel_returns_streams(system: ActorSystem) -> None:
    conn = _mock_conn()
    reader, writer = MagicMock(), MagicMock()

    async def mock_open(host: str, port: int) -> tuple[MagicMock, MagicMock]:
        return reader, writer

    conn.open_connection = mock_open

    ref = system.spawn(
        ssh_transport(host="x", user="u", key_path="k", connect_fn=lambda: _as_coro(conn)),
        "transport",
    )
    await asyncio.sleep(0.2)

    future = asyncio.get_event_loop().create_future()
    reply = MagicMock()
    reply.tell = lambda msg: future.set_result(msg) if not future.done() else None

    ref.tell(OpenChannel(remote_host="127.0.0.1", remote_port=8080, reply_to=reply))

    result = await asyncio.wait_for(future, timeout=2.0)
    assert isinstance(result, ChannelOpened)
    assert result.reader is reader
    assert result.writer is writer


@pytest.mark.asyncio
async def test_open_channel_failure(system: ActorSystem) -> None:
    conn = _mock_conn()

    async def mock_open(host: str, port: int) -> tuple[MagicMock, MagicMock]:
        raise OSError("connection refused")

    conn.open_connection = mock_open

    ref = system.spawn(
        ssh_transport(host="x", user="u", key_path="k", connect_fn=lambda: _as_coro(conn)),
        "transport",
    )
    await asyncio.sleep(0.2)

    future = asyncio.get_event_loop().create_future()
    reply = MagicMock()
    reply.tell = lambda msg: future.set_result(msg) if not future.done() else None

    ref.tell(OpenChannel(remote_host="127.0.0.1", remote_port=8080, reply_to=reply))

    result = await asyncio.wait_for(future, timeout=2.0)
    assert isinstance(result, ChannelFailed)
    assert "connection refused" in result.error
