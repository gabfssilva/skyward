import asyncio
import uuid
from pathlib import Path

import pytest
from casty import InMemoryJournal

from skyward.daemon.protocol import DaemonError, GetNodeCount, Ping, Pong
from skyward.daemon.server import DaemonServer
from skyward.daemon.wire import async_recv, async_send


def _short_sock() -> Path:
    """Short socket path to stay under macOS 104-char AF_UNIX limit."""
    return Path(f"/tmp/sky-test-{uuid.uuid4().hex[:8]}.sock")


class TestDaemonServer:
    @pytest.mark.asyncio
    async def test_ping_pong(self) -> None:
        sock_path = _short_sock()
        server = DaemonServer(socket_path=sock_path, journal=InMemoryJournal())

        async with server:
            reader, writer = await asyncio.open_unix_connection(sock_path)
            await async_send(writer, Ping())
            resp = await async_recv(reader)
            writer.close()

        assert isinstance(resp, Pong)

    @pytest.mark.asyncio
    async def test_unknown_pool_returns_error(self) -> None:
        sock_path = _short_sock()
        server = DaemonServer(socket_path=sock_path, journal=InMemoryJournal())

        async with server:
            reader, writer = await asyncio.open_unix_connection(sock_path)
            await async_send(writer, GetNodeCount(pool_name="nonexistent"))
            resp = await async_recv(reader)
            writer.close()

        assert isinstance(resp, DaemonError)
        assert "nonexistent" in resp.error
