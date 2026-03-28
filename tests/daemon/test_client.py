import uuid
from pathlib import Path

import pytest
from casty import InMemoryJournal

from skyward.daemon.client import DaemonClient
from skyward.daemon.protocol import Pong
from skyward.daemon.server import DaemonServer


def _short_sock() -> Path:
    """Short socket path to stay under macOS 104-char AF_UNIX limit."""
    return Path(f"/tmp/sky-test-{uuid.uuid4().hex[:8]}.sock")


class TestDaemonClient:
    @pytest.mark.asyncio
    async def test_ping(self) -> None:
        sock = _short_sock()
        server = DaemonServer(socket_path=sock, journal=InMemoryJournal())

        async with server:
            async with DaemonClient(socket_path=sock) as client:
                resp = await client.ping()
                assert isinstance(resp, Pong)

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self) -> None:
        sock = _short_sock()
        server = DaemonServer(socket_path=sock, journal=InMemoryJournal())

        async with server:
            async with DaemonClient(socket_path=sock) as client:
                assert client._writer is not None
            assert client._writer is None
