"""The client half of a forwarded port.

A loopback TCP listener, and for every connection it accepts, one pair of HTTP
streams to the daemon — bytes up, bytes down — that carry the connection to a
node and its answer back. The daemon chooses the node and pumps the bytes; this
side only mints an id, opens the two streams, and closes them cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Protocol

from skyward.sdk.spec import Port

CHUNK = 65536


class Forwarding(Protocol):
    """The two calls a proxy makes on the client — up one way, down the other."""

    async def forward_up(self, compute: str, cid: str, port: int, route: str, chunks: AsyncIterator[bytes]) -> None: ...

    def forward_down(self, compute: str, cid: str) -> AsyncIterator[bytes]: ...


class TcpProxy:
    """A loopback port bridged to a node port, one connection at a time.

    Binds ``127.0.0.1:<local>`` and, for every accepted connection, opens an
    up-stream (this socket's bytes, as a request body) and a down-stream (the
    node's bytes, as a response), tied by an id it mints. The daemon round-robins
    the node; the two directions run until either end hangs up, and a half-close
    on one is not the end of the other.
    """

    def __init__(self, client: Forwarding, compute: str, port: Port) -> None:
        self._client = client
        self._compute = compute
        self._port = port
        self._server: asyncio.Server | None = None
        self._bridges: set[asyncio.Task[None]] = set()

    @property
    def local(self) -> int:
        """The bound local port — the OS-chosen one, when ``Port.local`` was ``0``."""
        if self._server is None or not self._server.sockets:
            raise RuntimeError("the proxy is not listening")
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", self._port.local)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
        for task in tuple(self._bridges):
            task.cancel()
        if self._bridges:
            await asyncio.gather(*self._bridges, return_exceptions=True)
        self._bridges.clear()
        if self._server is not None:
            with suppress(Exception):
                await self._server.wait_closed()
            self._server = None

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._bridge(reader, writer))
        self._bridges.add(task)
        task.add_done_callback(self._bridges.discard)

    async def _bridge(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        cid = uuid.uuid4().hex

        async def up() -> AsyncIterator[bytes]:
            with suppress(OSError):
                while data := await reader.read(CHUNK):
                    yield data

        async def down() -> None:
            with suppress(OSError):
                async for chunk in self._client.forward_down(self._compute, cid):
                    writer.write(chunk)
                    await writer.drain()

        try:
            await asyncio.gather(
                self._client.forward_up(self._compute, cid, self._port.remote, self._port.route, up()),
                down(),
                return_exceptions=True,
            )
        finally:
            with suppress(OSError):
                writer.close()
