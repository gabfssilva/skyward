"""TCP proxy — exposes a remote node port on a fixed local port.

Binds an asyncio TCP server on ``127.0.0.1:<local>`` and, for each
accepted local connection, round-robins across the ready nodes and
bridges bytes over that node's existing SSH connection via
``SshTransport.open_channel``. Bridges are tracked and cancelled on
``stop``.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Protocol

from skyward.api.spec import Route
from skyward.infra.ssh_transport import SshTransport
from skyward.observability.logger import logger

log = logger.bind(component="tcp_proxy")


class _Reader(Protocol):
    async def read(self, n: int) -> bytes: ...


class _Writer(Protocol):
    def write(self, data: bytes) -> None: ...
    async def drain(self) -> None: ...
    def write_eof(self) -> None: ...
    def close(self) -> None: ...


class TcpProxy:
    """Local TCP listener bridging connections to node ports over SSH."""

    def __init__(
        self,
        remote_port: int,
        local_port: int,
        route: Route = "round_robin",
        initial_nodes: tuple[tuple[int, SshTransport], ...] = (),
    ) -> None:
        self._remote_port = remote_port
        self._local_port = local_port
        self._route = route
        self._nodes = initial_nodes
        self._cursor = 0
        self._server: asyncio.Server | None = None
        self._bridges: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._accept, host="127.0.0.1", port=self._local_port,
        )
        log.info(
            "Proxy listening on 127.0.0.1:{lp} -> node :{rp} ({route})",
            lp=self._local_port, rp=self._remote_port, route=self._route,
        )

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

    def add_node(self, node_id: int, transport: SshTransport) -> None:
        if any(n == node_id for n, _ in self._nodes):
            return
        self._nodes = (*self._nodes, (node_id, transport))

    def remove_node(self, node_id: int) -> None:
        self._nodes = tuple((n, t) for n, t in self._nodes if n != node_id)

    def _select(self) -> tuple[int, SshTransport] | None:
        if not self._nodes:
            return None
        match self._route:
            case "round_robin":
                return self._nodes[self._cursor % len(self._nodes)]

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        node = self._select()
        if node is None:
            log.debug("No ready nodes; rejecting connection")
            with suppress(Exception):
                writer.close()
            return
        self._cursor += 1
        nid, transport = node
        task = asyncio.create_task(self._bridge(nid, transport, reader, writer))
        self._bridges.add(task)
        task.add_done_callback(self._bridges.discard)

    async def _bridge(
        self,
        node_id: int,
        transport: SshTransport,
        local_reader: asyncio.StreamReader,
        local_writer: asyncio.StreamWriter,
    ) -> None:
        try:
            remote_reader, remote_writer = await transport.open_channel(
                "127.0.0.1", self._remote_port,
            )
        except Exception as e:
            log.debug("Bridge via node {nid} failed to open: {err}", nid=node_id, err=str(e))
            with suppress(Exception):
                local_writer.close()
            return
        try:
            await _pump(local_reader, local_writer, remote_reader, remote_writer)
        finally:
            for w in (local_writer, remote_writer):
                with suppress(Exception):
                    w.close()


async def _pump(
    local_reader: _Reader,
    local_writer: _Writer,
    remote_reader: _Reader,
    remote_writer: _Writer,
) -> None:
    async def copy(src: _Reader, dst: _Writer) -> None:
        try:
            while data := await src.read(65536):
                dst.write(data)
                await dst.drain()
        finally:
            with suppress(Exception):
                dst.write_eof()

    await asyncio.gather(
        copy(local_reader, remote_writer),
        copy(remote_reader, local_writer),
        return_exceptions=True,
    )
