"""TCP proxy actor — exposes a remote node port on a fixed local port.

A port proxy tells this story: binding → bridging → stopped.

It binds an asyncio TCP server on ``127.0.0.1:<local>`` and, for each accepted
local connection, round-robins across the ready nodes and bridges bytes over
that node's existing SSH connection via an ``OpenChannel`` request to its
transport actor. The raw asyncssh connection never leaves the transport — only
the per-channel streams are pumped here.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import replace
from typing import Protocol

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import NodeId
from skyward.api.spec import Route
from skyward.observability.logger import logger

from .messages import NodeDown, NodeUp, ProxyMsg, StopProxy, _Accepted, _BridgeClosed
from .state import ProxyState

log = logger.bind(actor="tcp_proxy")


class _Reader(Protocol):
    async def read(self, n: int) -> bytes: ...


class _Writer(Protocol):
    def write(self, data: bytes) -> None: ...
    async def drain(self) -> None: ...
    def write_eof(self) -> None: ...
    def close(self) -> None: ...


def tcp_proxy(
    remote_port: int,
    local_port: int,
    route: Route = "round_robin",
    initial_nodes: tuple[tuple[NodeId, ActorRef], ...] = (),
) -> Behavior[ProxyMsg]:
    """binding → bridging → stopped."""

    async def _setup(ctx: ActorContext[ProxyMsg]) -> Behavior[ProxyMsg]:
        self_ref = ctx.self
        try:
            server = await asyncio.start_server(
                lambda r, w: self_ref.tell(_Accepted(reader=r, writer=w)),
                host="127.0.0.1",
                port=local_port,
            )
        except OSError as e:
            log.error("Port {lp} bind failed: {err}", lp=local_port, err=e)
            return Behaviors.stopped()

        log.info(
            "Proxy listening on 127.0.0.1:{lp} -> node :{rp} ({route})",
            lp=local_port, rp=remote_port, route=route,
        )
        return bridging(ProxyState(
            remote_port=remote_port, server=server,
            route=route, nodes=initial_nodes,
        ))

    return Behaviors.setup(_setup)


def bridging(s: ProxyState) -> Behavior[ProxyMsg]:
    async def receive(ctx: ActorContext[ProxyMsg], msg: ProxyMsg) -> Behavior[ProxyMsg]:
        match msg:
            case NodeUp(node_id=nid, transport_ref=tref):
                if any(n == nid for n, _ in s.nodes):
                    return Behaviors.same()
                return bridging(replace(s, nodes=(*s.nodes, (nid, tref))))

            case NodeDown(node_id=nid):
                return bridging(replace(
                    s, nodes=tuple((n, t) for n, t in s.nodes if n != nid),
                ))

            case _Accepted(reader=lr, writer=lw):
                node = _select(s)
                if node is None:
                    log.debug("No ready nodes; rejecting connection")
                    with suppress(Exception):
                        lw.close()
                    return Behaviors.same()
                nid, tref = node
                ctx.pipe_to_self(
                    _bridge_via(tref, lr, lw, s.remote_port),
                    mapper=lambda _: _BridgeClosed(node_id=nid),
                    on_failure=lambda e: _BridgeClosed(node_id=nid, error=str(e)),
                )
                return bridging(replace(s, cursor=s.cursor + 1))

            case _BridgeClosed(node_id=nid, error=error):
                if error:
                    log.debug("Bridge via node {nid} closed: {err}", nid=nid, err=error)
                return Behaviors.same()

            case StopProxy():
                return Behaviors.stopped()

        return Behaviors.same()

    async def _post_stop(_ctx: ActorContext[ProxyMsg]) -> None:
        s.server.close()
        with suppress(Exception):
            await s.server.wait_closed()

    return Behaviors.with_lifecycle(Behaviors.receive(receive), post_stop=_post_stop)


def _select(s: ProxyState) -> tuple[NodeId, ActorRef] | None:
    if not s.nodes:
        return None
    match s.route:
        case "round_robin":
            return s.nodes[s.cursor % len(s.nodes)]


async def _bridge_via(
    transport_ref: ActorRef,
    local_reader: asyncio.StreamReader,
    local_writer: asyncio.StreamWriter,
    remote_port: int,
) -> None:
    from skyward.infra.ssh_actor import ChannelFailed, ChannelOpened, transport_open_channel

    result = await transport_open_channel(transport_ref, "127.0.0.1", remote_port)
    match result:
        case ChannelFailed(error=error):
            with suppress(Exception):
                local_writer.close()
            raise RuntimeError(error)
        case ChannelOpened(reader=remote_reader, writer=remote_writer):
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
