from __future__ import annotations

import asyncio
from dataclasses import dataclass

from casty import ActorRef

from skyward.actors.messages import NodeId
from skyward.api.spec import Route


@dataclass(frozen=True, slots=True)
class ProxyState:
    """State of a single port proxy.

    Parameters
    ----------
    remote_port : int
        Port to reach on the node (over its SSH connection).
    server : asyncio.Server
        Local listener bound to ``127.0.0.1:<local>``.
    route : Route
        Selection policy across ready nodes.
    nodes : tuple[tuple[NodeId, ActorRef], ...]
        Currently-ready nodes paired with their SSH transport refs.
    cursor : int
        Monotonic counter; the active index is ``cursor % len(nodes)`` so it
        survives changes to the node set without resetting.
    """

    remote_port: int
    server: asyncio.Server
    route: Route = "round_robin"
    nodes: tuple[tuple[NodeId, ActorRef], ...] = ()
    cursor: int = 0
