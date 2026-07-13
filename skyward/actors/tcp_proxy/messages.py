from __future__ import annotations

import asyncio
from dataclasses import dataclass

from casty import ActorRef

from skyward.actors.messages import NodeId


@dataclass(frozen=True, slots=True)
class NodeUp:
    """Pool → proxy: a ready node joined the rotation."""

    node_id: NodeId
    transport_ref: ActorRef


@dataclass(frozen=True, slots=True)
class NodeDown:
    """Pool → proxy: a node left the rotation."""

    node_id: NodeId


@dataclass(frozen=True, slots=True)
class StopProxy:
    """Stop the proxy and close its listener."""


@dataclass(frozen=True, slots=True)
class _Accepted:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter


@dataclass(frozen=True, slots=True)
class _BridgeClosed:
    node_id: NodeId
    error: str | None = None


type ProxyMsg = NodeUp | NodeDown | StopProxy | _Accepted | _BridgeClosed
