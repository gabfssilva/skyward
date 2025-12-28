"""Protocols for provider-specific execution hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from skyward.types import Instance


@dataclass(frozen=True, slots=True)
class PeerInfo:
    """Peer node information for cluster setup."""

    node: int
    addr: str
    instance_id: str
    private_ip: str = ""


class PeerResolver(Protocol):
    """Protocol for resolving peer information from instances."""

    def __call__(self, instances: tuple[Instance, ...]) -> tuple[PeerInfo, ...]:
        """Build peer info from instances."""
        ...


class HeadAddrResolver(Protocol):
    """Protocol for resolving head node address."""

    def __call__(self, instances: tuple[Instance, ...]) -> str:
        """Return the head node's address for cluster communication."""
        ...
