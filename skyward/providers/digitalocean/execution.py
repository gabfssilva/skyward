"""DigitalOcean-specific execution resolvers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from skyward.providers.common import PeerInfo
from skyward.providers.common import run as common_run

if TYPE_CHECKING:
    from skyward.types import Instance


def _resolve_peers_do(instances: tuple[Instance, ...]) -> tuple[PeerInfo, ...]:
    """Build peer info from DigitalOcean instances."""
    return tuple(
        PeerInfo(
            node=i,
            addr=inst.get_meta("droplet_ip", inst.public_ip or inst.private_ip),
            instance_id=inst.id,
            private_ip=inst.private_ip,
        )
        for i, inst in enumerate(instances)
    )


def _resolve_head_addr_do(instances: tuple[Instance, ...]) -> str:
    """Get head node address for DigitalOcean cluster."""
    head = instances[0]
    addr: str = head.get_meta("droplet_ip", head.public_ip or head.private_ip)
    return addr


# Create partially applied run function for DigitalOcean
run = partial(
    common_run,
    resolve_peers=_resolve_peers_do,
    resolve_head_addr=_resolve_head_addr_do,
)
