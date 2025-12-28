"""AWS-specific execution resolvers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from skyward.providers.common import PeerInfo
from skyward.providers.common import run as common_run

if TYPE_CHECKING:
    from skyward.types import Instance


def _resolve_peers_aws(instances: tuple[Instance, ...]) -> tuple[PeerInfo, ...]:
    """Build peer info from AWS instances."""
    return tuple(
        PeerInfo(
            node=i,
            addr=inst.private_ip,
            instance_id=inst.id,
            private_ip=inst.private_ip,
        )
        for i, inst in enumerate(instances)
    )


def _resolve_head_addr_aws(instances: tuple[Instance, ...]) -> str:
    """Get head node address for AWS cluster."""
    return instances[0].private_ip


# Create partially applied run function for AWS
run = partial(
    common_run,
    resolve_peers=_resolve_peers_aws,
    resolve_head_addr=_resolve_head_addr_aws,
)
