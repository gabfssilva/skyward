"""Discovery utilities for cloud providers."""

from __future__ import annotations

from dataclasses import replace

from skyward.types import Instance


def assign_node_indices(instances: list[Instance]) -> tuple[Instance, ...]:
    """Sort instances by private_ip and assign node indices.

    This is the common final step in discover_peers() across all providers:
    1. Sort instances by private_ip for consistent ordering
    2. Reassign node indices based on sort order

    Args:
        instances: List of Instance objects (node field will be overwritten).

    Returns:
        Tuple of instances sorted by IP with sequential node indices.
    """
    instances.sort(key=lambda i: i.private_ip)
    return tuple(replace(inst, node=idx) for idx, inst in enumerate(instances))
