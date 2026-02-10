"""Shared utilities for building COMPUTE_POOL environment variable."""

from __future__ import annotations

from typing import Any

from skyward.cluster.info import AcceleratorInfo, InstanceInfo, NetworkInfo, PeerInfo


def build_pool_info(
    node: int,
    total_nodes: int,
    accelerator_count: int,
    total_accelerators: int,
    head_addr: str,
    head_port: int,
    job_id: str,
    peers: list[dict[str, Any]],
    accelerator_type: str | None = None,
    placement_group: str | None = None,
    worker: int = 0,
    workers_per_node: int = 1,
) -> InstanceInfo:
    """Build ComputePoolInfo for COMPUTE_POOL environment variable.

    This function creates a ComputePoolInfo instance that can be serialized
    to JSON and set as the COMPUTE_POOL environment variable on worker nodes.

    Args:
        node: This node's index (0 = head node).
        total_nodes: Total number of nodes in the cluster.
        accelerator_count: Number of accelerators on this node.
        total_accelerators: Total accelerators across all nodes.
        head_addr: IP address of the head node.
        head_port: Port for head node communication.
        job_id: Unique identifier for this job/cluster.
        peers: List of peer info dicts with keys: node, addr, instance_id.
        accelerator_type: Type of accelerator (e.g., "A100-80", "H100", "Trainium1").
        placement_group: Optional placement group name.
        worker: Worker index within this node (0-indexed, for MIG partitions).
        workers_per_node: Number of workers per node (e.g., 2 for MIG 3g.40gb).

    Returns:
        ComputePoolInfo instance ready to be serialized with .model_dump_json().
    """
    accelerator_info: AcceleratorInfo | None = None
    if accelerator_type is not None:
        is_trainium = accelerator_type in ("Trainium1", "Trainium2", "Trainium3")
        accelerator_info = AcceleratorInfo(
            type=accelerator_type,
            count=accelerator_count,
            is_trainium=is_trainium,
        )

    peer_infos: list[PeerInfo] = [
        PeerInfo(
            node=p.get("node", 0),
            private_ip=p.get("private_ip", p.get("addr", "")),
        )
        for p in peers
    ]

    network_info = NetworkInfo()
    if placement_group:
        network_info["interface"] = placement_group  # Using interface field for placement_group

    return InstanceInfo(
        node=node,
        worker=worker,
        total_nodes=total_nodes,
        workers_per_node=workers_per_node,
        accelerators=accelerator_count,
        total_accelerators=total_accelerators,
        head_addr=head_addr,
        head_port=head_port,
        job_id=job_id,
        peers=peer_infos,
        accelerator=accelerator_info,
        network=network_info,
    )
