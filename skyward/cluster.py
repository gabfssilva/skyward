"""Cluster information for distributed execution.

Provides cluster information access for distributed execution.

Example:
    from skyward import ComputePool, AWS, compute_pool, compute

    @compute
    def train():
        info = compute_pool()
        if info.is_head:
            print(f"Head node, total_nodes={info.total_nodes}")

    pool = ComputePool(
        provider=AWS(),
        nodes=4,
        accelerator="A100",
        pip=["torch"],
    )

    with pool:
        train() >> pool
"""

from __future__ import annotations

from typing import Self, TypedDict

from pydantic import BaseModel, Field

# =============================================================================
# TypedDicts for structured data
# =============================================================================


class PeerInfo(TypedDict, total=False):
    """Information about a peer node in the compute pool."""

    node: int
    private_ip: str
    public_ip: str | None


class AcceleratorInfo(TypedDict, total=False):
    """Accelerator configuration for a compute pool."""

    type: str
    count: int
    memory_gb: float
    is_trainium: bool


class NetworkInfo(TypedDict, total=False):
    """Network configuration for a compute pool."""

    interface: str
    bandwidth_gbps: float


# =============================================================================
# Pydantic Models
# =============================================================================


class InstanceInfo(BaseModel):
    """Information about the current compute pool, parsed from COMPUTE_POOL env var.

    This class provides a typed interface to access pool information
    inside a distributed function.

    Example:
        from skyward import compute_pool

        pool = compute_pool()
        if pool and pool.is_head:
            print(f"I am head node of {pool.total_nodes}")
    """

    node: int = Field(description="Index of this node (0 to total_nodes - 1)")
    total_nodes: int = Field(description="Total number of nodes in the pool")
    accelerators: int = Field(description="Number of accelerators on this node")
    total_accelerators: int = Field(description="Total accelerators in the pool")
    head_addr: str = Field(description="IP address of the head node")
    head_port: int = Field(description="Port for head node coordination")
    job_id: str = Field(description="Unique identifier for this pool execution")

    peers: list[PeerInfo] = Field(description="Information about all peers")
    accelerator: AcceleratorInfo | None = Field(default=None, description="Accelerator configuration")
    network: NetworkInfo = Field(description="Network configuration")

    @property
    def is_trainium(self) -> bool:
        """Check if using AWS Trainium accelerator."""
        if self.accelerator is None:
            return False
        return bool(self.accelerator.get("is_trainium", False))

    @classmethod
    def current(cls) -> Self | None:
        """Get pool info from COMPUTE_POOL environment variable.

        Returns:
            ComputePoolInfo parsed from the environment, or None if not in a pool.
        """
        import os

        cluster_info_json = os.environ.get("COMPUTE_POOL")

        if not cluster_info_json:
            return None

        return cls.model_validate_json(cluster_info_json)

    @property
    def is_head(self) -> bool:
        """Check if this is the head node."""
        return self.node == 0


def instance_info() -> InstanceInfo | None:
    """Get the current pool info from environment.

    Convenience function equivalent to InstanceInfo.current().

    Returns:
        InstanceInfo for the current node, or None if not in a pool.

    Example:
        from skyward import compute_pool

        pool = compute_pool()
        if pool and pool.is_head:
            print("I am the head node")
    """
    return InstanceInfo.current()


__all__ = [
    "PeerInfo",
    "AcceleratorInfo",
    "NetworkInfo",
    "InstanceInfo",
    "instance_info",
]
