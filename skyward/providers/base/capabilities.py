"""Capability protocols for provider features.

These protocols declare optional capabilities that providers may implement.
Use isinstance() or hasattr() checks to test for capability support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from skyward.types import Instance
    from skyward.volume import Volume


class VolumeCapable(Protocol):
    """Provider supports persistent volumes.

    Volumes can be attached to instances for persistent storage.
    Currently supports S3Volume via AWS Mountpoint.
    """

    def mount_volumes(
        self,
        instance: Instance,
        volumes: list[Volume],
    ) -> None:
        """Mount volumes on instance.

        Args:
            instance: Target instance.
            volumes: List of volumes to mount.
        """
        ...


class MIGCapable(Protocol):
    """Provider supports NVIDIA Multi-Instance GPU partitioning.

    MIG allows partitioning a single GPU into multiple smaller
    GPU instances for workload isolation and resource efficiency.
    """

    def configure_mig(
        self,
        instance: Instance,
        partitions: list[tuple[str, int]],
    ) -> None:
        """Configure MIG partitions on instance.

        Args:
            instance: Target instance with MIG-capable GPU.
            partitions: List of (profile, count) tuples.
                       e.g., [("1g.5gb", 7)] for H100
        """
        ...


class PlacementCapable(Protocol):
    """Provider supports placement groups for low-latency networking.

    Placement groups ensure instances are co-located on the same
    network segment for optimal inter-node communication.
    """

    def create_placement_group(self, name: str) -> str:
        """Create or get existing placement group.

        Args:
            name: Placement group name.

        Returns:
            Placement group ID or name.
        """
        ...

    def delete_placement_group(self, name: str) -> None:
        """Delete placement group.

        Args:
            name: Placement group name.
        """
        ...
