"""AWS cluster state tracking.

Manages runtime state for AWS clusters including infrastructure
resources and instance information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from skyward.actors.messages import InstanceMetadata
from skyward.api.spec import PoolSpec


# =============================================================================
# Infrastructure Resources
# =============================================================================


@dataclass(frozen=True, slots=True)
class AWSResources:
    """Container for AWS infrastructure resource identifiers.

    These resources are created once per region and reused across clusters.
    """

    bucket: str
    iam_role_arn: str
    instance_profile_arn: str
    security_group_id: str
    region: str
    subnet_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, str | list[str]]:
        """Serialize to dictionary."""
        return {
            "bucket": self.bucket,
            "iam_role_arn": self.iam_role_arn,
            "instance_profile_arn": self.instance_profile_arn,
            "security_group_id": self.security_group_id,
            "region": self.region,
            "subnet_ids": list(self.subnet_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | list[str]]) -> AWSResources:
        """Deserialize from dictionary."""
        subnet_ids_raw = data.get("subnet_ids", [])
        match subnet_ids_raw:
            case list():
                subnet_ids = tuple(str(s) for s in subnet_ids_raw)
            case _:
                subnet_ids = (str(subnet_ids_raw),)

        return cls(
            bucket=str(data["bucket"]),
            iam_role_arn=str(data["iam_role_arn"]),
            instance_profile_arn=str(data["instance_profile_arn"]),
            security_group_id=str(data["security_group_id"]),
            region=str(data["region"]),
            subnet_ids=subnet_ids,
        )


# =============================================================================
# Instance Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class InstanceConfig:
    """Configuration for launching an EC2 instance.

    Contains all information needed to launch via EC2 Fleet.
    """

    instance_type: str
    ami: str
    spot: bool = True


# =============================================================================
# Cluster State
# =============================================================================


@dataclass(frozen=True, slots=True)
class AWSClusterState:
    """Runtime state for an AWS cluster.

    Tracks all information needed to manage a cluster\'s lifecycle
    including infrastructure resources, launched instances, and spec.
    """

    cluster_id: str
    spec: PoolSpec
    resources: AWSResources
    region: str
    ssh_key_name: str
    ssh_key_path: str
    username: str
    instances: MappingProxyType[str, InstanceMetadata] = field(default_factory=lambda: MappingProxyType({}))
    pending_nodes: frozenset[int] = frozenset()
    fleet_instance_ids: MappingProxyType[int, str] = field(default_factory=lambda: MappingProxyType({}))
