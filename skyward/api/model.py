from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from skyward.api.spec import Architecture, PoolSpec

if TYPE_CHECKING:
    from skyward.accelerators import Accelerator
    from skyward.api.spec import Volume
    from skyward.storage import Storage

type InstanceStatus = Literal[
    "provisioning",
    "provisioned",
    "boostrapping",
    "boostrapped",
    "ready"
]
"""Lifecycle status of an individual instance.

Progression: ``provisioning`` -> ``provisioned`` -> ``bootstrapping``
-> ``bootstrapped`` -> ``ready``.
"""


@dataclass(frozen=True, slots=True)
class InstanceType[S]:
    """Normalized machine type -- cacheable, provider-agnostic hardware description.

    Parameters
    ----------
    name
        Provider-specific instance type name (e.g., ``"p4d.24xlarge"``).
    accelerator
        GPU/accelerator spec, or ``None`` for CPU-only instances.
    vcpus
        Number of virtual CPUs.
    memory_gb
        System RAM in gigabytes.
    architecture
        CPU architecture (``"x86_64"`` or ``"arm64"``).
    specific
        Provider-specific metadata (generic type ``S``).
    """
    name: str
    accelerator: Accelerator | None
    vcpus: float
    memory_gb: float
    architecture: Architecture
    specific: S


@dataclass(frozen=True, slots=True)
class Offer[S]:
    """Ephemeral availability + pricing for a specific instance type.

    Returned by ``Provider.offers()`` and used by the pool to select
    the best instance for provisioning.

    Parameters
    ----------
    id
        Provider-specific offer identifier.
    instance_type
        The hardware specification this offer provides.
    spot_price
        Spot/preemptible price per billing unit, or ``None`` if unavailable.
    on_demand_price
        On-demand price per billing unit, or ``None`` if unavailable.
    billing_unit
        Granularity of pricing (``"second"``, ``"minute"``, or ``"hour"``).
    specific
        Provider-specific metadata (generic type ``S``).
    """
    id: str
    instance_type: InstanceType
    spot_price: float | None
    on_demand_price: float | None
    billing_unit: Literal["second", "minute", "hour"]
    specific: S


@dataclass(frozen=True, slots=True)
class Instance:
    """A provisioned compute instance in a cluster.

    Parameters
    ----------
    id
        Provider-specific instance identifier.
    status
        Current lifecycle status.
    offer
        The offer this instance was provisioned from.
    ip
        Public IP address, or ``None`` if not yet assigned.
    private_ip
        Private/VPC IP address, or ``None`` if not available.
    ssh_port
        SSH port number. Default ``22``.
    spot
        Whether this is a spot/preemptible instance.
    region
        Cloud region where the instance is running.
    """
    id: str
    status: InstanceStatus
    offer: Offer
    ip: str | None = None
    private_ip: str | None = None
    ssh_port: int = 22
    spot: bool = False
    region: str = ""


type ClusterStatus = Literal[
    "setting_up",
    "provisioning",
    "bootstrapping",
    "ready",
    "shutting_down",
    "destroyed",
]
"""Lifecycle status of the cluster as a whole.

Progression: ``setting_up`` -> ``provisioning`` -> ``bootstrapping``
-> ``ready`` -> ``shutting_down`` -> ``destroyed``.
"""


@dataclass(frozen=True, slots=True)
class Cluster[S]:
    """Full cluster state -- spec + offer + instances + provider metadata.

    Passed to provider methods and plugin hooks as the authoritative
    view of the current cluster configuration.

    Parameters
    ----------
    id
        Unique cluster identifier.
    status
        Current cluster lifecycle status.
    spec
        The pool specification that created this cluster.
    offer
        The selected offer for this cluster.
    ssh_key_path
        Path to the SSH private key for node access.
    ssh_user
        SSH username for node access.
    use_sudo
        Whether commands require ``sudo`` on remote nodes.
    shutdown_command
        Shell command used to shutdown instances.
    specific
        Provider-specific cluster metadata (generic type ``S``).
    instances
        Current instances in the cluster.
    prebaked
        Whether the cluster uses a pre-baked AMI/snapshot.
    resolved_volumes
        Volume-storage pairs after provider resolution.
    """
    id: str
    status: ClusterStatus
    spec: PoolSpec
    offer: Offer
    ssh_key_path: str
    ssh_user: str
    use_sudo: bool
    shutdown_command: str
    specific: S
    instances: tuple[Instance, ...] = ()
    prebaked: bool = False
    resolved_volumes: tuple[tuple[Volume, Storage], ...] | None = None
