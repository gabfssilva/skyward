"""Novita.ai API response types.

TypedDicts for API responses - no conversion needed.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class ProductResponse(TypedDict):
    """GPU product from the Novita catalog."""

    id: str
    name: str
    cpuPerGpu: int
    memoryPerGpu: int
    diskPerGpu: NotRequired[int]
    price: str
    spotPrice: NotRequired[str]
    billingMethods: NotRequired[list[str]]
    availableDeploy: NotRequired[bool]
    inventoryState: NotRequired[str]
    regions: NotRequired[list[str]]
    minRootFS: NotRequired[int]
    maxRootFS: NotRequired[int]
    canBuy: NotRequired[bool]


class ClusterResponse(TypedDict):
    """Cluster/region info from Novita."""

    id: str
    name: str
    availableGpuType: NotRequired[list[str]]
    supportNetworkStorage: NotRequired[bool]
    supportInstanceNetwork: NotRequired[bool]


class SSHComponent(TypedDict):
    """SSH connection component from instance details."""

    sshCommand: NotRequired[str]
    password: NotRequired[str]
    isRunning: NotRequired[bool]


class PortMapping(TypedDict):
    """Port mapping for an instance."""

    port: int
    endpoint: NotRequired[str]
    type: NotRequired[str]


class InstanceResponse(TypedDict):
    """Instance details from Novita API."""

    id: str
    status: str
    name: NotRequired[str]
    clusterId: NotRequired[str]
    clusterName: NotRequired[str]
    productId: NotRequired[str]
    productName: NotRequired[str]
    imageUrl: NotRequired[str]
    cpuNum: NotRequired[str]
    memory: NotRequired[str]
    gpuNum: NotRequired[str]
    billingMode: NotRequired[str]
    spotStatus: NotRequired[str]
    sshPassword: NotRequired[str]
    connectComponentSSH: NotRequired[SSHComponent]
    portMappings: NotRequired[list[PortMapping]]


class CreateInstanceResponse(TypedDict):
    """Response from instance creation."""

    id: NotRequired[str]
    instanceId: NotRequired[str]
    error: NotRequired[str]
    message: NotRequired[str]


class DataListResponse(TypedDict):
    """Wrapper for list endpoints — Novita wraps all lists in ``data``."""

    data: NotRequired[list]


def parse_ssh_command(ssh_command: str) -> tuple[str, int]:
    """Extract host and port from an SSH command string.

    Parameters
    ----------
    ssh_command
        SSH command like ``"ssh -p 12345 root@proxy.novita.ai"``.

    Returns
    -------
    tuple[str, int]
        Host and port extracted from the command.

    Raises
    ------
    ValueError
        If the SSH command cannot be parsed.
    """
    parts = ssh_command.strip().split()
    port = 22
    host = ""

    for i, part in enumerate(parts):
        match part:
            case "-p" if i + 1 < len(parts):
                port = int(parts[i + 1])
            case s if "@" in s:
                host = s.split("@", 1)[1]

    if not host:
        raise ValueError(f"Cannot parse SSH command: {ssh_command!r}")

    return host, port
