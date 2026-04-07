"""TypedDict shapes for Lambda Cloud API responses."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class InstanceSpecs(TypedDict):
    vcpus: int
    memory_gib: int
    storage_gib: int
    gpus: int


class InstanceTypeInfo(TypedDict):
    name: str
    description: str
    gpu_description: str
    price_cents_per_hour: int
    specs: InstanceSpecs


class RegionInfo(TypedDict):
    name: str
    description: str


class InstanceTypeEntry(TypedDict):
    instance_type: InstanceTypeInfo
    regions_with_capacity_available: list[RegionInfo]


class SSHKeyResponse(TypedDict):
    id: str
    name: str
    public_key: NotRequired[str]


class InstanceResponse(TypedDict):
    id: str
    name: NotRequired[str]
    status: str
    ip: NotRequired[str]
    private_ip: NotRequired[str]
    hostname: NotRequired[str]
    ssh_key_names: NotRequired[list[str]]
    file_system_names: NotRequired[list[str]]
    region: NotRequired[RegionInfo]
    instance_type: NotRequired[InstanceTypeInfo]
    jupyter_token: NotRequired[str]
    is_reserved: NotRequired[bool]


class LaunchResponse(TypedDict):
    instance_ids: list[str]
