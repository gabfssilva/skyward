"""Who to rent from, and what it takes to log in.

The factories below read the environment, and that is the whole reason they
exist: the daemon never does. An adapter is handed its credentials through
``create()`` and has no other way to get them, so somebody in the user's own
process has to look them up and send them — and that somebody should be code the
user can read, not a hidden chain inside a server they may not even be running.

Passing a credential explicitly always wins over the environment.
"""

from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Provider:
    """A provider kind, its credentials, and whatever configuration it reads."""

    kind: str
    credentials: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.kind)


def AWS(  # noqa: N802
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    session_token: str | None = None,
    region: str | None = None,
    regions: tuple[str, ...] | None = None,
    ami: str | None = None,
    ubuntu_version: str = "24.04",
    subnet_id: str | None = None,
    security_group_id: str | None = None,
    instance_profile_arn: str | None = None,
    username: str | None = None,
    instance_timeout: int = 300,
    request_timeout: int = 30,
    allocation_strategy: str = "price-capacity-optimized",
    exclude_burstable: bool = False,
    name: str = "",
) -> Provider:
    shared = _aws_shared()
    return _provider(
        "aws",
        name,
        _some(
            access_key_id=access_key_id or os.environ.get("AWS_ACCESS_KEY_ID") or shared.get("aws_access_key_id"),
            secret_access_key=secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY") or shared.get("aws_secret_access_key"),
            session_token=session_token or os.environ.get("AWS_SESSION_TOKEN") or shared.get("aws_session_token"),
        ),
        _some(
            regions=_one_or_many(region, regions, ("us-east-1",)),
            ami=ami,
            ubuntu_version=ubuntu_version,
            subnet_id=subnet_id,
            security_group_id=security_group_id,
            instance_profile_arn=instance_profile_arn,
            username=username,
            instance_timeout=instance_timeout,
            request_timeout=request_timeout,
            allocation_strategy=allocation_strategy,
            exclude_burstable=exclude_burstable,
        ),
    )


def GCP(  # noqa: N802
    service_account_json: str | None = None,
    project: str | None = None,
    zone: str | None = None,
    zones: tuple[str, ...] | None = None,
    network: str = "default",
    subnet: str | None = None,
    disk_size_gb: int = 200,
    disk_type: str = "pd-balanced",
    instance_timeout: int = 300,
    service_account: str | None = None,
    thread_pool_size: int = 8,
    name: str = "",
) -> Provider:
    """``GOOGLE_APPLICATION_CREDENTIALS`` names a file; its contents are what travel."""
    credentials = service_account_json or _file(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    return _provider(
        "gcp",
        name,
        _some(service_account_json=credentials),
        _some(
            project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            zones=_one_or_many(zone, zones, ("us-central1-a",)),
            network=network,
            subnet=subnet,
            disk_gb=disk_size_gb,
            disk_type=disk_type,
            instance_timeout=instance_timeout,
            service_account=service_account,
            thread_pool_size=thread_pool_size,
        ),
    )


def Hyperstack(  # noqa: N802
    api_key: str | None = None,
    region: str | tuple[str, ...] | None = None,
    image: str | None = None,
    network_optimised: bool = False,
    network_optimised_regions: tuple[str, ...] = ("CANADA-1", "US-1"),
    object_storage_region: str = "CANADA-1",
    object_storage_endpoint: str = "https://ca1.obj.nexgencloud.io",
    instance_timeout: int = 300,
    request_timeout: int = 30,
    teardown_timeout: int = 120,
    teardown_poll_interval: float = 2.0,
    name: str = "",
) -> Provider:
    return _provider(
        "hyperstack",
        name,
        _some(api_key=api_key or os.environ.get("HYPERSTACK_API_KEY")),
        _some(
            region=region,
            image=image,
            network_optimised=network_optimised,
            network_optimised_regions=network_optimised_regions,
            object_storage_region=object_storage_region,
            object_storage_endpoint=object_storage_endpoint,
            instance_timeout=instance_timeout,
            request_timeout=request_timeout,
            teardown_timeout=teardown_timeout,
            teardown_poll_interval=teardown_poll_interval,
        ),
    )


def JarvisLabs(  # noqa: N802
    api_key: str | None = None,
    region: str | None = None,
    template: str = "pytorch",
    storage_gb: int = 50,
    instance_timeout: int = 300,
    thread_pool_size: int = 8,
    name: str = "",
) -> Provider:
    return _provider(
        "jarvislabs",
        name,
        _some(api_key=api_key or os.environ.get("JL_API_KEY")),
        _some(
            region=region,
            template=template,
            storage_gb=storage_gb,
            instance_timeout=instance_timeout,
            thread_pool_size=thread_pool_size,
        ),
    )


def Lambda(  # noqa: N802
    api_key: str | None = None,
    region: str | None = None,
    request_timeout: int = 30,
    name: str = "",
) -> Provider:
    return _provider(
        "lambda",
        name,
        _some(api_key=api_key or os.environ.get("LAMBDA_API_KEY")),
        _some(region=region, request_timeout=request_timeout),
    )


def MassedCompute(  # noqa: N802
    api_key: str | None = None,
    image_id: int = 184,
    request_timeout: int = 30,
    name: str = "",
) -> Provider:
    return _provider(
        "massed_compute",
        name,
        _some(api_key=api_key or os.environ.get("MASSED_API_KEY")),
        _some(image_id=image_id, request_timeout=request_timeout),
    )


def Novita(  # noqa: N802
    api_key: str | None = None,
    cluster_id: str | None = None,
    rootfs_size: int = 50,
    docker_image: str | None = None,
    min_cuda_version: str | None = None,
    request_timeout: int = 30,
    name: str = "",
) -> Provider:
    return _provider(
        "novita",
        name,
        _some(api_key=api_key or os.environ.get("NOVITA_API_KEY")),
        _some(
            cluster_id=cluster_id,
            rootfs_size=rootfs_size,
            docker_image=docker_image,
            min_cuda_version=min_cuda_version,
            request_timeout=request_timeout,
        ),
    )


def RunPod(  # noqa: N802
    api_key: str | None = None,
    cluster_mode: str = "individual",
    cloud_type: str = "secure",
    base_image: str = "nvidia",
    container_image: str | None = None,
    ubuntu: str = "newest",
    container_disk_gb: int = 50,
    volume_gb: int = 20,
    volume_mount_path: str = "/workspace",
    data_center_ids: tuple[str, ...] | str = "global",
    country_codes: tuple[str, ...] | str | None = None,
    exclude_country_codes: tuple[str, ...] | str = (),
    ports: tuple[str, ...] = ("22/tcp",),
    bid_multiplier: float = 1.0,
    registry_auth: str | None = "docker hub",
    min_inet_down: float | None = None,
    min_inet_up: float | None = None,
    global_networking: bool | None = None,
    request_timeout: int = 30,
    cpu_clock: str = "3c",
    name: str = "",
) -> Provider:
    return _provider(
        "runpod",
        name,
        _some(api_key=api_key or os.environ.get("RUNPOD_API_KEY")),
        _some(
            cluster_mode=cluster_mode,
            cloud_type=cloud_type,
            base_image=base_image,
            container_image=container_image,
            ubuntu=ubuntu,
            container_disk_gb=container_disk_gb,
            volume_gb=volume_gb,
            volume_mount_path=volume_mount_path,
            data_center_ids=_as_tuple(data_center_ids, keep=("global",)),
            country_codes=_as_tuple(country_codes),
            exclude_country_codes=_as_tuple(exclude_country_codes),
            ports=ports,
            bid_multiplier=bid_multiplier,
            registry_auth=registry_auth,
            min_inet_down=min_inet_down,
            min_inet_up=min_inet_up,
            global_networking=global_networking,
            request_timeout=request_timeout,
            cpu_clock=cpu_clock,
        ),
    )


def Scaleway(  # noqa: N802
    secret_key: str | None = None,
    project_id: str | None = None,
    zone: str | None = None,
    zones: tuple[str, ...] | None = None,
    image: str | None = None,
    instance_timeout: int = 300,
    request_timeout: int = 30,
    name: str = "",
) -> Provider:
    return _provider(
        "scaleway",
        name,
        _some(
            secret_key=secret_key or os.environ.get("SCW_SECRET_KEY"),
            project_id=project_id or os.environ.get("SCW_DEFAULT_PROJECT_ID"),
        ),
        _some(
            zones=_one_or_many(zone, zones),
            image=image,
            instance_timeout=instance_timeout,
            request_timeout=request_timeout,
        ),
    )


def TensorDock(  # noqa: N802
    api_key: str | None = None,
    api_token: str | None = None,
    location: str | None = None,
    tier: int | None = None,
    storage_gb: int = 100,
    operating_system: str = "ubuntu2404",
    instance_timeout: int = 300,
    request_timeout: int = 120,
    min_ram_gb: int | None = None,
    min_vcpus: int | None = None,
    name: str = "",
) -> Provider:
    return _provider(
        "tensordock",
        name,
        _some(
            api_key=api_key or os.environ.get("TENSORDOCK_API_KEY"),
            api_token=api_token or os.environ.get("TENSORDOCK_API_TOKEN"),
        ),
        _some(
            location=location,
            tier=tier,
            storage_gb=storage_gb,
            operating_system=operating_system,
            instance_timeout=instance_timeout,
            request_timeout=request_timeout,
            min_ram_gb=min_ram_gb,
            min_vcpus=min_vcpus,
        ),
    )


def VastAI(  # noqa: N802
    api_key: str | None = None,
    min_reliability: float = 0.95,
    verified_only: bool = True,
    min_cuda: float = 12.0,
    geolocation: str | None = None,
    bid_multiplier: float = 1.2,
    instance_timeout: int = 300,
    request_timeout: int = 30,
    docker_image: str | None = None,
    disk_gb: int = 100,
    overlay_timeout: int = 120,
    require_direct_port: bool = False,
    min_inet_down: float | None = None,
    min_inet_up: float | None = None,
    limit: int | None = None,
    name: str = "",
) -> Provider:
    return _provider(
        "vastai",
        name,
        _some(api_key=api_key or os.environ.get("VAST_API_KEY")),
        _some(
            verified_only=verified_only,
            min_reliability=min_reliability,
            min_cuda=min_cuda,
            geolocation=geolocation,
            bid_multiplier=bid_multiplier,
            instance_timeout=instance_timeout,
            request_timeout=request_timeout,
            docker_image=docker_image,
            disk_gb=disk_gb,
            overlay_timeout=overlay_timeout,
            direct_port=require_direct_port,
            min_inet_down=min_inet_down,
            min_inet_up=min_inet_up,
            limit=limit,
        ),
    )


def Verda(  # noqa: N802
    client_id: str | None = None,
    client_secret: str | None = None,
    region: str = "FIN-01",
    ssh_key_id: str | None = None,
    image: str | None = None,
    cuda: str = "13.0",
    instance_timeout: int = 300,
    request_timeout: int = 30,
    name: str = "",
) -> Provider:
    return _provider(
        "verda",
        name,
        _some(
            client_id=client_id or os.environ.get("VERDA_CLIENT_ID"),
            client_secret=client_secret or os.environ.get("VERDA_CLIENT_SECRET"),
        ),
        _some(
            region=region,
            ssh_key_id=ssh_key_id,
            image=image,
            cuda=cuda,
            instance_timeout=instance_timeout,
            request_timeout=request_timeout,
        ),
    )


def Vultr(  # noqa: N802
    api_key: str | None = None,
    mode: str = "cloud",
    region: str | None = None,
    os_id: int = 2284,
    instance_timeout: int = 300,
    request_timeout: int = 30,
    name: str = "",
) -> Provider:
    return _provider(
        "vultr",
        name,
        _some(api_key=api_key or os.environ.get("VULTR_API_KEY")),
        _some(
            mode=mode,
            region=region,
            os_id=os_id,
            instance_timeout=instance_timeout,
            request_timeout=request_timeout,
        ),
    )


def Container(  # noqa: N802
    image: str = "ghcr.io/gabfssilva/skyward:py{python_version}",
    ssh_user: str = "root",
    binary: str = "docker",
    container_prefix: str | None = None,
    network: str | None = None,
    name: str = "",
) -> Provider:
    """Local containers. The one provider that needs no credentials."""
    return _provider(
        "container",
        name,
        {},
        _some(
            image=image,
            ssh_user=ssh_user,
            binary=binary,
            container_prefix=container_prefix,
            network=network,
        ),
    )


def _key(kind: str, variable: str, api_key: str | None, name: str) -> Provider:
    return _provider(kind, name, _some(api_key=api_key or os.environ.get(variable)), {})


def _provider(kind: str, name: str, credentials: dict[str, str], config: dict[str, Any]) -> Provider:
    return Provider(kind=kind, credentials=credentials, config=config, name=name)


def _some[T](**fields: T | None) -> dict[str, T]:
    """Only what was actually given; the adapter's own defaults answer for the rest."""
    return {name: value for name, value in fields.items() if value is not None}


def _as_tuple(value: tuple[str, ...] | str | None, *, keep: tuple[str, ...] = ()) -> tuple[str, ...] | str | None:
    """A lone string becomes a one-tuple, except the sentinels that must stay scalar.

    The provider factories accept ``"EU-RO-1"`` as shorthand for ``("EU-RO-1",)``
    so a single value reads naturally, while a sentinel like ``"global"`` is a mode,
    not a member, and is passed through untouched.
    """
    match value:
        case str() if value in keep:
            return value
        case str():
            return (value,)
        case _:
            return value


def _one_or_many(
    one: str | None,
    many: tuple[str, ...] | None,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...] | None:
    if one is not None and many is not None:
        raise ValueError("pass either the singular or plural region option, not both")
    if one is not None:
        return (one,)
    return many if many is not None else default


def _file(path: str | None) -> str | None:
    return Path(path).read_text() if path and Path(path).is_file() else None


def _aws_shared() -> dict[str, str]:
    """The static keys in ``~/.aws/credentials`` for ``AWS_PROFILE`` (default ``default``).

    Only static keys resolve here. SSO, assume-role, and process credentials need
    botocore's full chain, which the SDK deliberately does not depend on.
    """
    path = Path(os.environ.get("AWS_SHARED_CREDENTIALS_FILE") or Path.home() / ".aws" / "credentials")
    if not path.is_file():
        return {}
    parser = ConfigParser()
    parser.read(path)
    profile = os.environ.get("AWS_PROFILE", "default")
    return dict(parser[profile]) if parser.has_section(profile) else {}
