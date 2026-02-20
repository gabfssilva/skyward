from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import ensure_ssh_key_on_provider, get_ssh_key_path

from .client import VerdaClient, VerdaError
from .config import Verda
from .types import (
    InstanceResponse,
    InstanceTypeResponse,
    get_accelerator,
    get_accelerator_count,
    get_accelerator_memory_gb,
    get_memory_gb,
    get_price_on_demand,
    get_price_spot,
    get_vcpu,
)

log = logger.bind(provider="verda")


@dataclass(frozen=True, slots=True)
class VerdaSpecific:
    """Verda-specific cluster data flowing through Cluster[VerdaSpecific]."""

    ssh_key_id: str
    startup_script_id: str
    instance_type: str
    os_image: str
    region: str
    hourly_rate: float
    on_demand_rate: float
    gpu_count: int
    gpu_model: str
    gpu_vram_gb: int
    vcpus: int
    memory_gb: float


class VerdaProvider(Provider[Verda, VerdaSpecific]):
    """Stateless Verda provider. Holds only immutable config + client."""

    def __init__(self, config: Verda, client: VerdaClient) -> None:
        self._config = config
        self._client = client

    @classmethod
    async def create(cls, config: Verda) -> VerdaProvider:
        from skyward.infra.http import HttpClient, OAuth2Auth

        from .client import VERDA_API_BASE, get_credentials

        client_id = config.client_id
        client_secret = config.client_secret
        if not client_id or not client_secret:
            client_id, client_secret = get_credentials()

        auth = OAuth2Auth(client_id, client_secret, f"{VERDA_API_BASE}/oauth2/token")
        http_client = HttpClient(VERDA_API_BASE, auth, timeout=config.request_timeout)
        client = VerdaClient(http_client)
        return cls(config, client)

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        use_spot = spec.allocation in ("spot", "spot-if-available")

        instance_types = await self._client.list_instance_types()
        availability = await self._client.get_availability(is_spot=use_spot)
        log.debug(
            "Offers query: {n} regions, {t} instance types",
            n=len(availability), t=len(instance_types),
        )

        available_types = {t for region_types in availability.values() for t in region_types}

        def _matches(itype: InstanceTypeResponse) -> bool:
            if itype["instance_type"] not in available_types:
                return False
            if not spec.accelerator_name:
                return True
            accel = get_accelerator(itype)
            if not accel:
                return False
            return (
                accel.upper() in spec.accelerator_name.upper()
                or spec.accelerator_name.upper() in accel.upper()
            )

        candidates = [itype for itype in instance_types if _matches(itype)]

        def sort_key(it: InstanceTypeResponse) -> float:
            price = get_price_spot(it) if use_spot else get_price_on_demand(it)
            return price if price is not None else float("inf")

        candidates.sort(key=sort_key)

        for itype_data in candidates:
            itype_name = itype_data["instance_type"]
            gpu_name = get_accelerator(itype_data)
            gpu_count = get_accelerator_count(itype_data)
            gpu_vram_gb = int(get_accelerator_memory_gb(itype_data))

            accel = (
                Accelerator(
                    name=gpu_name,
                    memory=f"{gpu_vram_gb}GB" if gpu_vram_gb else "",
                    count=gpu_count,
                )
                if gpu_name and gpu_count > 0
                else None
            )

            it = InstanceType(
                name=itype_name,
                accelerator=accel,
                vcpus=float(get_vcpu(itype_data)),
                memory_gb=get_memory_gb(itype_data),
                architecture="x86_64",
                specific=itype_data,
            )

            spot_price = get_price_spot(itype_data)
            on_demand_price = get_price_on_demand(itype_data)
            os_image = _select_os_image(spec, itype_data.get("supported_os", []))

            yield Offer(
                id=f"verda-{itype_name}",
                instance_type=it,
                spot_price=spot_price,
                on_demand_price=on_demand_price,
                billing_unit="hour",
                specific=os_image,
            )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[VerdaSpecific]:
        ssh_key_id = await ensure_ssh_key_on_provider(
            list_keys_fn=self._client.list_ssh_keys,  # type: ignore[reportArgumentType]
            create_key_fn=lambda name, key: self._client.create_ssh_key(name, key),  # type: ignore[reportArgumentType]
            provider_name="verda",
        )
        log.debug("SSH key ensured: id={kid}", kid=ssh_key_id)
        ssh_key_path = get_ssh_key_path()

        instance_type = offer.instance_type.name
        os_image = offer.specific if isinstance(offer.specific, str) else "ubuntu-22.04"

        use_spot = spec.allocation in ("spot", "spot-if-available")
        hourly_rate = (
            offer.spot_price if use_spot and offer.spot_price else offer.on_demand_price
        ) or 0.0

        gpu_count = offer.instance_type.accelerator.count if offer.instance_type.accelerator else 0
        gpu_model = offer.instance_type.accelerator.name if offer.instance_type.accelerator else ""
        gpu_vram_gb = 0
        if offer.instance_type.accelerator and offer.instance_type.accelerator.memory:
            mem = offer.instance_type.accelerator.memory
            gpu_vram_gb = int(mem.replace("GB", "")) if mem.endswith("GB") else 0

        ttl = spec.ttl or self._config.instance_timeout
        startup_content = _self_destruction_script(ttl, "shutdown -h now")
        script_name = f"skyward-startup-verda-{uuid.uuid4().hex[:8]}"
        script = await self._client.create_startup_script(script_name, startup_content)

        return Cluster(
            id=f"verda-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="root",
            use_sudo=True,
            shutdown_command="shutdown -h now",
            specific=VerdaSpecific(
                ssh_key_id=ssh_key_id,
                startup_script_id=script["id"],
                instance_type=instance_type,
                os_image=os_image,
                region=self._config.region,
                hourly_rate=hourly_rate,
                on_demand_rate=offer.on_demand_price or 0.0,
                gpu_count=gpu_count,
                gpu_model=gpu_model,
                gpu_vram_gb=gpu_vram_gb,
                vcpus=int(offer.instance_type.vcpus),
                memory_gb=offer.instance_type.memory_gb,
            ),
        )

    async def provision(
        self, cluster: Cluster[VerdaSpecific], count: int,
    ) -> tuple[Cluster[VerdaSpecific], Sequence[Instance]]:
        specific = cluster.specific
        use_spot = cluster.spec.allocation in ("spot", "spot-if-available")

        instances: list[Instance] = []
        for i in range(count):
            log.info("Provisioning instance {i}/{count}", i=i + 1, count=count)
            region = await _find_available_region(
                self._client, specific.instance_type, use_spot, specific.region,
            )

            hostname = f"skyward-{cluster.id}-{i}"
            try:
                resp = await self._client.create_instance(
                    instance_type=specific.instance_type,
                    image=specific.os_image,
                    ssh_key_ids=[specific.ssh_key_id],
                    location=region,
                    hostname=hostname,
                    description=f"Skyward managed - cluster {cluster.id}",
                    startup_script_id=specific.startup_script_id,
                    is_spot=use_spot,
                )
            except VerdaError as e:
                log.error("Failed to create instance: {err}", err=e)
                continue

            instances.append(Instance(
                id=str(resp["id"]),
                status="provisioning",
                offer=cluster.offer,
                spot=use_spot,
                region=region,
            ))

        if not instances:
            raise RuntimeError(
                f"Failed to provision any of the {count} requested instances",
            )

        return cluster, instances

    async def get_instance(
        self, cluster: Cluster[VerdaSpecific], instance_id: str,
    ) -> tuple[Cluster[VerdaSpecific], Instance | None]:
        info = await self._client.get_instance(instance_id)
        if not info:
            return cluster, None

        match info["status"]:
            case "error" | "discontinued" | "deleted":
                return cluster, None
            case "running" if info.get("ip"):
                return cluster, _build_verda_instance(info, "provisioned", cluster)
            case _:
                return cluster, _build_verda_instance(info, "provisioning", cluster)

    async def terminate(
        self, cluster: Cluster[VerdaSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[VerdaSpecific]:
        for iid in instance_ids:
            try:
                await self._client.delete_instance(iid)
            except Exception as e:
                log.error("Failed to delete instance {iid}: {err}", iid=iid, err=e)
        return cluster

    async def teardown(self, cluster: Cluster[VerdaSpecific]) -> Cluster[VerdaSpecific]:
        try:
            await self._client.delete_startup_script(cluster.specific.startup_script_id)
        except Exception as e:
            log.error("Failed to delete startup script: {err}", err=e)
        await self._client.close()
        return cluster


def _self_destruction_script(ttl: int, shutdown_command: str) -> str:
    from skyward.providers.bootstrap.compose import resolve
    from skyward.providers.bootstrap.ops import instance_timeout

    lines = ["#!/bin/bash", "set -e"]
    if ttl:
        lines.append(resolve(instance_timeout(ttl, shutdown_command=shutdown_command)))
    return "\n".join(lines) + "\n"


def _build_verda_instance(
    info: InstanceResponse,
    status: InstanceStatus,
    cluster: Cluster[VerdaSpecific],
) -> Instance:
    return Instance(
        id=str(info["id"]),
        status=status,
        offer=cluster.offer,
        ip=str(info.get("ip", "")),
        ssh_port=22,
        spot=bool(info.get("is_spot", False)),
        region=cluster.specific.region,
    )


async def _resolve_instance_type(
    client: VerdaClient, spec: PoolSpec,
) -> tuple[str, str, InstanceTypeResponse]:
    use_spot = spec.allocation in ("spot", "spot-if-available")

    instance_types = await client.list_instance_types()
    availability = await client.get_availability(is_spot=use_spot)
    log.debug(
        "Availability: {n} regions, {t} instance types",
        n=len(availability), t=len(instance_types),
    )

    available_types = {t for region_types in availability.values() for t in region_types}

    def _matches(itype: InstanceTypeResponse) -> bool:
        if itype["instance_type"] not in available_types:
            return False
        if not spec.accelerator_name:
            return True
        accel = get_accelerator(itype)
        if not accel:
            return False
        return (
            accel.upper() in spec.accelerator_name.upper()
            or spec.accelerator_name.upper() in accel.upper()
        )

    candidates = [itype for itype in instance_types if _matches(itype)]
    log.debug("Found {n} matching instance type candidates", n=len(candidates))

    if not candidates:
        raise RuntimeError(f"No instance types match accelerator={spec.accelerator_name}")

    def sort_key(it: InstanceTypeResponse) -> float:
        price = get_price_spot(it) if use_spot else get_price_on_demand(it)
        return price if price is not None else float("inf")

    candidates.sort(key=sort_key)

    selected = candidates[0]
    os_image = _select_os_image(spec, selected.get("supported_os", []))

    log.debug(
        "Selected {itype} with image {img}",
        itype=selected["instance_type"], img=os_image,
    )
    return selected["instance_type"], os_image, selected


def _select_os_image(spec: PoolSpec, supported_os: list[str]) -> str:
    if not spec.accelerator_name:
        return supported_os[0] if supported_os else "ubuntu-22.04"

    def is_preferred(img: str) -> bool:
        lower = img.lower()
        return (
            lower.startswith("ubuntu-")
            and "cuda" in lower
            and not any(x in lower for x in ("kubernetes", "jupyter", "docker", "cluster", "open"))
        )

    def parse_version(img: str) -> tuple[int, int, int, int]:
        ubuntu = re.search(r"ubuntu-(\d+)\.(\d+)", img.lower())
        cuda = re.search(r"cuda-?(\d+)\.(\d+)", img.lower())
        return (
            int(cuda.group(1)) if cuda else 0,
            int(cuda.group(2)) if cuda else 0,
            int(ubuntu.group(1)) if ubuntu else 0,
            int(ubuntu.group(2)) if ubuntu else 0,
        )

    preferred = sorted(
        filter(is_preferred, supported_os),
        key=parse_version, reverse=True,
    )
    if preferred:
        return preferred[0]

    cuda_images = sorted(
        (os for os in supported_os if "cuda" in os.lower()),
        key=parse_version, reverse=True,
    )
    if cuda_images:
        return cuda_images[0]

    return supported_os[0] if supported_os else "ubuntu-22.04-cuda-12.1"


async def _find_available_region(
    client: VerdaClient, instance_type: str, is_spot: bool, preferred_region: str,
) -> str:
    availability = await client.get_availability(is_spot)

    if preferred_region in availability and instance_type in availability[preferred_region]:
        return preferred_region

    for region, types in availability.items():
        if instance_type in types:
            log.info("Auto-selected region {region}", region=region)
            return region

    raise RuntimeError(f"No region has instance type '{instance_type}' available")
