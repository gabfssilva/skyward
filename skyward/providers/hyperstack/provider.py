"""Hyperstack provider implementation.

Stateless provider — all mutable state flows through Cluster[HyperstackSpecific].
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import generate_key_name, get_local_ssh_key, get_ssh_key_path

from .client import HyperstackClient, get_api_key
from .config import Hyperstack
from .types import FlavorResponse, PricebookEntry, VMResponse, normalize_gpu_name

log = logger.bind(provider="hyperstack")

_PREFERRED_IMAGE = "Ubuntu Server 22.04 LTS R535 CUDA 12.2"


@dataclass(frozen=True, slots=True)
class HyperstackSpecific:
    """Hyperstack-specific cluster data flowing through Cluster[HyperstackSpecific]."""

    environment_name: str
    environment_id: int
    key_name: str
    flavor_name: str
    image_name: str
    region: str


class HyperstackProvider(Provider[Hyperstack, HyperstackSpecific]):
    """Stateless Hyperstack provider. Holds only immutable config."""

    def __init__(self, config: Hyperstack) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: Hyperstack) -> HyperstackProvider:
        return cls(config)

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        api_key = get_api_key(self._config)

        async with HyperstackClient(api_key, config=self._config) as client:
            flavors = await client.list_flavors(region=self._config.region)
            pricebook = await client.get_pricebook()

        price_map = _build_price_map(pricebook, self._config.region)

        for flavor in flavors:
            if not _matches_spec(flavor, spec):
                continue
            yield _to_offer(flavor, price_map)

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[HyperstackSpecific]:
        api_key = get_api_key(self._config)
        ssh_key_path = get_ssh_key_path()
        public_path, public_key = get_local_ssh_key()
        key_name = generate_key_name(public_path)

        env_name = f"skyward-{uuid.uuid4().hex[:8]}"
        region = self._config.region

        async with HyperstackClient(api_key, config=self._config) as client:
            env = await client.create_environment(env_name, region)
            env_id = env["id"]
            log.info(
                "Created environment {name} (id={eid})",
                name=env_name, eid=env_id,
            )

            await client.import_keypair(env_name, key_name, public_key)
            log.info("Imported SSH keypair {name}", name=key_name)

            image_name = await _resolve_image(client, region)
            log.info("Resolved image: {img}", img=image_name)

        flavor_name: str = offer.specific

        return Cluster(
            id=f"hyperstack-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="ubuntu",
            use_sudo=True,
            shutdown_command="sudo shutdown -h now",
            specific=HyperstackSpecific(
                environment_name=env_name,
                environment_id=env_id,
                key_name=key_name,
                flavor_name=flavor_name,
                image_name=image_name,
                region=region,
            ),
        )

    async def provision(
        self,
        cluster: Cluster[HyperstackSpecific],
        count: int,
    ) -> tuple[Cluster[HyperstackSpecific], Sequence[Instance]]:
        api_key = get_api_key(self._config)
        specific = cluster.specific
        instances: list[Instance] = []

        async with HyperstackClient(api_key, config=self._config) as client:
            remaining = count
            batch_idx = 0
            while remaining > 0:
                batch_size = min(remaining, 20)
                batch_idx += 1
                name = f"skyward-{cluster.id.split('-', 1)[-1]}-{batch_idx}"

                ttl = cluster.spec.ttl or self._config.instance_timeout
                user_data = _self_destruction_script(ttl, cluster.shutdown_command)

                payload = {
                    "name": name,
                    "environment_name": specific.environment_name,
                    "image_name": specific.image_name,
                    "flavor_name": specific.flavor_name,
                    "key_name": specific.key_name,
                    "assign_floating_ip": True,
                    "count": batch_size,
                    "user_data": user_data,
                }

                log.info(
                    "Creating {n} VMs (batch {b})",
                    n=batch_size, b=batch_idx,
                )
                result = await client.create_vms(payload)

                for vm in result.get("instances", []):
                    instances.append(
                        Instance(
                            id=str(vm["id"]),
                            status="provisioning",
                            offer=cluster.offer,
                            spot=False,
                            region=specific.region,
                        ),
                    )

                remaining -= batch_size

        return cluster, instances

    async def get_instance(
        self,
        cluster: Cluster[HyperstackSpecific],
        instance_id: str,
    ) -> tuple[Cluster[HyperstackSpecific], Instance | None]:
        api_key = get_api_key(self._config)

        async with HyperstackClient(api_key, config=self._config) as client:
            info = await client.get_vm(int(instance_id))

        if not info:
            return cluster, None

        match info.get("status", "").upper():
            case "SHUTOFF" | "HIBERNATED" | "ERROR" | "DELETING" | "DELETED":
                return cluster, None
            case "ACTIVE" if info.get("floating_ip"):
                return cluster, _build_instance(info, "provisioned", cluster)
            case _:
                return cluster, _build_instance(info, "provisioning", cluster)

    async def terminate(
        self,
        cluster: Cluster[HyperstackSpecific],
        instance_ids: tuple[str, ...],
    ) -> Cluster[HyperstackSpecific]:
        if not instance_ids:
            return cluster

        api_key = get_api_key(self._config)

        async with HyperstackClient(api_key, config=self._config) as client:

            async def _destroy(iid: str) -> None:
                try:
                    await client.delete_vm(int(iid))
                except Exception as e:
                    log.error("Failed to delete VM {iid}: {err}", iid=iid, err=e)

            await asyncio.gather(*(_destroy(iid) for iid in instance_ids))

        return cluster

    async def teardown(
        self, cluster: Cluster[HyperstackSpecific],
    ) -> Cluster[HyperstackSpecific]:
        api_key = get_api_key(self._config)

        async with HyperstackClient(api_key, config=self._config) as client:
            try:
                await client.delete_environment(cluster.specific.environment_id)
                log.info(
                    "Deleted environment {name}",
                    name=cluster.specific.environment_name,
                )
            except Exception as e:
                log.error(
                    "Failed to delete environment {name}: {err}",
                    name=cluster.specific.environment_name,
                    err=e,
                )

        return cluster


# =============================================================================
# Helpers
# =============================================================================


def _self_destruction_script(ttl: int, shutdown_command: str) -> str:
    from skyward.providers.bootstrap.compose import resolve
    from skyward.providers.bootstrap.ops import instance_timeout

    lines = ["#!/bin/bash", "set -e"]
    if ttl:
        lines.append(resolve(instance_timeout(ttl, shutdown_command=shutdown_command)))
    return "\n".join(lines) + "\n"


def _build_price_map(
    pricebook: list[PricebookEntry], region: str,
) -> dict[str, float]:
    """Build GPU name -> price/GPU/hr map filtered by region."""
    prices: dict[str, float] = {}
    for entry in pricebook:
        entry_region = entry.get("region_name", "")
        gpu_name = entry.get("gpu_name", "")
        price = entry.get("price_per_gpu_hr", 0.0)
        if entry_region.upper() == region.upper() and gpu_name and price:
            prices[gpu_name.upper()] = price
    return prices


def _matches_spec(flavor: FlavorResponse, spec: PoolSpec) -> bool:
    """Check if a flavor matches the requested spec."""
    if spec.accelerator_name:
        req_norm = normalize_gpu_name(spec.accelerator_name)
        flavor_gpu = normalize_gpu_name(flavor.get("gpu", ""))
        if req_norm not in flavor_gpu and flavor_gpu not in req_norm:
            return False
    if spec.vcpus and flavor.get("cpu", 0) < spec.vcpus:
        return False
    if spec.memory_gb and flavor.get("ram", 0) < spec.memory_gb:
        return False
    return not (spec.accelerator_count and flavor.get("gpu_count", 0) != spec.accelerator_count)


def _to_offer(flavor: FlavorResponse, price_map: dict[str, float]) -> Offer:
    """Convert a Hyperstack flavor to a Skyward Offer."""
    gpu_name = flavor.get("gpu", "")
    gpu_count = flavor.get("gpu_count", 0)
    accel = Accelerator(name=gpu_name, count=gpu_count) if gpu_name else None

    it = InstanceType(
        name=flavor["name"],
        accelerator=accel,
        vcpus=float(flavor.get("cpu", 0)),
        memory_gb=float(flavor.get("ram", 0)),
        architecture="x86_64",
        specific=None,
    )

    gpu_upper = gpu_name.upper()
    per_gpu = price_map.get(gpu_upper, 0.0)
    hourly = per_gpu * gpu_count

    return Offer(
        id=str(flavor["id"]),
        instance_type=it,
        spot_price=None,
        on_demand_price=hourly if hourly > 0 else None,
        billing_unit="hour",
        specific=flavor["name"],
    )


def _build_instance(
    info: VMResponse, status: InstanceStatus, cluster: Cluster[HyperstackSpecific],
) -> Instance:
    """Convert a Hyperstack VM response to a Skyward Instance."""
    return Instance(
        id=str(info["id"]),
        status=status,
        offer=cluster.offer,
        ip=info.get("floating_ip") or "",
        private_ip=info.get("fixed_ip"),
        ssh_port=22,
        spot=False,
        region=cluster.specific.region,
    )


async def _resolve_image(client: HyperstackClient, region: str) -> str:
    """Find the best Ubuntu + CUDA image available."""
    images = await client.list_images(region=region)

    for img in images:
        if img["name"] == _PREFERRED_IMAGE:
            return img["name"]

    # Fallback: any Ubuntu image with CUDA
    for img in images:
        name = img["name"].lower()
        if "ubuntu" in name and "cuda" in name:
            return img["name"]

    # Last resort: any Ubuntu image
    for img in images:
        if "ubuntu" in img["name"].lower():
            return img["name"]

    if images:
        return images[0]["name"]

    raise RuntimeError(f"No images found in region {region}")
