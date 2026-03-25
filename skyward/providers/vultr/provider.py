"""Vultr GPU provider implementation."""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.core import PoolSpec
from skyward.core.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import ensure_ssh_key_on_provider, get_ssh_key_path

from .client import VultrClient, VultrError, get_api_key
from .config import Vultr
from .types import (
    _GPU_MEMORY_MAP,
    _GPU_NAME_MAP,
    BareMetalCreateParams,
    BareMetalResponse,
    InstanceCreateParams,
    InstanceResponse,
    MetalPlanResponse,
    PlanResponse,
)

log = logger.bind(provider="vultr")

def _extract_gpu_count_from_plan_id(plan_id: str) -> int:
    """Extract GPU count from plan ID (e.g., 'vbm-8x-gpu-a100' -> 8)."""
    for part in plan_id.split("-"):
        if part.endswith("x") and part[:-1].isdigit():
            return int(part[:-1])
    return 1


def _extract_vram_from_plan_id(plan_id: str) -> int:
    """Extract allocated VRAM from cloud GPU plan ID.

    Vultr cloud GPU plan IDs encode the allocated VRAM:
    ``vcg-a40-2c-10g-4vram`` → 4 (GB).
    """
    if m := re.search(r"(\d+)vram$", plan_id):
        return int(m.group(1))
    return 0


@dataclass(frozen=True, slots=True)
class VultrSpecific:
    """Vultr-specific cluster data flowing through Cluster[VultrSpecific]."""

    ssh_key_id: str
    plan_id: str
    region: str
    is_bare_metal: bool


class VultrProvider(Provider[Vultr, VultrSpecific]):
    """Stateless Vultr provider. Holds only immutable config."""

    name = "vultr"

    def __init__(self, config: Vultr) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: Vultr) -> VultrProvider:
        return cls(config)

    @property
    def _is_bare_metal(self) -> bool:
        return self._config.mode == "bare-metal"

    async def offers(self) -> AsyncIterator[Offer]:
        api_key = get_api_key(self._config.api_key)
        async with VultrClient(api_key, timeout=self._config.request_timeout) as client:
            plans_iter = (
                self._offers_bare_metal(await client.list_metal_plans())
                if self._is_bare_metal
                else self._offers_cloud(await client.list_gpu_plans())
            )

        async for offer in plans_iter:
            yield offer

    async def _offers_cloud(
        self, plans: list[PlanResponse],
    ) -> AsyncIterator[Offer]:
        for plan in plans:
            gpu_type = plan.get("gpu_type", "")
            if not gpu_type:
                continue

            normalized = _GPU_NAME_MAP.get(gpu_type, gpu_type)
            vram = (
                _extract_vram_from_plan_id(plan["id"])
                or plan.get("gpu_vram_gb")
                or _GPU_MEMORY_MAP.get(normalized, 0)
            )

            locations = plan.get("locations", [])
            if self._config.region:
                if self._config.region not in locations:
                    continue
                locations = [self._config.region]

            hourly = plan.get("hourly_cost") or (
                round(plan.get("monthly_cost", 0) / 730, 4)
                if plan.get("monthly_cost") else None
            )

            full_vram = _GPU_MEMORY_MAP.get(normalized, 0)
            gpu_count = vram / full_vram if full_vram else 1

            accel = Accelerator(
                name=normalized,
                memory=f"{vram}GB" if vram else "",
                count=gpu_count,
            )
            it = InstanceType(
                name=plan["id"],
                accelerator=accel,
                vcpus=float(plan.get("vcpu_count", 0)),
                memory_gb=float(plan.get("ram", 0)) / 1024,
                architecture="x86_64",
                specific=None,
            )
            for region in locations:
                yield Offer(
                    id=f"vultr-{region}-{plan['id']}",
                    instance_type=it,
                    spot_price=None,
                    on_demand_price=hourly,
                    billing_unit="hour",
                    specific=None,
                )

    async def _offers_bare_metal(
        self, plans: list[MetalPlanResponse],
    ) -> AsyncIterator[Offer]:
        for plan in plans:
            gpu_type = plan.get("gpu_type", "")
            if not gpu_type:
                continue

            normalized = _GPU_NAME_MAP.get(gpu_type, gpu_type)
            vram = plan.get("gpu_vram_gb") or _GPU_MEMORY_MAP.get(normalized, 0)
            gpu_count = _extract_gpu_count_from_plan_id(plan["id"])

            locations = plan.get("locations", [])
            if self._config.region:
                if self._config.region not in locations:
                    continue
                locations = [self._config.region]

            monthly = plan.get("monthly_cost", 0)
            hourly = round(monthly / 730, 4) if monthly else None

            accel = Accelerator(
                name=normalized,
                memory=f"{vram}GB" if vram else "",
                count=gpu_count,
            )
            it = InstanceType(
                name=plan["id"],
                accelerator=accel,
                vcpus=float(plan.get("cpu_count", 0)),
                memory_gb=float(plan.get("ram", 0)) / 1024,
                architecture="x86_64",
                specific=None,
            )
            for region in locations:
                yield Offer(
                    id=f"vultr-{region}-{plan['id']}",
                    instance_type=it,
                    spot_price=None,
                    on_demand_price=hourly,
                    billing_unit="hour",
                    specific=None,
                )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[VultrSpecific]:
        api_key = get_api_key(self._config.api_key)
        ssh_key_path = get_ssh_key_path()

        async with VultrClient(api_key, timeout=self._config.request_timeout) as client:
            ssh_key_id = await ensure_ssh_key_on_provider(
                list_keys_fn=lambda: _list_keys_adapted(client),
                create_key_fn=lambda name, key: _create_key_adapted(client, name, key),
                provider_name="vultr",
            )
            log.debug("SSH key ensured: {kid}", kid=ssh_key_id)

        return Cluster(
            id=f"vultr-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="root",
            use_sudo=False,
            shutdown_command="shutdown -h now",
            specific=VultrSpecific(
                ssh_key_id=ssh_key_id,
                plan_id=offer.instance_type.name,
                region=_region_from_offer(offer),
                is_bare_metal=self._is_bare_metal,
            ),
        )

    async def provision(
        self, cluster: Cluster[VultrSpecific], count: int,
    ) -> tuple[Cluster[VultrSpecific], Sequence[Instance]]:
        api_key = get_api_key(self._config.api_key)
        specific = cluster.specific
        instances: list[Instance] = []

        async with VultrClient(api_key, timeout=self._config.request_timeout) as client:
            for i in range(count):
                try:
                    instance_id = await self._create_one(client, cluster, i)
                except VultrError as e:
                    log.error("Failed to create instance: {err}", err=e)
                    continue

                instances.append(Instance(
                    id=instance_id,
                    status="provisioning",
                    offer=cluster.offer,
                    region=specific.region,
                ))

        return cluster, instances

    async def _create_one(
        self, client: VultrClient, cluster: Cluster[VultrSpecific], index: int,
    ) -> str:
        specific = cluster.specific
        label = f"skyward-{cluster.id}-{index}"

        if specific.is_bare_metal:
            params: BareMetalCreateParams = {
                "region": specific.region,
                "plan": specific.plan_id,
                "os_id": self._config.os_id,
                "label": label,
                "hostname": label,
                "sshkey_id": [specific.ssh_key_id],
            }
            bm = await client.create_bare_metal(params)
            return bm["id"]

        params_cloud: InstanceCreateParams = {
            "region": specific.region,
            "plan": specific.plan_id,
            "os_id": self._config.os_id,
            "label": label,
            "hostname": label,
            "sshkey_id": [specific.ssh_key_id],
        }
        inst = await client.create_instance(params_cloud)
        return inst["id"]

    async def get_instance(
        self, cluster: Cluster[VultrSpecific], instance_id: str,
    ) -> tuple[Cluster[VultrSpecific], Instance | None]:
        api_key = get_api_key(self._config.api_key)
        async with VultrClient(api_key, timeout=self._config.request_timeout) as client:
            if cluster.specific.is_bare_metal:
                raw = await client.get_bare_metal(instance_id)
            else:
                raw = await client.get_instance(instance_id)

        if not raw:
            return cluster, None

        ip = raw.get("main_ip", "")
        status_str = raw.get("status", "")
        power = raw.get("power_status", "")

        log.debug(
            "Instance {id} status={status} power={power} ip={ip}",
            id=instance_id, status=status_str, power=power, ip=ip,
        )

        match status_str:
            case "active" if power == "running" and _has_ip(ip):
                return cluster, _build_instance(raw, "provisioned", cluster)
            case "active" | "pending":
                return cluster, _build_instance(raw, "provisioning", cluster)
            case _:
                return cluster, None

    async def terminate(
        self, cluster: Cluster[VultrSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[VultrSpecific]:
        if not instance_ids:
            return cluster

        api_key = get_api_key(self._config.api_key)
        is_bm = cluster.specific.is_bare_metal

        async with VultrClient(api_key, timeout=self._config.request_timeout) as client:
            async def _kill(iid: str) -> None:
                try:
                    if is_bm:
                        await client.delete_bare_metal(iid)
                    else:
                        await client.delete_instance(iid)
                except Exception as e:
                    log.error("Failed to terminate {iid}: {err}", iid=iid, err=e)

            await asyncio.gather(*(_kill(iid) for iid in instance_ids))

        return cluster

    async def teardown(self, cluster: Cluster[VultrSpecific]) -> Cluster[VultrSpecific]:
        return cluster


# =============================================================================
# Helpers
# =============================================================================


def _region_from_offer(offer: Offer) -> str:
    """Extract region from offer ID (``vultr-{region}-{plan_id}``)."""
    return offer.id.removeprefix("vultr-").split("-", 1)[0]


def _has_ip(ip: str) -> bool:
    return bool(ip) and ip != "0.0.0.0"


def _build_instance(
    raw: InstanceResponse | BareMetalResponse,
    status: InstanceStatus,
    cluster: Cluster[VultrSpecific],
) -> Instance:
    ip = raw.get("main_ip", "")
    return Instance(
        id=raw["id"],
        status=status,
        offer=cluster.offer,
        ip=ip if _has_ip(ip) else None,
        region=raw.get("region", cluster.specific.region),
    )


async def _list_keys_adapted(client: VultrClient) -> list[dict[str, str]]:
    """Adapt Vultr SSH key list for ensure_ssh_key_on_provider."""
    keys = await client.list_ssh_keys()
    return [{"id": k["id"], "name": k["name"]} for k in keys]


async def _create_key_adapted(
    client: VultrClient, name: str, public_key: str,
) -> dict[str, str]:
    """Adapt Vultr SSH key creation for ensure_ssh_key_on_provider."""
    result = await client.create_ssh_key(name, public_key)
    return {"id": result["id"], "name": result["name"]}
