"""Lambda Cloud provider implementation."""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from skyward.accelerators import Accelerator
from skyward.core.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.core.spec import PoolSpec
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import (
    ensure_ssh_key_on_provider,
    generate_key_name,
    get_local_ssh_key,
    get_ssh_key_path,
)

from .client import LambdaClient, LambdaError, get_api_key
from .config import LambdaCloud
from .types import InstanceResponse

log = logger.bind(provider="lambda")


@dataclass(frozen=True, slots=True)
class LambdaSpecific:
    """Lambda-specific cluster data flowing through Cluster[LambdaSpecific]."""

    ssh_key_name: str
    ssh_key_id: str


@dataclass(frozen=True, slots=True)
class LambdaOfferData:
    """Carried in Offer.specific for Lambda offers."""

    instance_type_name: str
    regions: tuple[str, ...]


_GPU_DESC_RE = re.compile(r"(?:Tesla\s+)?(\w+)\s+\((\d+)\s+GB")


def _parse_gpu_description(desc: str) -> tuple[str, int]:
    """Parse Lambda GPU description into (name, vram_gb).

    Examples: ``"GH200 (96 GB)"`` → ``("GH200", 96)``,
    ``"Tesla V100 (16 GB)"`` → ``("V100", 16)``.
    """
    if m := _GPU_DESC_RE.match(desc):
        return m.group(1), int(m.group(2))
    return desc, 0


class LambdaCloudProvider(Provider[LambdaCloud, LambdaSpecific]):
    name = "lambda"

    def __init__(self, config: LambdaCloud, client: LambdaClient) -> None:
        self._config = config
        self._client = client

    @classmethod
    async def create(cls, config: LambdaCloud) -> LambdaCloudProvider:
        api_key = get_api_key(config.api_key)
        client = LambdaClient(api_key, timeout=config.request_timeout)
        return cls(config, client)

    async def offers(self) -> AsyncIterator[Offer]:
        instance_types = await self._client.list_instance_types()

        for name, entry in instance_types.items():
            it = entry["instance_type"]
            specs = it["specs"]
            gpu_desc = it["gpu_description"]

            if gpu_desc == "N/A" or specs["gpus"] == 0:
                continue

            gpu_name, vram_gb = _parse_gpu_description(gpu_desc)
            gpu_count = specs["gpus"]
            price_per_hour = it["price_cents_per_hour"] / 100.0
            regions = tuple(r["name"] for r in entry["regions_with_capacity_available"])

            accel = Accelerator(
                name=gpu_name,
                memory=f"{vram_gb}GB" if vram_gb else "",
                count=gpu_count,
            )
            instance_type = InstanceType(
                name=name,
                accelerator=accel,
                vcpus=float(specs["vcpus"]),
                memory_gb=float(specs["memory_gib"]),
                architecture="x86_64",
                specific=None,
            )
            yield Offer(
                id=f"lambda-{name}",
                instance_type=instance_type,
                spot_price=None,
                on_demand_price=price_per_hour,
                billing_unit="minute",
                specific=LambdaOfferData(name, regions),
            )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[LambdaSpecific]:
        ssh_key_path = get_ssh_key_path()
        pub_path, _ = get_local_ssh_key()
        key_name = generate_key_name(pub_path)

        async def _list_keys() -> list[dict[str, Any]]:
            return [dict(k) for k in await self._client.list_ssh_keys()]

        async def _create_key(name: str, public_key: str) -> dict[str, Any]:
            return dict(await self._client.create_ssh_key(name, public_key))

        key_id = await ensure_ssh_key_on_provider(
            list_keys_fn=_list_keys,
            create_key_fn=_create_key,
            provider_name="Lambda",
        )

        return Cluster(
            id=f"lambda-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="ubuntu",
            use_sudo=True,
            shutdown_command="sudo shutdown -h now",
            specific=LambdaSpecific(ssh_key_name=key_name, ssh_key_id=key_id),
        )

    async def provision(
        self, cluster: Cluster[LambdaSpecific], count: int,
    ) -> tuple[Cluster[LambdaSpecific], Sequence[Instance]]:
        type_name = cluster.offer.instance_type.name
        region = await self._resolve_region(type_name)

        try:
            result = await self._client.launch_instances(
                region_name=region,
                instance_type_name=type_name,
                ssh_key_names=[cluster.specific.ssh_key_name],
                quantity=count,
                name=f"skyward-{cluster.id}",
            )
        except LambdaError as e:
            log.error("Failed to launch instances: {err}", err=e)
            raise

        instances = tuple(
            Instance(
                id=iid,
                status="provisioning",
                offer=cluster.offer,
                region=region,
            )
            for iid in result["instance_ids"]
        )

        return cluster, instances

    async def _resolve_region(self, instance_type_name: str) -> str:
        if self._config.region:
            return self._config.region

        instance_types = await self._client.list_instance_types()
        if entry := instance_types.get(instance_type_name):
            regions = entry["regions_with_capacity_available"]
            if regions:
                region = regions[0]["name"]
                log.info("Auto-selected region {region} for {type}", region=region, type=instance_type_name)
                return region

        log.warning("No regions with capacity for {type}, defaulting to us-east-3", type=instance_type_name)
        return "us-east-3"

    async def get_instance(
        self, cluster: Cluster[LambdaSpecific], instance_id: str,
    ) -> tuple[Cluster[LambdaSpecific], Instance | None]:
        info = await self._client.get_instance(instance_id)
        if not info:
            return cluster, None

        match info["status"]:
            case "terminated" | "terminating" | "error" | "unhealthy":
                return cluster, None
            case "active" if info.get("ip"):
                return cluster, _build_instance(info, "provisioned", cluster.offer)
            case _:
                return cluster, _build_instance(info, "provisioning", cluster.offer)

    async def terminate(
        self, cluster: Cluster[LambdaSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[LambdaSpecific]:
        if instance_ids:
            try:
                await self._client.terminate_instances(list(instance_ids))
            except LambdaError as e:
                log.error("Failed to terminate instances: {err}", err=e)
        return cluster

    async def teardown(self, cluster: Cluster[LambdaSpecific]) -> Cluster[LambdaSpecific]:
        try:
            await self._client.delete_ssh_key(cluster.specific.ssh_key_id)
        except LambdaError as e:
            log.debug("SSH key cleanup failed (may already be deleted): {err}", err=e)
        await self._client._http.aclose()
        return cluster


def _build_instance(
    info: InstanceResponse,
    status: InstanceStatus,
    offer: Offer,
) -> Instance:
    region = ""
    if r := info.get("region"):
        region = r.get("name", "")

    return Instance(
        id=info["id"],
        status=status,
        offer=offer,
        ip=info.get("ip") or None,
        private_ip=info.get("private_ip") or None,
        spot=False,
        region=region,
    )
