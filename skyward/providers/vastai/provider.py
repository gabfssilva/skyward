from __future__ import annotations

import asyncio
import random
import string
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceStatus
from skyward.observability.logger import logger
from skyward.providers.provider import CloudProvider
from skyward.providers.ssh_keys import get_ssh_key_path

from .client import VastAIClient, VastAIError, get_api_key, select_all_valid_clusters
from .config import VastAI
from .types import InstanceResponse, OfferResponse, get_direct_ssh_port

log = logger.bind(provider="vastai")

_DEFAULT_IMAGE = "nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04"


def _get_docker_image(config: VastAI, spec: PoolSpec) -> str:
    match getattr(spec.image, "container_image", None):
        case str() as img:
            return img
        case _:
            return config.docker_image or _DEFAULT_IMAGE


@dataclass(frozen=True, slots=True)
class VastAISpecific:
    """VastAI-specific cluster data flowing through Cluster[VastAISpecific]."""

    ssh_key_id: int
    ssh_public_key: str
    overlay_name: str | None = None
    overlay_cluster_id: int | None = None
    docker_image: str = _DEFAULT_IMAGE
    geolocation: str | None = None

    @property
    def has_overlay(self) -> bool:
        return self.overlay_name is not None


class VastAICloudProvider(CloudProvider[VastAI, VastAISpecific]):
    """Stateless VastAI provider. Holds only immutable config."""

    def __init__(self, config: VastAI) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: VastAI) -> VastAICloudProvider:
        return cls(config)

    async def prepare(self, spec: PoolSpec) -> Cluster[VastAISpecific]:
        api_key = get_api_key()
        ssh_key_path = get_ssh_key_path()

        async with VastAIClient(api_key, config=self._config) as client:
            ssh_key_id, ssh_public_key = await client.ensure_ssh_key()

            overlay_name: str | None = None
            overlay_cluster_id: int | None = None

            if spec.nodes > 1 and self._config.use_overlay:
                overlay_name, overlay_cluster_id = await _setup_overlay_network(
                    client, self._config, spec,
                )

        docker_image = _get_docker_image(self._config, spec)

        shutdown_command = (
            "eval $(cat /proc/1/environ "
            "| tr '\\0' '\\n' "
            "| grep -E 'CONTAINER_ID"
            "|CONTAINER_API_KEY' "
            "| sed 's/^/export /'); "
            "curl -s -X DELETE "
            "https://console.vast.ai"
            "/api/v0/instances/"
            "$CONTAINER_ID/ "
            "-H \"Authorization: "
            "Bearer $CONTAINER_API_KEY\""
        )

        return Cluster(
            id=f"vastai-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            ssh_key_path=ssh_key_path,
            ssh_user="root",
            use_sudo=False,
            shutdown_command=shutdown_command,
            instances=(),
            specific=VastAISpecific(
                ssh_key_id=ssh_key_id,
                ssh_public_key=ssh_public_key,
                overlay_name=overlay_name,
                overlay_cluster_id=overlay_cluster_id,
                docker_image=docker_image,
                geolocation=self._config.geolocation,
            ),
        )

    async def provision(
        self, cluster: Cluster[VastAISpecific], count: int,
    ) -> tuple[Cluster[VastAISpecific], Sequence[Instance]]:
        specific = cluster.specific
        api_key = get_api_key()

        instances: list[Instance] = []
        reserved: set[int] = set()

        async with VastAIClient(api_key, config=self._config) as client:
            for _ in range(count):
                offers = await _search_offers(
                    client, self._config, cluster.spec, specific,
                )

                if not offers:
                    log.error("No offers found")
                    continue

                result = await _try_create_from_offers(
                    client, self._config, cluster, offers, reserved,
                )

                if result is None:
                    log.error("All offers failed")
                    continue

                instances.append(Instance(
                    id=str(result.instance_id),
                    status="provisioning",
                    spot=result.spot,
                    instance_type=result.gpu_name,
                    gpu_count=result.gpu_count,
                    gpu_model=result.gpu_name,
                    gpu_vram_gb=result.gpu_vram_gb,
                    vcpus=result.vcpus,
                    memory_gb=result.memory_gb,
                    region=specific.geolocation or "Global",
                    hourly_rate=result.hourly_rate,
                    on_demand_rate=result.on_demand_rate,
                    billing_increment=1,
                ))

        return cluster, instances

    async def get_instance(
        self, cluster: Cluster[VastAISpecific], instance_id: str,
    ) -> tuple[Cluster[VastAISpecific], Instance | None]:
        api_key = get_api_key()
        async with VastAIClient(api_key, config=self._config) as client:
            info = await client.get_instance(int(instance_id))

        if not info:
            return cluster, None

        match info.get("actual_status"):
            case "exited" | "error" | "destroyed":
                return cluster, None
            case "running" if info.get("ssh_host") or info.get("public_ipaddr"):
                return cluster, _build_vastai_instance(info, "provisioned", cluster.specific)
            case _:
                return cluster, _build_vastai_instance(info, "provisioning", cluster.specific)

    async def terminate(
        self, cluster: Cluster[VastAISpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[VastAISpecific]:
        if not instance_ids:
            return cluster

        api_key = get_api_key()
        async with VastAIClient(api_key, config=self._config) as client:
            async def _destroy(iid: str) -> None:
                try:
                    await client.destroy_instance(int(iid))
                except Exception as e:
                    log.error("Failed to destroy {iid}: {err}", iid=iid, err=e)

            await asyncio.gather(*(_destroy(iid) for iid in instance_ids))
        return cluster

    async def teardown(self, cluster: Cluster[VastAISpecific]) -> Cluster[VastAISpecific]:
        specific = cluster.specific
        if not specific.has_overlay:
            return cluster

        api_key = get_api_key()
        async with VastAIClient(api_key, config=self._config) as client:
            try:
                await client.delete_overlay(specific.overlay_name)  # type: ignore[arg-type]
            except Exception as e:
                log.error(
                    "Failed to delete overlay '{name}': {err}",
                    name=specific.overlay_name, err=e,
                )
        return cluster


def _self_destruction_script(ttl: int, shutdown_command: str) -> str:
    from skyward.providers.bootstrap.compose import resolve
    from skyward.providers.bootstrap.ops import instance_timeout

    lines = ["#!/bin/bash", "set -e", "tail -f /dev/null &"]
    if ttl:
        lines.append(resolve(instance_timeout(ttl, shutdown_command=shutdown_command)))
    return "\n".join(lines) + "\n"


def _build_vastai_instance(
    info: InstanceResponse, status: InstanceStatus, specific: VastAISpecific,
) -> Instance:
    str_id = str(info["id"])
    direct_port = get_direct_ssh_port(info)

    match (info.get("public_ipaddr"), direct_port):
        case (str() as pub_ip, int() as port) if pub_ip:
            ssh_host = pub_ip
            ssh_port = port
        case _:
            ssh_host = info.get("ssh_host", "")
            ssh_port = info.get("ssh_port", 22)

    hourly_rate = info.get("dph_total", 0.0)
    gpu_name = info.get("gpu_name", "")
    gpu_count = info.get("num_gpus", 0)
    total_vram_mb: float = info.get("gpu_ram", 0)  # type: ignore[assignment]

    return Instance(
        id=str_id,
        status=status,
        ip=ssh_host,
        private_ip=info.get("public_ipaddr") or ssh_host or None,
        ssh_port=ssh_port,
        spot=info.get("is_bid", False),
        instance_type=gpu_name,
        gpu_count=gpu_count,
        gpu_model=gpu_name,
        vcpus=int(info.get("cpu_cores_effective", 0)) if "cpu_cores_effective" in info else 0,  # type: ignore[operator]
        memory_gb=info.get("cpu_ram", 0) / 1024,  # type: ignore[operator]
        gpu_vram_gb=int(total_vram_mb / 1024 / gpu_count) if gpu_count else 0,
        region=specific.geolocation or "Global",
        hourly_rate=hourly_rate,
        on_demand_rate=hourly_rate,
        billing_increment=1,
    )


async def _search_offers(
    client: VastAIClient,
    config: VastAI,
    spec: PoolSpec,
    specific: VastAISpecific,
) -> list[OfferResponse]:
    use_interruptible = spec.allocation in ("spot", "spot-if-available")
    gpu_name = (
        spec.accelerator_name.replace(" ", "_").replace("-", "_")
        if spec.accelerator_name
        else None
    )

    log.debug(
        "Searching offers: gpu={gpu}, geo={geo}, interruptible={spot}",
        gpu=gpu_name, geo=config.geolocation, spot=use_interruptible,
    )
    offers = await client.search_offers(
        gpu_name=gpu_name,
        min_reliability=config.min_reliability,
        geolocation=config.geolocation,
        use_interruptible=use_interruptible,
        with_cluster_id=spec.nodes > 1,
    )
    log.debug("Found {n} offers", n=len(offers))

    if gpu_name:
        req_norm = gpu_name.upper()
        offers = [
            o for o in offers
            if req_norm in o["gpu_name"].replace(" ", "_").upper()
        ]

    if specific.overlay_cluster_id is not None:
        offers = [o for o in offers if o.get("cluster_id") == specific.overlay_cluster_id]

    price_key = "min_bid" if use_interruptible else "dph_total"
    offers.sort(key=lambda o: o.get(price_key, float("inf")))

    if spec.max_hourly_cost:
        max_per_instance = spec.max_hourly_cost / spec.nodes

        def offer_price(o: OfferResponse) -> float:
            if use_interruptible:
                return o.get("min_bid", float("inf")) * config.bid_multiplier
            return o.get("dph_total", float("inf"))

        offers = [o for o in offers if offer_price(o) <= max_per_instance]

    return offers


@dataclass(frozen=True, slots=True)
class _ProvisionResult:
    instance_id: int
    hourly_rate: float
    on_demand_rate: float
    gpu_name: str
    gpu_count: int
    gpu_vram_gb: int
    vcpus: int
    memory_gb: float
    spot: bool


async def _try_create_from_offers(
    client: VastAIClient,
    config: VastAI,
    cluster: Cluster[VastAISpecific],
    offers: list[OfferResponse],
    reserved: set[int],
) -> _ProvisionResult | None:
    use_interruptible = cluster.spec.allocation in ("spot", "spot-if-available")
    docker_image = cluster.specific.docker_image
    label = f"skyward-{cluster.id}-{len(cluster.instances)}"
    ttl = cluster.spec.ttl or config.instance_timeout
    minimal_onstart = _self_destruction_script(ttl, cluster.shutdown_command)

    for idx, offer in enumerate(offers):
        offer_id = offer["id"]
        if offer_id in reserved:
            continue

        reserved.add(offer_id)

        price = (
            offer["min_bid"] * config.bid_multiplier
            if use_interruptible
            else None
        )

        log.debug(
            "Trying offer {i}/{total}: id={oid}, gpu={gpu}",
            i=idx + 1, total=len(offers),
            oid=offer_id, gpu=offer.get("gpu_name"),
        )
        try:
            instance_id = await client.create_instance(
                offer_id=offer_id,
                image=docker_image,
                disk=config.disk_gb,
                label=label,
                onstart_cmd=minimal_onstart,
                overlay_name=cluster.specific.overlay_name,
                price=price,
            )

            if cluster.specific.has_overlay:
                try:
                    await client.join_overlay(cluster.specific.overlay_name, instance_id)  # type: ignore[arg-type]
                    log.info("Joined overlay '{name}'", name=cluster.specific.overlay_name)
                except VastAIError as e:
                    log.error("Failed to join overlay: {err}", err=e)
                    await client.destroy_instance(instance_id)
                    reserved.discard(offer_id)
                    continue

            on_demand_rate = offer.get("dph_total", 0.0)
            hourly_rate = price if price else on_demand_rate
            gpu_count = offer.get("num_gpus", 0)
            total_vram_mb = offer.get("gpu_ram", 0)

            return _ProvisionResult(
                instance_id=instance_id,
                hourly_rate=hourly_rate,
                on_demand_rate=on_demand_rate,
                gpu_name=offer.get("gpu_name", ""),
                gpu_count=gpu_count,
                gpu_vram_gb=int(total_vram_mb / 1024 / gpu_count) if gpu_count else 0,
                vcpus=int(offer.get("cpu_cores", 0)),
                memory_gb=offer.get("cpu_ram", 0) / 1024,
                spot=use_interruptible,
            )
        except VastAIError as e:
            reserved.discard(offer_id)
            log.warning(
                "Offer {i}/{total} failed: {err}",
                i=idx + 1, total=len(offers), err=e,
            )

    return None


async def _setup_overlay_network(
    client: VastAIClient,
    config: VastAI,
    spec: PoolSpec,
) -> tuple[str | None, int | None]:
    use_interruptible = spec.allocation in ("spot", "spot-if-available")
    gpu_name = (
        spec.accelerator_name.replace(" ", "_").replace("-", "_")
        if spec.accelerator_name
        else None
    )

    offers = await client.search_offers(
        gpu_name=gpu_name,
        min_reliability=config.min_reliability,
        geolocation=config.geolocation,
        use_interruptible=use_interruptible,
        with_cluster_id=True,
    )

    valid_clusters = select_all_valid_clusters(offers, spec.nodes, use_interruptible)
    if not valid_clusters:
        log.warning("No clusters found with {n} nodes", n=spec.nodes)
        return None, None

    for idx, (physical_cluster_id, _) in enumerate(valid_clusters):
        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
        overlay_name = f"skyward-{suffix}"

        log.info(
            "Trying cluster {cid} ({i}/{total})",
            cid=physical_cluster_id, i=idx + 1, total=len(valid_clusters),
        )

        try:
            await client.create_overlay(physical_cluster_id, overlay_name)
            log.info("Overlay '{name}' created", name=overlay_name)
            return overlay_name, physical_cluster_id
        except VastAIError as e:
            log.warning("Overlay failed on cluster {cid}: {err}", cid=physical_cluster_id, err=e)

    log.warning("Failed to create overlay on any cluster")
    return None, None
