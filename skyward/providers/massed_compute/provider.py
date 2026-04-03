"""Massed Compute provider implementation.

Stateless provider using HTTP API with SSH key registration.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.core import PoolSpec
from skyward.core.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import (
    compute_fingerprint,
    get_local_ssh_key,
    get_ssh_key_path,
)

from .client import MassedComputeClient, MassedComputeError, get_api_key
from .config import MassedCompute
from .types import InstanceResponse, InventoryItem, is_spot_product, parse_product_name

log = logger.bind(provider="massed_compute")


def _sanitize_key_name(name: str) -> str:
    """Strip non-alphanumeric chars — Massed Compute SSH key names are alphanumeric only."""
    return re.sub(r"[^a-zA-Z0-9]", "", name)


def _resolve_accelerator(name: str) -> Accelerator:
    """Resolve accelerator from catalog, falling back to raw name."""
    try:
        return Accelerator.from_name(name, count=1)
    except ValueError:
        return Accelerator(name=name, count=1)


def _to_offer(product_name: str, item: InventoryItem) -> Offer:
    """Convert an inventory item to a Skyward Offer."""
    it = item["instance_type"]
    specs = it["specs"]
    gpu_count, catalog_name = parse_product_name(product_name)
    spot = is_spot_product(product_name)
    price_usd = it["price_cents_per_hour"] / 100.0

    accel: Accelerator | None = None
    if not product_name.startswith("cpu_"):
        resolved = _resolve_accelerator(catalog_name)
        accel = Accelerator(
            name=resolved.name,
            count=gpu_count,
            memory=resolved.memory,
            metadata=resolved.metadata,
        )

    regions = item.get("regions_with_capacity_available", [])
    region_str = regions[0]["name"] if regions else "any"

    instance_type = InstanceType(
        name=product_name,
        accelerator=accel,
        vcpus=float(specs["vcpu_count"]),
        memory_gb=float(specs["memory_gib"]),
        architecture="x86_64",
        specific=None,
    )

    return Offer(
        id=f"massed-{product_name}-{region_str}",
        instance_type=instance_type,
        spot_price=price_usd if spot else None,
        on_demand_price=price_usd if not spot else None,
        billing_unit="minute",
        specific={
            "product_name": product_name,
            "capacity": item["capacity_available"],
            "regions": [r["name"] for r in regions],
            "spot": spot,
        },
    )


def _build_instance(
    info: InstanceResponse,
    status: InstanceStatus,
    offer: Offer,
    *,
    spot: bool,
) -> Instance:
    ip = info.get("ip") or None
    return Instance(
        id=info["uuid"],
        status=status,
        offer=offer,
        ip=ip if ip else None,
        private_ip=ip if ip else None,
        ssh_port=22,
        ssh_password=info.get("password"),
        spot=spot,
        region=info.get("region", {}).get("name", ""),
    )


@dataclass(frozen=True, slots=True)
class MassedComputeSpecific:
    """Massed Compute-specific cluster data flowing through Cluster[MassedComputeSpecific]."""

    ssh_key_id: str
    ssh_key_name: str
    product_name: str


class MassedComputeProvider(Provider[MassedCompute, MassedComputeSpecific]):
    """Stateless Massed Compute provider. Holds only immutable config."""

    name = "massed_compute"

    def __init__(self, config: MassedCompute) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: MassedCompute) -> MassedComputeProvider:
        return cls(config)

    async def offers(self) -> AsyncIterator[Offer]:
        api_key = get_api_key(self._config.api_key)

        async with MassedComputeClient(api_key, request_timeout=self._config.request_timeout) as client:
            inventory = await client.gpu_inventory()

        for product_name, item in inventory.items():
            if item["capacity_available"] <= 0:
                continue
            yield _to_offer(product_name, item)

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[MassedComputeSpecific]:
        api_key = get_api_key(self._config.api_key)
        ssh_key_path = get_ssh_key_path()
        _, public_key = get_local_ssh_key()
        fingerprint = compute_fingerprint(public_key)

        async with MassedComputeClient(api_key, request_timeout=self._config.request_timeout) as client:
            existing = await client.list_ssh_keys()

            key_name = _sanitize_key_name(f"skyward{fingerprint.replace(':', '')[:12]}")
            ssh_key_id = ""

            for key in existing:
                if key.get("name") == key_name:
                    ssh_key_id = key["id"]
                    break

            if not ssh_key_id:
                new_key = await client.create_ssh_key(key_name, public_key)
                ssh_key_id = new_key["id"]
                log.info(
                    "Registered SSH key {name} ({kid})",
                    name=key_name, kid=ssh_key_id,
                )

        product_name = offer.specific["product_name"]

        return Cluster(
            id=f"massed-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="Ubuntu",
            use_sudo=True,
            shutdown_command="sudo shutdown -h now",
            ssh_pty=True,
            specific=MassedComputeSpecific(
                ssh_key_id=ssh_key_id,
                ssh_key_name=key_name,
                product_name=product_name,
            ),
        )

    async def provision(
        self, cluster: Cluster[MassedComputeSpecific], count: int,
    ) -> tuple[Cluster[MassedComputeSpecific], Sequence[Instance]]:
        api_key = get_api_key(self._config.api_key)
        specific = cluster.specific
        spot = cluster.offer.specific.get("spot", False)

        instances: list[Instance] = []
        async with MassedComputeClient(api_key, request_timeout=self._config.request_timeout) as client:
            for i in range(count):
                label = f"skyward-{cluster.id}-{len(cluster.instances) + i}"

                try:
                    instance_uuid = await client.launch_instance(
                        product_name=specific.product_name,
                        image_id=self._config.image_id,
                        instance_name=label,
                        ssh_key_names=(specific.ssh_key_name,),
                    )
                except MassedComputeError as e:
                    log.error("Failed to launch instance: {err}", err=e)
                    continue

                instances.append(Instance(
                    id=instance_uuid,
                    status="provisioning",
                    offer=cluster.offer,
                    spot=spot,
                ))

        return cluster, instances

    async def get_instance(
        self, cluster: Cluster[MassedComputeSpecific], instance_id: str,
    ) -> tuple[Cluster[MassedComputeSpecific], Instance | None]:
        api_key = get_api_key(self._config.api_key)
        async with MassedComputeClient(api_key, request_timeout=self._config.request_timeout) as client:
            info = await client.get_instance(instance_id)

        if not info:
            return cluster, None

        spot = cluster.offer.specific.get("spot", False)

        match info.get("status"):
            case "terminated" | "error" | "Stopped":
                return cluster, None
            case "Running" if info.get("os_booted") == 1 and info.get("ip"):
                return cluster, _build_instance(
                    info, "provisioned", cluster.offer, spot=spot,
                )
            case _:
                return cluster, _build_instance(
                    info, "provisioning", cluster.offer, spot=spot,
                )

    async def terminate(
        self, cluster: Cluster[MassedComputeSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[MassedComputeSpecific]:
        if not instance_ids:
            return cluster

        api_key = get_api_key(self._config.api_key)
        async with MassedComputeClient(api_key, request_timeout=self._config.request_timeout) as client:
            try:
                await client.terminate_instances(instance_ids)
            except Exception as e:
                log.error("Failed to terminate instances: {err}", err=e)

        return cluster

    async def teardown(self, cluster: Cluster[MassedComputeSpecific]) -> Cluster[MassedComputeSpecific]:
        api_key = get_api_key(self._config.api_key)
        async with MassedComputeClient(api_key, request_timeout=self._config.request_timeout) as client:
            try:
                await client.delete_ssh_key(cluster.specific.ssh_key_id)
                log.info(
                    "Deleted SSH key {name}",
                    name=cluster.specific.ssh_key_name,
                )
            except Exception as e:
                log.warning("Failed to delete SSH key: {err}", err=e)

        return cluster
