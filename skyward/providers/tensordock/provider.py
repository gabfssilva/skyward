"""TensorDock provider implementation (v2 API).

Provisions GPU VMs on TensorDock's marketplace via their v2 REST API.
Uses location-based auto-placement for simplified provisioning.
"""

from __future__ import annotations

import asyncio
import os
import random
import string
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import get_local_ssh_key, get_ssh_key_path

from .client import TensorDockClient, TensorDockError
from .config import TensorDock
from .types import (
    Location,
    get_gpu_memory_gb,
    get_ssh_port_v2,
    gpu_matches_v2,
    normalize_gpu_name,
    resolve_v2_image,
)

log = logger.bind(provider="tensordock")


def _get_token(config: TensorDock) -> str:
    """Resolve API token from config or environment."""
    token = config.api_token or os.environ.get("TENSORDOCK_API_TOKEN")
    if not token:
        raise RuntimeError(
            "TensorDock API token required. Set TENSORDOCK_API_TOKEN "
            "environment variable or pass it to TensorDock(api_token=...)."
        )
    return token


@dataclass(frozen=True, slots=True)
class TensorDockOfferData:
    """Carried in Offer.specific — location + GPU details for provisioning."""

    location_id: str
    gpu_model: str
    gpu_count: int
    vcpus: int
    ram_gb: int
    hourly_rate: float


@dataclass(frozen=True, slots=True)
class TensorDockSpecific:
    """TensorDock-specific cluster data flowing through Cluster[TensorDockSpecific]."""

    ssh_public_key: str
    location_id: str


def _filter_locations(
    locations: list[Location],
    spec: PoolSpec,
    config: TensorDock,
) -> list[tuple[str, str, int, int, int, float]]:
    """Filter and rank location GPUs by match, availability, and price.

    Returns
    -------
    list[tuple[str, str, int, int, int, float]]
        (location_id, gpu_v0name, gpu_count, vcpus, ram_gb, hourly_rate)
        sorted by price ascending.
    """
    gpu_count = spec.accelerator_count or 1
    candidates: list[tuple[str, str, int, int, int, float]] = []

    for loc in locations:
        if config.location:
            country = loc.get("country", "").lower()
            if country != config.location.lower():
                continue

        for gpu in loc.get("gpus", []):
            v0_name = gpu.get("v0Name", "")
            available = gpu.get("max_count", 0)
            if available < gpu_count:
                continue

            if spec.accelerator_name and not gpu_matches_v2(gpu, spec.accelerator_name):
                continue

            gpu_price = gpu.get("price_per_hr", 0.0)
            pricing = gpu.get("pricing", {})
            resources = gpu.get("resources", {})

            min_ram = config.min_ram_gb or 16
            min_vcpus = config.min_vcpus or 4
            ram_gb = int(max(min_ram, spec.memory_gb or 0))
            vcpus = max(min_vcpus, int(spec.vcpus or 0))

            max_vcpus = resources.get("max_vcpus", 0)
            max_ram = resources.get("max_ram_gb", 0)
            if max_vcpus < vcpus or max_ram < ram_gb:
                continue

            ram_price = pricing.get("per_gb_ram_hr", 0.0)
            cpu_price = pricing.get("per_vcpu_hr", 0.0)
            storage_price = pricing.get("per_gb_storage_hr", 0.0)

            hourly_rate = (
                gpu_price * gpu_count
                + ram_price * ram_gb
                + cpu_price * vcpus
                + storage_price * config.storage_gb
            )

            candidates.append((loc["id"], v0_name, gpu_count, vcpus, ram_gb, hourly_rate))

    if spec.max_hourly_cost:
        max_per_instance = spec.max_hourly_cost / spec.nodes
        candidates = [c for c in candidates if c[5] <= max_per_instance]

    candidates.sort(key=lambda c: c[5])
    return candidates


class TensorDockProvider(Provider[TensorDock, TensorDockSpecific]):
    """TensorDock provider using v2 API."""

    def __init__(self, config: TensorDock) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: TensorDock) -> TensorDockProvider:
        return cls(config)

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        token = _get_token(self._config)

        async with TensorDockClient(token, timeout=self._config.request_timeout) as client:
            locations = await client.list_locations()

        candidates = _filter_locations(locations, spec, self._config)

        if not candidates:
            log.debug("No matching locations found")
            return

        for loc_id, gpu_model, gpu_count, vcpus, ram_gb, hourly_rate in candidates:
            display_name = normalize_gpu_name(gpu_model)
            memory_gb = get_gpu_memory_gb(gpu_model)
            accel = Accelerator(
                name=display_name,
                memory=f"{memory_gb}GB" if memory_gb else "",
                count=gpu_count,
            )

            it = InstanceType(
                name=display_name,
                accelerator=accel,
                vcpus=float(vcpus),
                memory_gb=float(ram_gb),
                architecture="x86_64",
                specific=None,
            )

            yield Offer(
                id=f"td-{loc_id[:8]}-{gpu_model}",
                instance_type=it,
                spot_price=None,
                on_demand_price=hourly_rate,
                billing_unit="second",
                specific=TensorDockOfferData(
                    location_id=loc_id,
                    gpu_model=gpu_model,
                    gpu_count=gpu_count,
                    vcpus=vcpus,
                    ram_gb=ram_gb,
                    hourly_rate=hourly_rate,
                ),
            )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[TensorDockSpecific]:
        token = _get_token(self._config)
        ssh_key_path = get_ssh_key_path()
        _, ssh_public_key = get_local_ssh_key()

        async with TensorDockClient(token, timeout=self._config.request_timeout) as client:
            if not await client.test_auth():
                raise RuntimeError("TensorDock authentication failed — check credentials")

        offer_data: TensorDockOfferData = offer.specific

        return Cluster(
            id=f"tensordock-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="user",
            use_sudo=True,
            shutdown_command="sudo shutdown -h now",
            specific=TensorDockSpecific(
                ssh_public_key=ssh_public_key,
                location_id=offer_data.location_id,
            ),
        )

    async def provision(
        self, cluster: Cluster[TensorDockSpecific], count: int,
    ) -> tuple[Cluster[TensorDockSpecific], Sequence[Instance]]:
        specific = cluster.specific
        offer_data: TensorDockOfferData = cluster.offer.specific
        token = _get_token(self._config)
        image = resolve_v2_image(self._config.operating_system)

        instances: list[Instance] = []

        async with TensorDockClient(token, timeout=self._config.request_timeout) as client:
            for i in range(count):
                suffix = "".join(random.choices(string.ascii_lowercase, k=6))
                vm_name = f"skyward-{cluster.id}-{i}-{suffix}"

                try:
                    result = await client.create_instance(
                        name=vm_name,
                        image=image,
                        gpu_model=offer_data.gpu_model,
                        gpu_count=offer_data.gpu_count,
                        vcpus=offer_data.vcpus,
                        ram_gb=offer_data.ram_gb,
                        storage_gb=self._config.storage_gb,
                        location_id=specific.location_id,
                        ssh_key=specific.ssh_public_key,
                    )
                except TensorDockError as e:
                    log.error(
                        "Failed to create instance {i}/{count}: {err}",
                        i=i + 1, count=count, err=e,
                    )
                    continue

                instance_id = result.get("id", "")
                if not instance_id:
                    log.warning("Create returned no instance ID: {r}", r=result)
                    continue

                log.info(
                    "Created instance {name}: id={id}",
                    name=vm_name, id=instance_id,
                )

                instances.append(Instance(
                    id=instance_id,
                    status="provisioning",
                    offer=cluster.offer,
                    ip="",
                    private_ip=None,
                    ssh_port=22,
                    spot=False,
                ))

        return cluster, instances

    async def get_instance(
        self, cluster: Cluster[TensorDockSpecific], instance_id: str,
    ) -> tuple[Cluster[TensorDockSpecific], Instance | None]:
        token = _get_token(self._config)

        async with TensorDockClient(token, timeout=self._config.request_timeout) as client:
            vm = await client.get_instance(instance_id)

        if not vm:
            return cluster, None

        status_str = (vm.get("status") or "").lower()
        ip = vm.get("ipAddress") or ""
        port_forwards = vm.get("portForwards") or []

        log.debug(
            "Instance {id} status={status}, ip={ip}",
            id=instance_id, status=status_str, ip=ip,
        )

        ssh_port = get_ssh_port_v2(port_forwards)

        match status_str:
            case "stopped" | "error" | "terminated" | "deleted":
                return cluster, None
            case "running" if ip:
                return cluster, Instance(
                    id=instance_id,
                    status="provisioned",
                    offer=cluster.offer,
                    ip=ip,
                    private_ip=ip or None,
                    ssh_port=ssh_port,
                    spot=False,
                )
            case _:
                return cluster, Instance(
                    id=instance_id,
                    status="provisioning",
                    offer=cluster.offer,
                    ip=ip,
                    private_ip=ip or None,
                    ssh_port=ssh_port,
                    spot=False,
                )

    async def terminate(
        self, cluster: Cluster[TensorDockSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[TensorDockSpecific]:
        if not instance_ids:
            return cluster

        token = _get_token(self._config)
        async with TensorDockClient(token, timeout=self._config.request_timeout) as client:
            async def _delete(iid: str) -> None:
                try:
                    await client.delete_instance(iid)
                except Exception as e:
                    log.error("Failed to delete instance {id}: {err}", id=iid, err=e)

            await asyncio.gather(*(_delete(iid) for iid in instance_ids))
        return cluster

    async def teardown(
        self, cluster: Cluster[TensorDockSpecific],
    ) -> Cluster[TensorDockSpecific]:
        return cluster
