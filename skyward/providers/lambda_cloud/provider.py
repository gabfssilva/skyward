"""Lambda Cloud provider — bare-metal GPU instances."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import (
    generate_key_name,
    get_local_ssh_key,
    get_ssh_key_path,
)

from .client import LambdaClient, LambdaError, get_api_key
from .config import Lambda
from .types import (
    InstanceResponse,
    InstanceSpecsResponse,
    InstanceTypeInfo,
    parse_gpu_from_type_name,
)

log = logger.bind(provider="lambda")


@dataclass(frozen=True, slots=True)
class LambdaSpecific:
    """Lambda-specific cluster data flowing through Cluster[LambdaSpecific]."""

    ssh_key_name: str
    instance_type_name: str
    region: str
    price_cents_per_hour: int


class LambdaProvider(Provider[Lambda, LambdaSpecific]):
    """Stateless Lambda Cloud provider. Holds only immutable config."""

    def __init__(self, config: Lambda) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: Lambda) -> LambdaProvider:
        return cls(config)

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        api_key = get_api_key(self._config)

        async with LambdaClient(api_key, self._config) as client:
            instance_types = await client.list_instance_types()

        all_offers: list[Offer] = []

        for type_name, entry in instance_types.items():
            info = entry["instance_type"]
            gpu_name, gpu_count = parse_gpu_from_type_name(type_name)
            specs = info["specs"]
            price_cents = info["price_cents_per_hour"]
            price = price_cents / 100

            # Filter by accelerator name
            if spec.accelerator_name:
                req = spec.accelerator_name.upper().replace("-", " ").replace("_", " ")
                if req not in gpu_name:
                    continue

            # Filter by accelerator count
            if spec.accelerator_count and gpu_count < spec.accelerator_count:
                continue

            # Filter by vcpus
            if spec.vcpus and specs["vcpus"] < spec.vcpus:
                continue

            # Filter by memory
            if spec.memory_gb and specs["memory_gib"] < spec.memory_gb:
                continue

            # Filter by max hourly cost
            if spec.max_hourly_cost and price > spec.max_hourly_cost / spec.nodes:
                continue

            # Filter regions with capacity
            regions = entry["regions_with_capacity_available"]
            if not regions:
                continue

            for region_info in regions:
                region_name = region_info["name"]
                if self._config.region and self._config.region != region_name:
                    continue

                all_offers.append(
                    _to_offer(type_name, info, gpu_name, gpu_count, specs, price, region_name),
                )

        # Sort by price (cheapest first)
        all_offers.sort(key=lambda o: o.on_demand_price or float("inf"))

        for offer in all_offers:
            yield offer

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[LambdaSpecific]:
        api_key = get_api_key(self._config)
        ssh_key_path = get_ssh_key_path()
        pub_path, public_key = get_local_ssh_key()
        key_name = generate_key_name(pub_path)

        async with LambdaClient(api_key, self._config) as client:
            ssh_key_name = await _ensure_ssh_key(client, key_name, public_key)

        type_name: str = offer.specific
        _, region = offer.id.rsplit(":", 1)

        # Look up price from offer
        price_cents = int((offer.on_demand_price or 0) * 100)

        return Cluster(
            id=f"lambda-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="ubuntu",
            use_sudo=True,
            shutdown_command="sudo shutdown -h now",
            specific=LambdaSpecific(
                ssh_key_name=ssh_key_name,
                instance_type_name=type_name,
                region=region,
                price_cents_per_hour=price_cents,
            ),
        )

    async def provision(
        self,
        cluster: Cluster[LambdaSpecific],
        count: int,
    ) -> tuple[Cluster[LambdaSpecific], Sequence[Instance]]:
        api_key = get_api_key(self._config)
        specific = cluster.specific
        label = f"skyward-{cluster.id}"

        # Generate cloud-init user_data from bootstrap
        user_data = _build_user_data(cluster)

        async with LambdaClient(api_key, self._config) as client:
            try:
                instance_ids = await client.launch_instances(
                    region_name=specific.region,
                    instance_type_name=specific.instance_type_name,
                    ssh_key_names=[specific.ssh_key_name],
                    quantity=count,
                    name=label,
                    user_data=user_data,
                )
            except LambdaError as e:
                log.error("Launch failed: {err}", err=e)
                return cluster, []

        instances = [
            Instance(
                id=iid,
                status="provisioning",
                offer=cluster.offer,
                spot=False,
                region=specific.region,
            )
            for iid in instance_ids
        ]

        log.info(
            "Launched {n} instances in {region}",
            n=len(instances), region=specific.region,
        )
        return cluster, instances

    async def get_instance(
        self,
        cluster: Cluster[LambdaSpecific],
        instance_id: str,
    ) -> tuple[Cluster[LambdaSpecific], Instance | None]:
        api_key = get_api_key(self._config)

        async with LambdaClient(api_key, self._config) as client:
            info = await client.get_instance(instance_id)

        if not info:
            return cluster, None

        match info.get("status"):
            case "terminated" | "terminating":
                return cluster, None
            case "active" if info.get("ip"):
                return cluster, _build_instance(info, "provisioned", cluster.offer)
            case "unhealthy":
                return cluster, _build_instance(info, "provisioning", cluster.offer)
            case _:
                return cluster, _build_instance(info, "provisioning", cluster.offer)

    async def terminate(
        self,
        cluster: Cluster[LambdaSpecific],
        instance_ids: tuple[str, ...],
    ) -> Cluster[LambdaSpecific]:
        if not instance_ids:
            return cluster

        api_key = get_api_key(self._config)
        async with LambdaClient(api_key, self._config) as client:
            try:
                await client.terminate_instances(list(instance_ids))
            except Exception as e:
                log.error("Failed to terminate instances: {err}", err=e)

        return cluster

    async def teardown(
        self,
        cluster: Cluster[LambdaSpecific],
    ) -> Cluster[LambdaSpecific]:
        # Lambda has no cluster-level resources to clean up
        return cluster


# =============================================================================
# Helpers
# =============================================================================


def _to_offer(
    type_name: str,
    info: InstanceTypeInfo,
    gpu_name: str,
    gpu_count: int,
    specs: InstanceSpecsResponse,
    price: float,
    region: str,
) -> Offer:
    accel = Accelerator(name=gpu_name, count=gpu_count)
    it = InstanceType(
        name=type_name,
        accelerator=accel,
        vcpus=float(specs["vcpus"]),
        memory_gb=float(specs["memory_gib"]),
        architecture="x86_64",
        specific=None,
    )
    return Offer(
        id=f"{type_name}:{region}",
        instance_type=it,
        spot_price=None,
        on_demand_price=price,
        billing_unit="hour",
        specific=type_name,
    )


def _build_instance(
    info: InstanceResponse,
    status: InstanceStatus,
    offer: Offer,
) -> Instance:
    return Instance(
        id=info["id"],
        status=status,
        offer=offer,
        ip=info.get("ip"),
        private_ip=info.get("private_ip"),
        ssh_port=22,
        spot=False,
        region=info.get("region", {}).get("name", ""),
    )


async def _ensure_ssh_key(
    client: LambdaClient,
    key_name: str,
    public_key: str,
) -> str:
    """Ensure SSH key is registered on Lambda. Returns key name."""
    existing = await client.list_ssh_keys()

    # Match by key data (most reliable)
    local_parts = public_key.strip().split()
    local_data = local_parts[1] if len(local_parts) >= 2 else public_key

    for key in existing:
        stored_parts = key["public_key"].strip().split()
        stored_data = stored_parts[1] if len(stored_parts) >= 2 else key["public_key"]
        if stored_data == local_data:
            log.debug("Found existing SSH key: {name}", name=key["name"])
            return key["name"]

    # Create new key
    log.info("Registering SSH key '{name}'", name=key_name)
    try:
        result = await client.add_ssh_key(key_name, public_key)
        return result["name"]
    except LambdaError as e:
        if "already" in str(e).lower() or "duplicate" in str(e).lower():
            # Race condition — refetch
            for key in await client.list_ssh_keys():
                stored_parts = key["public_key"].strip().split()
                stored_data = stored_parts[1] if len(stored_parts) >= 2 else key["public_key"]
                if stored_data == local_data:
                    return key["name"]
        raise


def _build_user_data(cluster: Cluster[LambdaSpecific]) -> str | None:
    """Build cloud-init user_data for instance bootstrap.

    Lambda supports cloud-init via user_data (max 1MB).
    Returns None if no bootstrap is needed.
    """
    ttl = cluster.spec.ttl
    if not ttl:
        return None

    # Minimal cloud-init: just the safety timeout
    lines = [
        "#!/bin/bash",
        "set -e",
        f"(sleep {ttl} && sudo shutdown -h now) &",
    ]
    return "\n".join(lines) + "\n"
