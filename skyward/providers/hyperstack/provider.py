"""Hyperstack provider implementation.

Mutable state: object storage access keys (created in prepare, deleted in teardown).
All other state flows through Cluster[HyperstackSpecific].
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from skyward.accelerators import Accelerator
from skyward.core import PoolSpec
from skyward.core.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Mountable, Provider

if TYPE_CHECKING:
    from skyward.storage import Storage
from skyward.providers.ssh_keys import generate_key_name, get_local_ssh_key, get_ssh_key_path

from .client import HYPERSTACK_API_BASE, HyperstackClient, get_api_key
from .config import Hyperstack
from .types import FlavorResponse, PricebookEntry, VMResponse, normalize_gpu_name

log = logger.bind(provider="hyperstack")




@dataclass(frozen=True, slots=True)
class HyperstackSpecific:
    """Hyperstack-specific cluster data flowing through Cluster[HyperstackSpecific]."""

    environment_name: str
    environment_id: int
    key_name: str
    flavor_name: str
    image_name: str
    region: str
    firewall_applied: frozenset[int] = frozenset()
    network_optimised: bool = False


class HyperstackProvider(Provider[Hyperstack, HyperstackSpecific], Mountable[HyperstackSpecific]):
    """Hyperstack provider with S3-compatible object storage support."""

    name = "hyperstack"

    def __init__(self, config: Hyperstack) -> None:
        self._config = config
        self._access_key: str | None = None
        self._secret_key: str | None = None
        self._access_key_id: int | None = None

    @classmethod
    async def create(cls, config: Hyperstack) -> HyperstackProvider:
        return cls(config)

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        api_key = get_api_key(self._config)
        regions = self._config.region
        has_volumes = bool(spec.volumes)

        if has_volumes:
            regions = _constrain_region_for_volumes(
                regions, self._config.object_storage_region,
            )

        if self._config.network_optimised:
            regions = _constrain_region_for_network(
                regions, self._config.network_optimised_regions,
            )

        async with HyperstackClient(api_key, config=self._config) as client:
            match regions:
                case str():
                    flavors = await client.list_flavors(region=regions)
                case tuple():
                    all_flavors = await client.list_flavors()
                    allowed = {r.upper() for r in regions}
                    flavors = [
                        f for f in all_flavors
                        if f.get("region_name", "").upper() in allowed
                    ]
                case None:
                    flavors = await client.list_flavors()
            pricebook = await client.get_pricebook()

            image_by_region: dict[str, str] = {}
            for flavor in flavors:
                r = flavor.get("region_name", "")
                if r and r not in image_by_region:
                    image_by_region[r] = (
                        self._config.image
                        or await _resolve_image(client, r)
                    )

        price_map = _build_price_map(pricebook)

        gpu_names = {f.get("gpu", "") for f in flavors if f.get("gpu")}
        log.debug(
            "Available GPUs: {gpus}, spec={spec}, price_map_sample={sample}",
            gpus=gpu_names,
            spec=spec.accelerator_name,
            sample=dict(list(price_map.items())[:5]),
        )

        net_regions = frozenset(
            r.upper() for r in self._config.network_optimised_regions
        )
        for flavor in flavors:
            if not _matches_spec(flavor, spec):
                continue
            region = flavor.get("region_name", "")
            yield _to_offer(flavor, price_map, net_regions, image_by_region.get(region))

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[HyperstackSpecific]:
        api_key = get_api_key(self._config)
        ssh_key_path = get_ssh_key_path()
        public_path, public_key = get_local_ssh_key()
        key_name = generate_key_name(public_path)

        env_name = f"skyward-{uuid.uuid4().hex[:8]}"
        offer_data: dict[str, str] = offer.specific
        flavor_name = offer_data["flavor_name"]
        region = offer_data["region"]

        async with HyperstackClient(api_key, config=self._config) as client:
            env = await client.create_environment(env_name, region)
            env_id = env["id"]
            features = env.get("features") or {}
            env_network_optimised = bool(features.get("network_optimised"))
            log.info(
                "Created environment {name} (id={eid}, network_optimised={net})",
                name=env_name, eid=env_id, net=env_network_optimised,
            )

            if self._config.network_optimised and not env_network_optimised:
                log.warning(
                    "Requested network_optimised but environment {name} in "
                    "region {region} reports network_optimised={actual}",
                    name=env_name, region=region, actual=env_network_optimised,
                )

            await client.import_keypair(env_name, key_name, public_key)
            log.info("Imported SSH keypair {name}", name=key_name)

            image_name = (
                offer_data.get("image_name")
                or self._config.image
                or await _resolve_image(client, region)
            )
            log.info("Resolved image: {img}", img=image_name)

            if spec.volumes:
                key = await client.create_access_key(
                    region=self._config.object_storage_region,
                )
                self._access_key = key["access_key"]
                self._secret_key = key.get("secret_key", "")
                self._access_key_id = key["id"]
                log.info("Created object storage access key for volumes")

                buckets = {v.bucket for v in spec.volumes}
                await _ensure_buckets(
                    self._access_key, self._secret_key, buckets,
                    endpoint=self._config.object_storage_endpoint,
                )

        return Cluster(
            id=f"hyperstack-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="ubuntu",
            use_sudo=True,
            shutdown_command=(
                "curl -sS -X DELETE"
                f" '{HYPERSTACK_API_BASE}/core/virtual-machines/{{instance_id}}'"
                f" -H 'api_key: {api_key}'"
            ),
            specific=HyperstackSpecific(
                environment_name=env_name,
                environment_id=env_id,
                key_name=key_name,
                flavor_name=flavor_name,
                image_name=image_name,
                region=region,
                network_optimised=env_network_optimised,
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
                    instances.append(_build_instance(vm, "provisioning", cluster))

                remaining -= batch_size

        return cluster, instances

    async def get_instance(
        self,
        cluster: Cluster[HyperstackSpecific],
        instance_id: str,
    ) -> tuple[Cluster[HyperstackSpecific], Instance | None]:
        api_key = get_api_key(self._config)
        vm_id = int(instance_id)

        async with HyperstackClient(api_key, config=self._config) as client:
            info = await client.get_vm(vm_id)

            if not info:
                return cluster, None

            match info.get("status", "").upper():
                case "SHUTOFF" | "HIBERNATED" | "ERROR" | "DELETING" | "DELETED":
                    return cluster, None
                case "ACTIVE" if info.get("floating_ip"):
                    cluster, ready = await _ensure_firewall(client, cluster, vm_id)
                    status = "provisioned" if ready else "provisioning"
                    return cluster, _build_instance(info, status, cluster)
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

    async def storage(self, cluster: Cluster[HyperstackSpecific]) -> Storage:
        from skyward.storage import Storage

        if self._access_key is None or self._secret_key is None:
            raise RuntimeError("Access key not created — call prepare() with volumes first")
        return Storage(
            endpoint=self._config.object_storage_endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            path_style=True,
        )

    async def teardown(
        self, cluster: Cluster[HyperstackSpecific],
    ) -> Cluster[HyperstackSpecific]:
        api_key = get_api_key(self._config)

        async with HyperstackClient(api_key, config=self._config) as client:
            if self._access_key_id:
                with suppress(Exception):
                    await client.delete_access_key(self._access_key_id)
                    log.info("Deleted object storage access key")
                self._access_key_id = None
            instance_ids = tuple(inst.id for inst in cluster.instances)
            if instance_ids:
                await asyncio.gather(*(
                    client.delete_vm(int(iid))
                    for iid in instance_ids
                ), return_exceptions=True)

                deadline = asyncio.get_event_loop().time() + self._config.teardown_timeout
                interval = self._config.teardown_poll_interval
                while asyncio.get_event_loop().time() < deadline:
                    await asyncio.sleep(interval)
                    vms = await asyncio.gather(*(
                        client.get_vm(int(iid)) for iid in instance_ids
                    ), return_exceptions=True)
                    if all(v is None or isinstance(v, Exception) for v in vms):
                        break

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


async def _ensure_buckets(
    access_key: str,
    secret_key: str,
    buckets: set[str],
    *,
    endpoint: str,
) -> None:
    """Create S3 buckets that don't exist yet via the S3-compatible API."""
    import aioboto3
    from botocore.exceptions import ClientError

    session = aioboto3.Session()
    async with session.client(  # pyright: ignore[reportGeneralTypeIssues]
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    ) as s3:
        for bucket in sorted(buckets):
            try:
                await s3.head_bucket(Bucket=bucket)
                log.debug("Bucket {b} already exists", b=bucket)
            except ClientError:
                await s3.create_bucket(Bucket=bucket)
                log.info("Created bucket {b}", b=bucket)


def _constrain_region_for_volumes(
    regions: str | tuple[str, ...] | None,
    storage_region: str,
) -> str | tuple[str, ...]:
    """Constrain region selection when volumes are requested.

    Object storage is only available in a specific region. When no region
    is specified, automatically select it. When a specific region is set,
    validate it includes the storage region.
    """
    match regions:
        case None:
            log.info("Volumes requested — restricting offers to {r}", r=storage_region)
            return storage_region
        case str() if regions.upper() != storage_region.upper():
            raise ValueError(
                f"Volumes require region {storage_region}, "
                f"but region is set to '{regions}'."
            )
        case tuple() if storage_region.upper() not in {r.upper() for r in regions}:
            raise ValueError(
                f"Volumes require region {storage_region}, "
                f"but configured regions {regions} do not include it."
            )
    return regions


def _constrain_region_for_network(
    regions: str | tuple[str, ...] | None,
    allowed_regions: tuple[str, ...],
) -> str | tuple[str, ...]:
    """Constrain region selection when network_optimised is requested.

    Network-optimised environments (SR-IOV, up to 350 Gbps) are only
    available in certain regions. When no region is specified, restrict
    to network-optimised regions. When specific regions are set, validate
    they are all network-optimised.
    """
    allowed_upper = {r.upper() for r in allowed_regions}
    match regions:
        case None:
            allowed = tuple(sorted(allowed_regions))
            log.info(
                "Network optimised — restricting offers to {r}", r=allowed,
            )
            return allowed
        case str() if regions.upper() not in allowed_upper:
            raise ValueError(
                f"network_optimised=True requires a supported region "
                f"({', '.join(sorted(allowed_regions))}), "
                f"but region is set to '{regions}'."
            )
        case tuple():
            unsupported = {
                r for r in regions
                if r.upper() not in allowed_upper
            }
            if unsupported:
                raise ValueError(
                    f"network_optimised=True requires all regions to be "
                    f"network-optimised ({', '.join(sorted(allowed_regions))}), "
                    f"but {unsupported} are not supported."
                )
    return regions


async def _ensure_firewall(
    client: HyperstackClient,
    cluster: Cluster[HyperstackSpecific],
    vm_id: int,
) -> tuple[Cluster[HyperstackSpecific], bool]:
    """Open SSH (22) publicly and all ports between cluster peers.

    Returns ``(cluster, ready)`` where *ready* is True only when SSH and
    **all** peer rules have been applied.  When a peer VM hasn't received
    its private IP yet the function returns ``ready=False`` so the caller
    keeps the instance in ``"provisioning"`` and the node actor continues
    polling until every peer is reachable.
    """
    specific = cluster.specific

    if vm_id in specific.firewall_applied:
        return cluster, True

    with suppress(Exception):
        await client.add_firewall_rule(vm_id, port_min=22, port_max=22)

    peers: list[tuple[int, str]] = []
    all_resolved = True
    for inst in cluster.instances:
        peer_id = int(inst.id)
        if peer_id == vm_id:
            continue
        pip = inst.private_ip
        if not pip:
            peer_info = await client.get_vm(peer_id)
            pip = peer_info.get("fixed_ip", "") if peer_info else ""
        if pip:
            peers.append((peer_id, pip))
        else:
            all_resolved = False

    log.debug(
        "Firewall VM {vm_id}: peers_resolved={peers}, all_resolved={ok}",
        vm_id=vm_id, peers=peers, ok=all_resolved,
    )

    if not all_resolved:
        return cluster, False

    try:
        await asyncio.gather(*(
            client.add_firewall_rule(
                vm_id, port_min=1, port_max=65535,
                protocol=proto, remote_ip=f"{peer_ip}/32",
            )
            for _, peer_ip in peers
            for proto in ("tcp", "udp")
        ))
    except Exception as e:
        log.warning("Peer firewall rules for VM {vm_id} failed: {e}", vm_id=vm_id, e=e)
        return cluster, False

    updated = replace(specific, firewall_applied=specific.firewall_applied | {vm_id})
    return replace(cluster, specific=updated), True


def _self_destruction_script(ttl: int, shutdown_command: str) -> str:
    from skyward.providers.bootstrap.compose import resolve
    from skyward.providers.bootstrap.ops import instance_timeout

    lines = ["#!/bin/bash", "set -e"]
    if ttl:
        lines.append(resolve(instance_timeout(ttl, shutdown_command=shutdown_command)))
    return "\n".join(lines) + "\n"


def _build_price_map(pricebook: list[PricebookEntry]) -> dict[str, float]:
    """Build resource name -> hourly price map.

    The pricebook entries use ``name`` for the resource/GPU and ``value``
    for the hourly price. Entries have no region field — the list is global.
    """
    prices: dict[str, float] = {}
    for entry in pricebook:
        name = entry.get("name", "")
        raw = entry.get("value", 0)
        try:
            price = float(raw) if raw else 0.0
        except (ValueError, TypeError):
            continue
        if name and price > 0:
            prices[name.upper()] = price
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


def _to_offer(
    flavor: FlavorResponse,
    price_map: dict[str, float],
    network_optimised_regions: frozenset[str],
    image_name: str | None = None,
) -> Offer:
    """Convert a Hyperstack flavor to a Skyward Offer."""
    gpu_name = flavor.get("gpu", "")
    gpu_count = flavor.get("gpu_count", 0)
    is_spot = gpu_name.lower().endswith("-spot")
    base_gpu = gpu_name.removesuffix("-spot").removesuffix("-Spot").removesuffix("-SPOT") if is_spot else gpu_name

    accel = Accelerator(name=base_gpu, count=gpu_count) if gpu_name else None

    it = InstanceType(
        name=flavor["name"],
        accelerator=accel,
        vcpus=float(flavor.get("cpu", 0)),
        memory_gb=float(flavor.get("ram", 0)),
        architecture="x86_64",
        specific=None,
    )

    gpu_upper = gpu_name.upper()
    per_gpu = price_map.get(gpu_upper, 0.0) or price_map.get(base_gpu.upper(), 0.0)
    hourly = per_gpu * gpu_count
    region = flavor.get("region_name", "")

    specific: dict[str, Any] = {
        "flavor_name": flavor["name"],
        "region": region,
        "network_optimised": region.upper() in network_optimised_regions,
    }
    if image_name:
        specific["image_name"] = image_name

    return Offer(
        id=str(flavor["id"]),
        instance_type=it,
        spot_price=hourly if hourly > 0 and is_spot else None,
        on_demand_price=hourly if hourly > 0 and not is_spot else None,
        billing_unit="hour",
        specific=specific,
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
        spot=cluster.offer.spot_price is not None,
        region=cluster.specific.region,
    )


async def _resolve_image(client: HyperstackClient, region: str) -> str:
    """Find the newest Ubuntu + CUDA image available.

    Priority: Ubuntu + CUDA (newest first), then any Ubuntu, then any image.
    """
    images = await client.list_images(region=region)
    if not images:
        raise RuntimeError(f"No images found in region {region}")

    by_date = sorted(images, key=lambda i: i.get("created_at", ""), reverse=True)

    ubuntu_cuda = [i for i in by_date if "ubuntu" in i["name"].lower() and "cuda" in i["name"].lower()]
    if ubuntu_cuda:
        return ubuntu_cuda[0]["name"]

    ubuntu = [i for i in by_date if "ubuntu" in i["name"].lower()]
    if ubuntu:
        return ubuntu[0]["name"]

    return by_date[0]["name"]
