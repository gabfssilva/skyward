"""Scaleway provider implementation.

All state flows through Cluster[ScalewaySpecific]. The provider is stateless.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.core import PoolSpec
from skyward.core.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import generate_key_name, get_local_ssh_key, get_ssh_key_path

from .client import ScalewayClient, ScalewayError, get_project_id, get_secret_key
from .config import Scaleway
from .types import (
    ImageResponse,
    ServerResponse,
    ServerTypeResponse,
)

log = logger.bind(provider="scaleway")

GPU_ZONES = ("fr-par-1", "fr-par-2", "fr-par-3", "nl-ams-1", "nl-ams-2", "pl-waw-3")


@dataclass(frozen=True, slots=True)
class ScalewaySpecific:
    """Scaleway-specific cluster data flowing through Cluster[ScalewaySpecific]."""

    commercial_type: str
    zone: str
    project_id: str
    image_id: str
    ssh_key_id: str


class ScalewayProvider(Provider[Scaleway, ScalewaySpecific]):
    """Scaleway GPU cloud provider."""

    name = "scaleway"

    def __init__(self, config: Scaleway) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: Scaleway) -> ScalewayProvider:
        return cls(config)

    async def offers(self) -> AsyncIterator[Offer]:
        secret_key = get_secret_key(self._config)
        zones = (self._config.zone,) if self._config.zone else GPU_ZONES

        async def _fetch_zone(zone: str) -> dict[str, ServerTypeResponse]:
            async with ScalewayClient(secret_key, zone, config=self._config) as client:
                return await client.list_server_types()

        results = await asyncio.gather(*(_fetch_zone(z) for z in zones))

        for zone, server_types in zip(zones, results, strict=True):
            for name, st in server_types.items():
                gpu_count = st.get("gpu", 0)
                gpu_info = st.get("gpu_info")

                gpu_name = gpu_info.get("gpu_name", "") if gpu_info else ""
                vcpus = float(st.get("ncpus", 0))

                ram_bytes = st.get("ram", 0)
                memory_gb = ram_bytes / (1024 ** 3) if ram_bytes > 1000 else float(ram_bytes)

                vram_bytes = gpu_info.get("gpu_memory", 0) if gpu_info else 0
                vram_gb = vram_bytes / (1024 ** 3) if vram_bytes > 1000 else 0

                hourly = st.get("hourly_price", 0.0)

                accel: Accelerator | None = None
                if gpu_count and gpu_name:
                    memory_str = f"{int(vram_gb)}GB" if vram_gb > 0 else ""
                    accel = Accelerator(name=gpu_name, count=gpu_count, memory=memory_str)

                it = InstanceType(
                    name=name,
                    accelerator=accel,
                    vcpus=vcpus,
                    memory_gb=memory_gb,
                    architecture=st.get("arch", "x86_64"),  # type: ignore[arg-type]
                    specific=None,
                )

                yield Offer(
                    id=f"scaleway-{zone}-{name}",
                    instance_type=it,
                    spot_price=None,
                    on_demand_price=hourly if hourly > 0 else None,
                    billing_unit="hour",
                    specific={
                        "commercial_type": name,
                        "zone": zone,
                    },
                )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[ScalewaySpecific]:
        secret_key = get_secret_key(self._config)
        project_id = get_project_id(self._config)
        ssh_key_path = get_ssh_key_path()
        public_path, public_key = get_local_ssh_key()
        key_name = generate_key_name(public_path)

        offer_data: dict = offer.specific
        commercial_type = offer_data["commercial_type"]
        zone = offer_data["zone"]

        async with ScalewayClient(secret_key, zone, config=self._config) as client:
            ssh_key_id = await _ensure_ssh_key(client, key_name, public_key, project_id)
            log.info("SSH key ready: {kid}", kid=ssh_key_id)

            image_id = self._config.image or await _resolve_image(client)
            log.info("Resolved image: {img}", img=image_id)

        return Cluster(
            id=f"scaleway-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="root",
            use_sudo=False,
            shutdown_command="shutdown -h now",
            specific=ScalewaySpecific(
                commercial_type=commercial_type,
                zone=zone,
                project_id=project_id,
                image_id=image_id,
                ssh_key_id=ssh_key_id,
            ),
        )

    async def provision(
        self,
        cluster: Cluster[ScalewaySpecific],
        count: int,
    ) -> tuple[Cluster[ScalewaySpecific], Sequence[Instance]]:
        secret_key = get_secret_key(self._config)
        specific = cluster.specific
        instances: list[Instance] = []

        async with ScalewayClient(secret_key, specific.zone, config=self._config) as client:
            for i in range(count):
                name = f"skyward-{cluster.id.split('-', 1)[-1]}-{i}"
                server_id = ""

                try:
                    server = await client.create_server(
                        name=name,
                        commercial_type=specific.commercial_type,
                        image=specific.image_id,
                        project=specific.project_id,
                        tags=["skyward", cluster.id],
                    )
                    server_id = server["id"]
                    log.info("Created server {name} (id={sid})", name=name, sid=server_id)

                    await client.server_action(server_id, "poweron")
                    log.info("Powered on server {sid}", sid=server_id)
                except ScalewayError:
                    log.warning("Failed to provision {name}, rolling back", name=name)
                    if server_id:
                        with contextlib.suppress(Exception):
                            await client.delete_server(server_id)
                    raise

                instances.append(Instance(
                    id=server_id,
                    status="provisioning",
                    offer=cluster.offer,
                    ip="",
                    ssh_port=22,
                    spot=False,
                    region=specific.zone,
                ))

        return cluster, instances

    async def get_instance(
        self,
        cluster: Cluster[ScalewaySpecific],
        instance_id: str,
    ) -> tuple[Cluster[ScalewaySpecific], Instance | None]:
        secret_key = get_secret_key(self._config)

        async with ScalewayClient(secret_key, cluster.specific.zone, config=self._config) as client:
            info = await client.get_server(instance_id)

        if not info:
            return cluster, None

        match info.get("state", ""):
            case "running":
                ip = _extract_ip(info)
                status: InstanceStatus = "provisioned" if ip else "provisioning"
                return cluster, _build_instance(info, status, ip, cluster)
            case "starting":
                return cluster, _build_instance(info, "provisioning", _extract_ip(info), cluster)
            case "stopped" | "stopped_in_place" | "stopping" | "locked":
                return cluster, None
            case _:
                return cluster, _build_instance(info, "provisioning", _extract_ip(info), cluster)

    async def terminate(
        self,
        cluster: Cluster[ScalewaySpecific],
        instance_ids: tuple[str, ...],
    ) -> Cluster[ScalewaySpecific]:
        if not instance_ids:
            return cluster
        secret_key = get_secret_key(self._config)

        async with ScalewayClient(secret_key, cluster.specific.zone, config=self._config) as client:
            volume_ids = await _collect_volume_ids(client, instance_ids)

            async def _kill(sid: str) -> None:
                try:
                    await client.server_action(sid, "terminate")
                except ScalewayError as e:
                    if e.status != 404:
                        log.error("Failed to terminate server {sid}: {err}", sid=sid, err=e)

            await asyncio.gather(*(_kill(sid) for sid in instance_ids))
            await _delete_volumes(client, volume_ids)

        return cluster

    async def teardown(
        self,
        cluster: Cluster[ScalewaySpecific],
    ) -> Cluster[ScalewaySpecific]:
        secret_key = get_secret_key(self._config)
        specific = cluster.specific
        ids = tuple(inst.id for inst in cluster.instances)

        async with ScalewayClient(secret_key, specific.zone, config=self._config) as client:
            volume_ids = await _collect_volume_ids(client, ids)

            for inst in cluster.instances:
                try:
                    await client.server_action(inst.id, "terminate")
                except Exception as e:
                    log.error("Failed to terminate server {sid}: {err}", sid=inst.id, err=e)

            await _delete_volumes(client, volume_ids)

        return cluster


# =============================================================================
# Helpers
# =============================================================================


def _extract_ip(info: ServerResponse) -> str:
    """Extract public IP from server response."""
    if pub := info.get("public_ip"):
        return pub.get("address", "")
    for ip in info.get("public_ips", []):
        if addr := ip.get("address"):
            return addr
    return ""


def _build_instance(
    info: ServerResponse,
    status: InstanceStatus,
    ip: str,
    cluster: Cluster[ScalewaySpecific],
) -> Instance:
    return Instance(
        id=info["id"],
        status=status,
        offer=cluster.offer,
        ip=ip,
        private_ip=info.get("private_ip"),
        ssh_port=22,
        spot=False,
        region=cluster.specific.zone,
    )


async def _collect_volume_ids(
    client: ScalewayClient,
    server_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Fetch SBS volume IDs attached to servers before termination."""

    async def _get_volumes(sid: str) -> list[str]:
        info = await client.get_server(sid)
        if not info:
            return []
        return [
            vol["id"]
            for vol in (info.get("volumes") or {}).values()
            if vol.get("volume_type", "").startswith("sbs")
        ]

    results = await asyncio.gather(*(_get_volumes(sid) for sid in server_ids))
    return tuple(vid for vids in results for vid in vids)


async def _delete_volumes(
    client: ScalewayClient,
    volume_ids: tuple[str, ...],
    *,
    max_attempts: int = 8,
    base_delay: float = 2.0,
) -> None:
    """Delete orphaned SBS volumes after server termination.

    Retries with exponential backoff because the Scaleway terminate action
    is async — volumes remain ``in_use`` (HTTP 412) until the server is
    fully gone.
    """
    if not volume_ids:
        return

    async def _del(vid: str) -> None:
        for attempt in range(max_attempts):
            try:
                await client.delete_block_volume(vid)
                log.info("Deleted SBS volume {vid}", vid=vid)
                return
            except ScalewayError as e:
                if e.status != 412 or attempt == max_attempts - 1:
                    log.warning("Failed to delete SBS volume {vid}: {err}", vid=vid, err=e)
                    return
                delay = base_delay * (2 ** attempt)
                log.debug("Volume {vid} still in_use, retrying in {d}s", vid=vid, d=delay)
                await asyncio.sleep(delay)

    await asyncio.gather(*(_del(vid) for vid in volume_ids))


async def _ensure_ssh_key(
    client: ScalewayClient,
    key_name: str,
    public_key: str,
    project_id: str,
) -> str:
    """Ensure SSH key exists on Scaleway IAM."""
    from skyward.providers.ssh_keys import compute_fingerprint

    local_fp = compute_fingerprint(public_key)
    existing = await client.list_ssh_keys()

    for key in existing:
        if key.get("fingerprint") == local_fp:
            log.debug("Found existing SSH key by fingerprint")
            return key["id"]
        if key.get("name") == key_name:
            log.debug("Found existing SSH key by name")
            return key["id"]

    log.info("Creating SSH key {name}", name=key_name)
    new_key = await client.create_ssh_key(key_name, public_key, project_id)
    return new_key["id"]


async def _resolve_image(client: ScalewayClient) -> str:
    """Find the newest Ubuntu GPU image with sbs_snapshot volume type.

    GPU instances (L4, etc.) require zero lssd volumes, so we must select
    images backed by sbs_snapshot rather than l_ssd.

    Priority: Ubuntu GPU sbs_snapshot > Ubuntu sbs_snapshot > any sbs_snapshot.
    """
    images = await client.list_images()
    if not images:
        raise RuntimeError("No public images found in zone")

    by_date = sorted(images, key=lambda i: i.get("creation_date", ""), reverse=True)

    def _is_sbs(img: ImageResponse) -> bool:
        root = img.get("root_volume")  # type: ignore[attr-defined]
        return root.get("volume_type") == "sbs_snapshot" if root else False

    sbs_images = [i for i in by_date if _is_sbs(i)]
    candidates = sbs_images or by_date

    ubuntu_gpu = [
        i for i in candidates
        if "ubuntu" in i.get("name", "").lower()
        and any(t in i.get("name", "").lower() for t in ("gpu", "cuda"))
    ]
    if ubuntu_gpu:
        return ubuntu_gpu[0]["id"]

    ubuntu = [i for i in candidates if "ubuntu" in i.get("name", "").lower()]
    if ubuntu:
        return ubuntu[0]["id"]

    return candidates[0]["id"]
