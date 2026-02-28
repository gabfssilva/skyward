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
from skyward.providers.ssh_keys import (
    ensure_ssh_key_on_provider,
    get_local_ssh_key,
    get_ssh_key_path,
)

from .client import ThunderClient, ThunderError, get_api_key
from .config import ThunderCompute
from .types import InstanceListItem, get_display_name, get_vram_gb, resolve_gpu_type

log = logger.bind(provider="thunder")


@dataclass(frozen=True, slots=True)
class ThunderSpecific:
    """Thunder-specific cluster data flowing through Cluster[ThunderSpecific]."""

    ssh_key_id: str
    ssh_public_key: str
    gpu_type: str
    mode: str
    template: str
    disk_size_gb: int
    cpu_cores: int


class ThunderProvider(Provider[ThunderCompute, ThunderSpecific]):
    """Stateless Thunder provider. Holds only immutable config + client."""

    def __init__(self, config: ThunderCompute, client: ThunderClient) -> None:
        self._config = config
        self._client = client

    @classmethod
    async def create(cls, config: ThunderCompute) -> ThunderProvider:
        from skyward.infra.http import BearerAuth, HttpClient

        from .client import THUNDER_API_BASE

        api_key = get_api_key(config.api_token)
        auth = BearerAuth(api_key)
        http_client = HttpClient(THUNDER_API_BASE, auth, timeout=config.request_timeout)
        client = ThunderClient(http_client)
        return cls(config, client)

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        if not spec.accelerator_name:
            return

        gpu_type = resolve_gpu_type(spec.accelerator_name)
        if not gpu_type:
            log.debug(
                "Unknown accelerator '{name}' for Thunder",
                name=spec.accelerator_name,
            )
            return

        pricing = await self._client.get_pricing()
        mode = self._config.mode
        num_gpus = spec.accelerator_count or 1

        gpu_pricing = pricing.get(gpu_type)
        if not gpu_pricing:
            log.debug("No pricing for gpu_type={gpu}", gpu=gpu_type)
            return

        hourly_rate: float | None = None
        match mode:
            case "production":
                hourly_rate = gpu_pricing.get("production")
            case "prototyping":
                hourly_rate = gpu_pricing.get("prototyping")

        if hourly_rate is None:
            log.debug("No {mode} pricing for {gpu}", mode=mode, gpu=gpu_type)
            return

        per_instance_rate = hourly_rate * num_gpus
        if spec.max_hourly_cost and per_instance_rate > spec.max_hourly_cost / max(spec.nodes, 1):
            return

        display_name = get_display_name(gpu_type)
        vram = get_vram_gb(gpu_type)

        accel = Accelerator(
            name=display_name,
            count=num_gpus,
            memory=f"{vram}GB" if vram else "",
        )

        it = InstanceType(
            name=f"{display_name} x{num_gpus}",
            accelerator=accel,
            vcpus=float(self._config.cpu_cores),
            memory_gb=0.0,
            architecture="x86_64",
            specific=None,
        )

        yield Offer(
            id=f"thunder-{gpu_type}-{mode}",
            instance_type=it,
            spot_price=None,
            on_demand_price=per_instance_rate,
            billing_unit="minute",
            specific=gpu_type,
        )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[ThunderSpecific]:
        ssh_key_path = get_ssh_key_path()
        _, ssh_public_key = get_local_ssh_key()

        ssh_key_id = await ensure_ssh_key_on_provider(
            list_keys_fn=self._client.list_ssh_keys,  # type: ignore[reportArgumentType]
            create_key_fn=lambda name, key: self._client.add_ssh_key(name, key),  # type: ignore[reportArgumentType]
            provider_name="thunder",
        )
        log.debug("SSH key ensured: id={kid}", kid=ssh_key_id)

        gpu_type = offer.specific if isinstance(offer.specific, str) else "h100"

        return Cluster(
            id=f"thunder-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="ubuntu",
            use_sudo=True,
            shutdown_command="shutdown -h now",
            specific=ThunderSpecific(
                ssh_key_id=ssh_key_id,
                ssh_public_key=ssh_public_key,
                gpu_type=gpu_type,
                mode=self._config.mode,
                template=self._config.template,
                disk_size_gb=self._config.disk_size_gb,
                cpu_cores=self._config.cpu_cores,
            ),
        )

    async def provision(
        self, cluster: Cluster[ThunderSpecific], count: int,
    ) -> tuple[Cluster[ThunderSpecific], Sequence[Instance]]:
        specific = cluster.specific
        num_gpus = (
            cluster.offer.instance_type.accelerator.count
            if cluster.offer.instance_type.accelerator
            else 1
        )

        instances: list[Instance] = []
        for i in range(count):
            log.info("Provisioning instance {i}/{count}", i=i + 1, count=count)
            try:
                resp = await self._client.create_instance(
                    cpu_cores=specific.cpu_cores,
                    disk_size_gb=specific.disk_size_gb,
                    gpu_type=specific.gpu_type,
                    num_gpus=num_gpus,
                    mode=specific.mode,
                    template=specific.template,
                    public_key=specific.ssh_public_key,
                )
            except ThunderError as e:
                log.error("Failed to create instance: {err}", err=e)
                continue

            instances.append(Instance(
                id=resp["uuid"],
                status="provisioning",
                offer=cluster.offer,
                spot=False,
                region="ca-qc",
            ))

        if not instances:
            raise RuntimeError(
                f"Failed to provision any of the {count} requested instances",
            )

        return cluster, instances

    async def get_instance(
        self, cluster: Cluster[ThunderSpecific], instance_id: str,
    ) -> tuple[Cluster[ThunderSpecific], Instance | None]:
        all_instances = await self._client.list_instances()
        info = all_instances.get(instance_id)

        if not info:
            return cluster, None

        match info["status"]:
            case "DELETING":
                return cluster, None
            case "RUNNING" if info.get("ip"):
                return cluster, _build_instance(info, "provisioned", cluster)
            case _:
                return cluster, _build_instance(info, "provisioning", cluster)

    async def terminate(
        self, cluster: Cluster[ThunderSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[ThunderSpecific]:
        if not instance_ids:
            return cluster

        async def _destroy(iid: str) -> None:
            try:
                await self._client.delete_instance(iid)
            except Exception as e:
                log.error("Failed to delete instance {iid}: {err}", iid=iid, err=e)

        await asyncio.gather(*(_destroy(iid) for iid in instance_ids))
        return cluster

    async def teardown(self, cluster: Cluster[ThunderSpecific]) -> Cluster[ThunderSpecific]:
        await self._client.close()
        return cluster


def _build_instance(
    info: InstanceListItem,
    status: InstanceStatus,
    cluster: Cluster[ThunderSpecific],
) -> Instance:
    return Instance(
        id=info["uuid"],
        status=status,
        offer=cluster.offer,
        ip=info.get("ip", ""),
        ssh_port=info.get("port", 22),
        spot=False,
        region="ca-qc",
    )
