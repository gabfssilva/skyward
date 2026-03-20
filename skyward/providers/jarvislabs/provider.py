"""Jarvis Labs provider implementation.

Provisions GPU instances on Jarvis Labs via their Python SDK.
Sync SDK calls are dispatched to a dedicated ThreadPoolExecutor.
Per-minute billing, three regions (IN1, IN2, EU1), no spot.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from skyward.accelerators import Accelerator
from skyward.accelerators.catalog import SPECS
from skyward.core import PoolSpec
from skyward.core.model import Cluster, Instance, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import (
    compute_fingerprint,
    ensure_ssh_key_on_provider,
    get_ssh_key_path,
)

from .config import JarvisLabs

log = logger.bind(provider="jarvislabs")

_MEMORY_SUFFIX = re.compile(r"[-_]?\d+GB$", re.IGNORECASE)
_CANONICAL_CACHE: dict[str, str] = {}


def _normalize(s: str) -> str:
    return re.sub(r"[\s_\-]", "", s).upper()


def _resolve_canonical(raw: str) -> str:
    """Resolve a JarvisLabs GPU type to its canonical catalog name.

    Strips memory suffixes (e.g. ``A100-80GB`` → ``A100``), then checks
    the accelerator catalog for an exact or suffix match.  Results are
    cached so the catalog is scanned at most once per unique raw name.
    """
    if cached := _CANONICAL_CACHE.get(raw):
        return cached

    stripped = _MEMORY_SUFFIX.sub("", raw)
    norm = _normalize(stripped)

    for name in SPECS:
        if _normalize(name) == norm:
            _CANONICAL_CACHE[raw] = name
            return name

    for name in SPECS:
        if _normalize(name).endswith(norm):
            _CANONICAL_CACHE[raw] = name
            return name

    _CANONICAL_CACHE[raw] = raw
    return raw


def _gpu_matches(jl_gpu_type: str, requested: str) -> bool:
    """Check if a Jarvis Labs GPU type matches a requested name."""
    canonical = _resolve_canonical(jl_gpu_type)
    return _normalize(canonical) == _normalize(requested)


def _parse_ssh_command(ssh_command: str) -> tuple[str, int]:
    """Parse SSH command string to extract host and port.

    Input format: 'ssh -o StrictHostKeyChecking=no -p {port} root@{host}'

    Returns
    -------
    tuple[str, int]
        (hostname, port)
    """
    import contextlib

    parts = ssh_command.split()
    host = ""
    port = 22
    for i, part in enumerate(parts):
        if part == "-p" and i + 1 < len(parts):
            with contextlib.suppress(ValueError):
                port = int(parts[i + 1])
        if "@" in part and not part.startswith("-"):
            host = part.split("@", 1)[1]
    return host, port


@dataclass(frozen=True, slots=True)
class JarvisLabsOfferData:
    """Carried in Offer.specific -- GPU + region for provisioning."""

    jl_gpu_type: str
    region: str


@dataclass(frozen=True, slots=True)
class JarvisLabsSpecific:
    """Jarvis Labs-specific cluster data flowing through Cluster[JarvisLabsSpecific]."""

    jl_gpu_type: str
    region: str
    template: str
    ssh_key_id: str


_REGION_DISPLAY_TO_INTERNAL: dict[str, str] = {
    "IN1": "india-01",
    "IN2": "india-noida-01",
    "EU1": "europe-01",
}

_REGION_INTERNAL_TO_DISPLAY: dict[str, str] = {v: k for k, v in _REGION_DISPLAY_TO_INTERNAL.items()}


def _to_internal_region(region: str) -> str:
    """Convert display code (IN1) to internal ID (india-01), or passthrough."""
    return _REGION_DISPLAY_TO_INTERNAL.get(region.upper(), region)


def _to_display_region(region: str) -> str:
    """Convert internal ID (india-noida-01) to display code (IN2), or passthrough."""
    return _REGION_INTERNAL_TO_DISPLAY.get(region, region)


def _region_matches(gpu_region: str, config_region: str) -> bool:
    """Compare regions accounting for display code vs internal ID."""
    return _normalize(gpu_region) == _normalize(_to_internal_region(config_region))


def _get_token(config: JarvisLabs) -> str:
    if token := config.api_key or os.environ.get("JL_API_KEY"):
        return token
    raise RuntimeError(
        "Jarvis Labs API key required. Set JL_API_KEY env var "
        "or pass api_key= to JarvisLabs()."
    )


class JarvisLabsProvider(Provider[JarvisLabs, JarvisLabsSpecific]):
    """Jarvis Labs GPU cloud provider.

    Sync SDK calls dispatched to a dedicated thread pool.
    """

    name = "jarvislabs"

    def __init__(
        self,
        config: JarvisLabs,
        client: Any,
        thread_pool: ThreadPoolExecutor,
    ) -> None:
        self._config = config
        self._client = client
        self._pool = thread_pool

    async def _run[T](self, fn: Callable[..., T], *args: object, **kwargs: object) -> T:
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                self._pool, lambda: fn(*args, **kwargs),
            )
        return await loop.run_in_executor(self._pool, fn, *args)

    @classmethod
    async def create(cls, config: JarvisLabs) -> JarvisLabsProvider:
        from jarvislabs import Client  # type: ignore[reportMissingImports]

        token = _get_token(config)
        thread_pool = ThreadPoolExecutor(
            max_workers=config.thread_pool_size,
            thread_name_prefix="jl-io",
        )
        client = Client(api_key=token)
        return cls(config=config, client=client, thread_pool=thread_pool)

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        gpu_list = await self._run(self._client.account.gpu_availability)
        num_gpus = int(spec.accelerator_count or 1)

        log.debug(
            "GPU availability: {gpus}",
            gpus=[
                {"type": g.gpu_type, "free": g.num_free_devices, "region": g.region, "vram": g.vram}
                for g in gpu_list
            ],
        )

        candidates: list[tuple[Any, float]] = []
        for gpu in gpu_list:
            jl_type = gpu.gpu_type
            free = gpu.num_free_devices
            price = gpu.price_per_hour or 0.0
            region = gpu.region
            vram_gb = int(gpu.vram) if gpu.vram else 0

            if free < num_gpus:
                continue
            if self._config.region and not _region_matches(region, self._config.region):
                continue
            if spec.accelerator_name and not _gpu_matches(jl_type, spec.accelerator_name):
                continue
            if spec.accelerator_memory_gb > 0 and vram_gb < spec.accelerator_memory_gb:
                continue

            hourly = price * num_gpus
            candidates.append((gpu, hourly))

        candidates.sort(key=lambda c: c[1])

        for gpu, hourly in candidates:
            jl_type = str(gpu.gpu_type)
            region = str(gpu.region)
            vram_gb = int(gpu.vram) if gpu.vram else 0
            cpus = gpu.cpus_per_gpu or 0
            ram = gpu.ram_per_gpu or 0
            display = _resolve_canonical(jl_type)
            accel = Accelerator(
                name=display,
                memory=f"{vram_gb}GB" if vram_gb else "",
                count=num_gpus,
            )
            it = InstanceType(
                name=display,
                accelerator=accel,
                vcpus=float(cpus * num_gpus),
                memory_gb=float(ram * num_gpus),
                architecture="x86_64",
                specific=None,
            )
            yield Offer(
                id=f"jl-{region}-{jl_type}",
                instance_type=it,
                spot_price=None,
                on_demand_price=hourly,
                billing_unit="minute",
                specific=JarvisLabsOfferData(
                    jl_gpu_type=jl_type,
                    region=region,
                ),
            )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[JarvisLabsSpecific]:
        ssh_key_path = get_ssh_key_path()
        raw = offer.specific
        offer_data = raw if isinstance(raw, JarvisLabsOfferData) else JarvisLabsOfferData(**raw)

        async def _list_keys() -> list[dict[str, Any]]:
            keys = await self._run(self._client.ssh_keys.list)
            return [
                {
                    "id": str(k.key_id),
                    "name": k.key_name,
                    "fingerprint": compute_fingerprint(k.ssh_key),
                }
                for k in keys
            ]

        async def _create_key(name: str, public_key: str) -> dict[str, Any]:
            await self._run(self._client.ssh_keys.add, public_key, name)
            keys = await self._run(self._client.ssh_keys.list)
            for k in keys:
                if k.key_name == name:
                    return {"id": str(k.key_id), "name": k.key_name}
            return {"id": "unknown", "name": name}

        ssh_key_id = await ensure_ssh_key_on_provider(
            _list_keys, _create_key, "jarvislabs",
        )

        template = self._config.template
        match template:
            case "vm":
                ssh_user = "cloud"
                use_sudo = True
            case _:
                ssh_user = "root"
                use_sudo = False

        storage_gb = spec.disk_gb or self._config.storage_gb
        if offer_data.region == "europe-01" or template == "vm":
            storage_gb = max(storage_gb, 100)

        return Cluster(
            id=f"jarvislabs-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user=ssh_user,
            use_sudo=use_sudo,
            shutdown_command="shutdown -h now",
            specific=JarvisLabsSpecific(
                jl_gpu_type=offer_data.jl_gpu_type,
                region=offer_data.region,
                template=template,
                ssh_key_id=ssh_key_id,
            ),
        )

    async def provision(
        self, cluster: Cluster[JarvisLabsSpecific], count: int,
    ) -> tuple[Cluster[JarvisLabsSpecific], Sequence[Instance]]:
        specific = cluster.specific
        num_gpus = int(cluster.spec.accelerator_count or 1)
        storage_gb = cluster.spec.disk_gb or self._config.storage_gb
        if specific.region == "europe-01" or specific.template == "vm":
            storage_gb = max(storage_gb, 100)

        async def _create_one(idx: int) -> Instance | None:
            name = f"skyward-{cluster.id}-{idx:04d}"
            try:
                sdk_inst = await self._run(
                    self._client.instances.create,
                    gpu_type=specific.jl_gpu_type,
                    num_gpus=num_gpus,
                    template=specific.template,
                    storage=storage_gb,
                    name=name,
                    region=_to_display_region(specific.region),
                )
            except Exception as e:
                log.error(
                    "Failed to create instance {idx}/{count}: {err}",
                    idx=idx + 1, count=count, err=e,
                )
                return None

            machine_id = str(sdk_inst.machine_id)
            log.info(
                "Created instance {name}: id={id}",
                name=name, id=machine_id,
            )

            ssh_cmd = sdk_inst.ssh_command or ""
            host, port = _parse_ssh_command(ssh_cmd) if ssh_cmd else ("", 22)

            return Instance(
                id=machine_id,
                status="provisioned" if host else "provisioning",
                offer=cluster.offer,
                ip=host or sdk_inst.public_ip,
                ssh_port=port,
                spot=False,
                region=specific.region,
            )

        results = await asyncio.gather(*(_create_one(i) for i in range(count)))
        instances = [inst for inst in results if inst is not None]

        if not instances:
            raise RuntimeError(
                f"Failed to create any Jarvis Labs instances (requested {count})",
            )

        log.info("Provisioned {n} instances", n=len(instances))
        return cluster, instances

    async def get_instance(
        self, cluster: Cluster[JarvisLabsSpecific], instance_id: str,
    ) -> tuple[Cluster[JarvisLabsSpecific], Instance | None]:
        try:
            sdk_inst = await self._run(
                self._client.instances.get,
                machine_id=int(instance_id),
            )
        except Exception as e:
            log.warning(
                "Failed to get instance {id}: {err}",
                id=instance_id, err=e,
            )
            return cluster, None

        status = sdk_inst.status or ""
        ssh_cmd = sdk_inst.ssh_command or ""

        log.debug(
            "Instance {id}: status={status}",
            id=instance_id, status=status,
        )

        match status:
            case "Failed":
                return cluster, None
            case "Running" if ssh_cmd:
                host, port = _parse_ssh_command(ssh_cmd)
                if host:
                    return cluster, Instance(
                        id=instance_id,
                        status="provisioned",
                        offer=cluster.offer,
                        ip=host,
                        ssh_port=port,
                        spot=False,
                        region=cluster.specific.region,
                    )
                return cluster, Instance(
                    id=instance_id,
                    status="provisioning",
                    offer=cluster.offer,
                    spot=False,
                    region=cluster.specific.region,
                )
            case _:
                return cluster, Instance(
                    id=instance_id,
                    status="provisioning",
                    offer=cluster.offer,
                    spot=False,
                    region=cluster.specific.region,
                )

    async def terminate(
        self, cluster: Cluster[JarvisLabsSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[JarvisLabsSpecific]:
        if not instance_ids:
            return cluster

        async def _destroy(iid: str) -> None:
            try:
                await self._run(
                    self._client.instances.destroy,
                    machine_id=int(iid),
                )
                log.info("Destroyed instance {id}", id=iid)
            except Exception as e:
                log.error(
                    "Failed to destroy instance {id}: {err}",
                    id=iid, err=e,
                )

        await asyncio.gather(*(_destroy(iid) for iid in instance_ids))
        return cluster

    async def teardown(
        self, cluster: Cluster[JarvisLabsSpecific],
    ) -> Cluster[JarvisLabsSpecific]:
        log.info("Teardown for cluster {cid}", cid=cluster.id)
        close_fn = getattr(self._client, "close", None) or getattr(
            self._client, "__exit__", None,
        )
        if close_fn:
            try:
                if getattr(close_fn, "__self__", None):
                    self._client.__exit__(None, None, None)
                else:
                    close_fn()
            except Exception:
                pass
        self._pool.shutdown(wait=False)
        return cluster
