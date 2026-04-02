"""Novita.ai provider implementation.

Stateless provider using HTTP API with SSH key injection
via environment variables (same pattern as RunPod).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.core import PoolSpec
from skyward.core.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider

from .client import NovitaClient, NovitaError, get_api_key
from .config import Novita
from .types import InstanceResponse, ProductResponse, parse_ssh_command

log = logger.bind(provider="novita")

_DEFAULT_IMAGE = "nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04"


def _extract_cuda_version(image: str) -> str | None:
    """Extract CUDA major.minor from a Docker image name.

    ``"nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04"`` → ``"12.9"``.
    """
    import re
    match = re.search(r"cuda:(\d+\.\d+)", image)
    return match.group(1) if match else None


async def _cuda_image(major_minor: str) -> str | None:
    """Resolve a CUDA runtime image for a given major.minor version.

    Fetches the latest patch tag from Docker Hub for the requested
    major.minor. Returns ``None`` if no matching tag exists.
    """
    import httpx

    prefix = f"{major_minor}."

    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(
                "https://hub.docker.com/v2/repositories/nvidia/cuda/tags",
                params={"page_size": 25, "name": prefix},
            )
        if resp.status_code != 200:
            return None

        tags = [
            r["name"] for r in resp.json().get("results", [])
            if r["name"].startswith(prefix) and "runtime-ubuntu" in r["name"]
            and "cudnn" not in r["name"]
        ]
        if not tags:
            return None

        best = sorted(tags, reverse=True)[0]
        return f"nvcr.io/nvidia/cuda:{best}"
    except Exception:
        return None


def _get_cuda_max(offer: Offer) -> str | None:
    """Extract cuda.max from the offer's accelerator metadata."""
    accel = offer.instance_type.accelerator
    if not accel or not accel.metadata:
        return None
    return accel.metadata.get("cuda", {}).get("max")


def _cuda_candidates(cuda_max: str) -> list[str]:
    """Generate CUDA major.minor versions from cuda_max down to 11.8, descending."""
    major, minor = (int(x) for x in cuda_max.split("."))
    versions: list[str] = []
    while (major, minor) >= (11, 8):
        versions.append(f"{major}.{minor}")
        if minor > 0:
            minor -= 1
        else:
            major -= 1
            minor = 9
    return versions


_PRICE_DIVISOR = 100_000


def _price_to_usd(raw: str | int | None) -> float | None:
    """Convert Novita price units to USD/hour. Prices are strings in 1/100,000 USD."""
    match raw:
        case str() as s if s.strip():
            try:
                return float(s) / _PRICE_DIVISOR
            except ValueError:
                return None
        case int() as n:
            return n / _PRICE_DIVISOR
        case _:
            return None


def _resolve_accelerator(name: str) -> Accelerator:
    """Resolve accelerator from catalog, trying progressively shorter names.

    ``"H100 SXM"`` → tries ``"H100 SXM"`` then ``"H100"``.
    """
    candidates = [name]
    parts = name.split()
    if len(parts) > 1:
        candidates.append(parts[0] if not parts[0].startswith("RTX") else f"{parts[0]} {parts[1]}")
    for candidate in candidates:
        try:
            return Accelerator.from_name(candidate, count=1)
        except ValueError:
            continue
    return Accelerator(name=name, count=1)


def _normalize_gpu_name(raw: str) -> str:
    """Strip memory suffix from Novita GPU names to match accelerator catalog.

    ``"RTX 3090 24GB"`` → ``"RTX 3090"``, ``"A100 SXM 80GB"`` → ``"A100 SXM"``.
    """
    import re
    return re.sub(r"\s+\d+GB(\s+\(.*\))?$", "", raw).strip()


def _to_offer(product: ProductResponse, cluster_id: str) -> Offer:
    gpu_name = _normalize_gpu_name(product.get("name", ""))
    cpu_per_gpu = product.get("cpuPerGpu", 0)
    mem_per_gpu = product.get("memoryPerGpu", 0)
    on_demand_price = _price_to_usd(product.get("price"))
    spot_price = _price_to_usd(product.get("spotPrice"))

    accel = _resolve_accelerator(gpu_name) if gpu_name else None

    it = InstanceType(
        name=gpu_name or "unknown",
        accelerator=accel,
        vcpus=float(cpu_per_gpu),
        memory_gb=float(mem_per_gpu),
        architecture="x86_64",
        specific=None,
    )

    billing_methods = product.get("billingMethods", [])
    inventory_state = product.get("inventoryState", "")

    return Offer(
        id=f"novita-{product['id']}-{cluster_id}",
        instance_type=it,
        spot_price=spot_price,
        on_demand_price=on_demand_price,
        billing_unit="hour",
        specific={
            "product_id": product["id"],
            "cluster_id": cluster_id,
            "gpu_num": 1,
            "product_name": gpu_name,
            "inventory": inventory_state,
            "billing_methods": billing_methods,
        },
    )


def _extract_ssh(
    info: InstanceResponse,
) -> tuple[str | None, int, str | None]:
    ssh_comp = info.get("connectComponentSSH")
    if not ssh_comp:
        return None, 22, None

    password = ssh_comp.get("password") or info.get("sshPassword")

    ssh_cmd = ssh_comp.get("sshCommand")
    if not ssh_cmd:
        return None, 22, password

    try:
        host, port = parse_ssh_command(ssh_cmd)
        return host, port, password
    except ValueError:
        return None, 22, password


def _build_instance(
    info: InstanceResponse,
    status: InstanceStatus,
    offer: Offer,
    host: str | None,
    port: int,
    *,
    spot: bool,
    ssh_password: str | None = None,
) -> Instance:
    return Instance(
        id=info["id"],
        status=status,
        offer=offer,
        ip=host,
        private_ip=host,
        ssh_port=port,
        ssh_password=ssh_password,
        spot=spot,
    )


@dataclass(frozen=True, slots=True)
class NovitaSpecific:
    """Novita-specific cluster data flowing through Cluster[NovitaSpecific]."""

    product_id: str
    cluster_id: str
    gpu_num: int
    docker_image: str = _DEFAULT_IMAGE


class NovitaProvider(Provider[Novita, NovitaSpecific]):
    """Stateless Novita.ai provider. Holds only immutable config."""

    name = "novita"

    def __init__(self, config: Novita) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: Novita) -> NovitaProvider:
        return cls(config)

    async def offers(self) -> AsyncIterator[Offer]:
        api_key = get_api_key(self._config.api_key)

        async with NovitaClient(api_key, request_timeout=self._config.request_timeout) as client:
            products = await client.list_products(
                cluster_id=self._config.cluster_id,
            )

        for product in products:
            if product.get("availableDeploy") is False:
                continue
            match product.get("inventoryState", ""):
                case "none":
                    continue
                case _:
                    pass

            regions = product.get("regions", [])
            if regions:
                for region in regions:
                    yield _to_offer(product, region)
            else:
                yield _to_offer(product, self._config.cluster_id or "")

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[NovitaSpecific]:
        specific_data = offer.specific
        product_id = specific_data["product_id"]
        cluster_id = specific_data["cluster_id"]
        gpu_num = specific_data.get("gpu_num", 1)

        docker_image = str(self._config.docker_image) if self._config.docker_image else _DEFAULT_IMAGE

        return Cluster(
            id=f"novita-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path="",
            ssh_user="root",
            use_sudo=False,
            shutdown_command="kill 1",
            ssh_pty=False,
            specific=NovitaSpecific(
                product_id=product_id,
                cluster_id=cluster_id,
                gpu_num=gpu_num,
                docker_image=docker_image,
            ),
        )

    async def provision(
        self, cluster: Cluster[NovitaSpecific], count: int,
    ) -> tuple[Cluster[NovitaSpecific], Sequence[Instance]]:
        specific = cluster.specific
        api_key = get_api_key(self._config.api_key)

        use_spot = cluster.spec.allocation in ("spot", "spot-if-available")
        billing_mode = "spot" if use_spot else "onDemand"

        cuda_max = _get_cuda_max(cluster.offer)
        cuda_versions = _cuda_candidates(cuda_max) if cuda_max else [_extract_cuda_version(specific.docker_image) or "12.9"]

        instances: list[Instance] = []
        async with NovitaClient(api_key, request_timeout=self._config.request_timeout) as client:
            for i in range(count):
                label = f"skyward-{cluster.id}-{len(cluster.instances) + i}"

                instance_id = await self._create_with_cuda_fallback(
                    client,
                    product_id=specific.product_id,
                    name=label,
                    gpu_num=specific.gpu_num,
                    billing_mode=billing_mode,
                    cuda_versions=cuda_versions,
                    user_image=str(self._config.docker_image) if self._config.docker_image else None,
                )
                if instance_id is None:
                    continue

                instances.append(Instance(
                    id=instance_id,
                    status="provisioning",
                    offer=cluster.offer,
                    spot=use_spot,
                ))

        return cluster, instances

    async def _create_with_cuda_fallback(
        self,
        client: NovitaClient,
        *,
        product_id: str,
        name: str,
        gpu_num: int,
        billing_mode: str,
        cuda_versions: list[str],
        user_image: str | None,
    ) -> str | None:
        for cuda_ver in cuda_versions:
            image = user_image or await _cuda_image(cuda_ver)
            if not image:
                log.debug("No image for CUDA {cuda}, skipping", cuda=cuda_ver)
                continue
            min_cuda = self._config.min_cuda_version or cuda_ver

            try:
                instance_id = await client.create_instance(
                    product_id=product_id,
                    name=name,
                    image_url=image,
                    gpu_num=gpu_num,
                    rootfs_size=self._config.rootfs_size,
                    billing_mode=billing_mode,
                    cluster_id=self._config.cluster_id,
                    min_cuda_version=min_cuda,
                )
                log.info(
                    "Created instance {name} with CUDA {cuda}",
                    name=name, cuda=cuda_ver,
                )
                return instance_id
            except NovitaError as e:
                if "INSUFFICIENT_RESOURCE" in str(e):
                    log.debug(
                        "No hosts for CUDA {cuda}, trying lower",
                        cuda=cuda_ver,
                    )
                    continue
                log.error("Failed to create instance: {err}", err=e)
                return None

        log.error("Exhausted all CUDA versions, no hosts available")
        return None

    async def get_instance(
        self, cluster: Cluster[NovitaSpecific], instance_id: str,
    ) -> tuple[Cluster[NovitaSpecific], Instance | None]:
        api_key = get_api_key(self._config.api_key)
        async with NovitaClient(api_key, request_timeout=self._config.request_timeout) as client:
            info = await client.get_instance(instance_id)

        if not info:
            return cluster, None

        use_spot = cluster.spec.allocation in ("spot", "spot-if-available")

        match info.get("status"):
            case "removed" | "exited" | "terminated" | "error" | "failed":
                return cluster, None
            case "running":
                host, port, password = _extract_ssh(info)
                if host:
                    return cluster, _build_instance(
                        info, "provisioned", cluster.offer, host, port,
                        spot=use_spot, ssh_password=password,
                    )
                return cluster, _build_instance(
                    info, "provisioning", cluster.offer, None, 22,
                    spot=use_spot, ssh_password=password,
                )
            case _:
                return cluster, _build_instance(
                    info, "provisioning", cluster.offer, None, 22, spot=use_spot,
                )

    async def terminate(
        self, cluster: Cluster[NovitaSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[NovitaSpecific]:
        if not instance_ids:
            return cluster

        api_key = get_api_key(self._config.api_key)
        async with NovitaClient(api_key, request_timeout=self._config.request_timeout) as client:
            async def _delete(iid: str) -> None:
                try:
                    await client.delete_instance(iid)
                except Exception as e:
                    log.error("Failed to delete {iid}: {err}", iid=iid, err=e)

            await asyncio.gather(*(_delete(iid) for iid in instance_ids))
        return cluster

    async def teardown(self, cluster: Cluster[NovitaSpecific]) -> Cluster[NovitaSpecific]:
        return cluster
