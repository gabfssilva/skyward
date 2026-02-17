from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from skyward.infra.cache import cached
from skyward.observability.logger import logger

log = logger.bind(component="aws-instances")

_GPU_MODEL_BY_MANUFACTURER: dict[str, dict[str, str]] = {
    "nvidia": {
        "t4": "T4",
        "t4g": "T4G",
        "a10g": "A10G",
        "a100": "A100",
        "h100": "H100",
        "h200": "H200",
        "l4": "L4",
        "l40s": "L40S",
        "v100": "V100",
        "k80": "K80",
        "m60": "M60",
    },
    "amd": {
        "radeon-pro-v520": "Radeon Pro V520",
    },
    "aws": {
        "trainium": "Trainium1",
        "trainium2": "Trainium2",
        "inferentia": "Inferentia1",
        "inferentia2": "Inferentia2",
    },
}


@dataclass(frozen=True, slots=True)
class InstanceResources:
    instance_type: str
    vcpus: int
    memory_mb: int
    architecture: str
    gpu_count: int = 0
    gpu_model: str = ""
    gpu_vram_mb: int = 0
    network_bandwidth_gbps: float = 0.0
    instance_storage_gb: int = 0
    instance_storage_type: str = ""

    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024

    @property
    def gpu_vram_gb(self) -> float:
        return self.gpu_vram_mb / 1024


def _parse_instance_type(raw: dict[str, Any]) -> InstanceResources:
    itype = raw["InstanceType"]
    vcpus = raw.get("VCpuInfo", {}).get("DefaultVCpus", 0)
    memory_mb = raw.get("MemoryInfo", {}).get("SizeInMiB", 0)

    archs = raw.get("ProcessorInfo", {}).get("SupportedArchitectures", [])
    architecture = archs[0] if archs else "x86_64"

    gpu_count = 0
    gpu_model = ""
    gpu_vram_mb = 0
    for gpu in raw.get("GpuInfo", {}).get("Gpus", []):
        gpu_count += gpu.get("Count", 0)
        manufacturer = gpu.get("Manufacturer", "").lower()
        name = gpu.get("Name", "").lower()
        gpu_model = _GPU_MODEL_BY_MANUFACTURER.get(
            manufacturer, {},
        ).get(name, gpu.get("Name", "") or "")
        gpu_vram_mb += gpu.get("MemoryInfo", {}).get("SizeInMiB", 0) * gpu.get("Count", 1)

    for acc in raw.get("InferenceAcceleratorInfo", {}).get("Accelerators", []):
        gpu_count += acc.get("Count", 0)
        manufacturer = acc.get("Manufacturer", "").lower()
        name = acc.get("Name", "").lower()
        gpu_model = _GPU_MODEL_BY_MANUFACTURER.get(
            manufacturer, {},
        ).get(name, acc.get("Name", "") or "")

    for acc in raw.get("NeuronInfo", {}).get("NeuronDevices", []):
        gpu_count += acc.get("Count", 0)
        name = acc.get("Name", "").lower()
        gpu_model = _GPU_MODEL_BY_MANUFACTURER.get("aws", {}).get(name, acc.get("Name", "") or "")
        core_count = acc.get("CoreInfo", {}).get("Count", 1)
        gpu_vram_mb += acc.get("MemoryInfo", {}).get("SizeInMiB", 0) * core_count

    net_info = raw.get("NetworkInfo", {})
    net_bw = net_info.get("NetworkPerformance", "")
    network_gbps = _parse_network_bandwidth(net_bw)

    storage_gb = 0
    storage_type = ""
    if raw.get("InstanceStorageSupported"):
        storage_info = raw.get("InstanceStorageInfo", {})
        storage_gb = storage_info.get("TotalSizeInGB", 0)
        storage_type = storage_info.get("NvmeSupport", "")

    return InstanceResources(
        instance_type=itype,
        vcpus=vcpus,
        memory_mb=memory_mb,
        architecture=architecture,
        gpu_count=gpu_count,
        gpu_model=gpu_model,
        gpu_vram_mb=gpu_vram_mb,
        network_bandwidth_gbps=network_gbps,
        instance_storage_gb=storage_gb,
        instance_storage_type=storage_type,
    )


def _parse_network_bandwidth(perf: str) -> float:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*Gigabit", perf, re.IGNORECASE)
    if match:
        return float(match.group(1))

    bandwidth_map = {
        "Low": 0.5,
        "Low to Moderate": 0.75,
        "Moderate": 1.0,
        "High": 10.0,
        "Very Low": 0.25,
    }
    return bandwidth_map.get(perf, 0.0)


@cached(namespace="aws-instance-types", ttl=timedelta(days=7))
async def _fetch_all_instance_types(
    region: str,
) -> dict[str, InstanceResources]:
    import aioboto3

    session = aioboto3.Session()
    results: dict[str, InstanceResources] = {}

    async with session.client("ec2", region_name=region) as client:  # type: ignore[reportGeneralTypeIssues]
        paginator = client.get_paginator("describe_instance_types")
        async for page in paginator.paginate():
            for raw in page.get("InstanceTypes", []):
                res = _parse_instance_type(raw)
                results[res.instance_type] = res

    log.info("Cached {n} instance types for {region}", n=len(results), region=region)
    return results


async def get_instance_resources(
    instance_type: str,
    region: str = "us-east-1",
) -> InstanceResources | None:
    index = await _fetch_all_instance_types(region)
    return index.get(instance_type)


async def list_gpu_instances(region: str = "us-east-1") -> list[InstanceResources]:
    index = await _fetch_all_instance_types(region)
    return [r for r in index.values() if r.gpu_count > 0]


async def list_instances_by_gpu(
    gpu_model: str, region: str = "us-east-1",
) -> list[InstanceResources]:
    index = await _fetch_all_instance_types(region)
    return [r for r in index.values() if r.gpu_model == gpu_model]


async def list_all_instance_types(region: str = "us-east-1") -> dict[str, InstanceResources]:
    return await _fetch_all_instance_types(region)


@dataclass(frozen=True, slots=True)
class InstanceSpec:
    instance_type: str
    region: str
    vcpus: int
    memory_gb: float
    architecture: str
    gpu_count: int = 0
    gpu_model: str = ""
    gpu_vram_gb: float = 0.0
    network_bandwidth_gbps: float = 0.0
    ondemand_price: float | None = None
    spot_price: float | None = None


@cached(namespace="aws-spot-prices", ttl=timedelta(minutes=15))
async def _fetch_spot_price(instance_type: str, region: str) -> float | None:
    import aioboto3

    session = aioboto3.Session()
    async with session.client("ec2", region_name=region) as ec2:  # type: ignore[reportGeneralTypeIssues]
        resp = await ec2.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            MaxResults=1,
        )
        history = resp.get("SpotPriceHistory", [])
        return float(history[0]["SpotPrice"]) if history else None


@cached(namespace="aws-ondemand-prices", ttl=timedelta(days=1))
async def _fetch_ondemand_price(instance_type: str, region: str) -> float | None:
    import json as json_mod

    import aioboto3

    session = aioboto3.Session()
    async with session.client("pricing", region_name="us-east-1") as pricing:
        resp = await pricing.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            ],
            MaxResults=1,
        )
        for raw in resp.get("PriceList", []):
            product = json_mod.loads(raw) if isinstance(raw, str) else raw
            terms = product.get("terms", {}).get("OnDemand", {})
            for term in terms.values():
                for dim in term.get("priceDimensions", {}).values():
                    price = dim.get("pricePerUnit", {}).get("USD")
                    if price:
                        return float(price)
        return None


async def get_instance_spec(
    instance_type: str,
    region: str = "us-east-1",
) -> InstanceSpec | None:
    import asyncio

    resources, spot, ondemand = await asyncio.gather(
        get_instance_resources(instance_type, region),
        _fetch_spot_price(instance_type, region),
        _fetch_ondemand_price(instance_type, region),
    )

    if resources is None:
        return None

    return InstanceSpec(
        instance_type=instance_type,
        region=region,
        vcpus=resources.vcpus,
        memory_gb=resources.memory_gb,
        architecture=resources.architecture,
        gpu_count=resources.gpu_count,
        gpu_model=resources.gpu_model,
        gpu_vram_gb=resources.gpu_vram_gb,
        network_bandwidth_gbps=resources.network_bandwidth_gbps,
        ondemand_price=ondemand,
        spot_price=spot,
    )
