import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, NamedTuple, Self

import aioboto3

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

DEFAULT_REGIONS = (
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "eu-west-1",
    "eu-central-1",
)

PRICING_REGION = "us-east-1"
MIB = 1024


class AWSProvider:
    """AWS EC2 — a catalog, an on-demand price list, and a live spot market.

    Three different APIs, because AWS keeps the three in three different places:
    ``ec2:DescribeInstanceTypes`` is the hardware, ``pricing:GetProducts`` is the
    on-demand rate card, and ``ec2:DescribeSpotPriceHistory`` is the market. The
    Pricing API only answers in ``us-east-1`` and is queried with a ``regionCode``
    filter, so it is one client for every region we list.

    The TTL is driven by the volatile third: on-demand rates move about as often as
    AWS issues a press release, but the spot price of a given instance type moves
    within the hour. Thirty minutes keeps the spot column honest without paying for
    a full catalog re-walk on every question.
    """

    kind: ClassVar[str] = "aws"
    credential_fields: ClassVar[tuple[str, ...]] = ("access_key_id", "secret_access_key")
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=30)

    def __init__(
        self,
        provider_id: str,
        name: str,
        access_key_id: str,
        secret_access_key: str,
        session_token: str | None,
        config: Mapping[str, Any],
    ) -> None:
        self._id = provider_id
        self._name = name
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        access_key_id = credentials.get("access_key_id")
        secret_access_key = credentials.get("secret_access_key")
        if not access_key_id or not secret_access_key:
            raise CapabilityMismatchError("aws requires access_key_id and secret_access_key credentials", provider=name)
        return cls(provider_id, name, access_key_id, secret_access_key, credentials.get("session_token") or None, config)

    @property
    def _regions(self) -> tuple[str, ...]:
        return tuple(self._config.get("regions") or DEFAULT_REGIONS)

    def _session(self) -> aioboto3.Session:
        """Always the explicit credentials — never boto3's default chain.

        The default chain would fall back to the caller's environment and
        ``~/.aws``, which is exactly what makes two AWS accounts in one process
        impossible.
        """
        return aioboto3.Session(
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
            aws_session_token=self._session_token,
        )

    async def offers(self) -> AsyncIterator[Offer]:
        regions = self._regions
        results = await asyncio.gather(*(self._fetch_region(region) for region in regions))

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for region, (instance_types, spot, on_demand) in zip(regions, results, strict=True):
            for raw in instance_types:
                instance_type = raw["InstanceType"]
                gpu = _gpu(raw)
                network = raw.get("NetworkInfo") or {}
                storage = raw.get("InstanceStorageInfo") or {}

                yield Offer(
                    id=f"aws-{region}-{instance_type}",
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    instance_type=instance_type,
                    accelerator=gpu.name,
                    accelerator_count=gpu.count,
                    vram=gpu.vram,
                    cpus=int((raw.get("VCpuInfo") or {}).get("DefaultVCpus") or 0),
                    memory_gb=float((raw.get("MemoryInfo") or {}).get("SizeInMiB") or 0) / MIB,
                    region=region,
                    disk_gb=float(storage.get("TotalSizeInGB")) if storage.get("TotalSizeInGB") else None,
                    spot_price=spot.get(instance_type),
                    on_demand_price=on_demand.get(instance_type),
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "architectures": (raw.get("ProcessorInfo") or {}).get("SupportedArchitectures"),
                        "hypervisor": raw.get("Hypervisor"),
                        "bare_metal": raw.get("BareMetal"),
                        "current_generation": raw.get("CurrentGeneration"),
                        "burstable": raw.get("BurstablePerformanceSupported"),
                        "ena_support": network.get("EnaSupport"),
                        "efa_supported": network.get("EfaSupported"),
                        "network_performance": network.get("NetworkPerformance"),
                        "instance_storage_supported": raw.get("InstanceStorageSupported"),
                        "instance_storage_nvme": storage.get("NvmeSupport"),
                        "instance_storage_disks": storage.get("Disks"),
                        "gpu_manufacturer": gpu.manufacturer,
                        "gpu_raw_name": gpu.name,
                        "supported_usage_classes": raw.get("SupportedUsageClasses"),
                        "supported_root_devices": raw.get("SupportedRootDeviceTypes"),
                        "supported_virtualization": raw.get("SupportedVirtualizationTypes"),
                    },
                )

    async def _fetch_region(self, region: str) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
        session = self._session()
        instance_types, spot, on_demand = await asyncio.gather(
            self._instance_types(session, region),
            self._spot_prices(session, region),
            self._on_demand_prices(session, region),
        )
        return instance_types, spot, on_demand

    async def _instance_types(self, session: aioboto3.Session, region: str) -> list[dict[str, Any]]:
        types: list[dict[str, Any]] = []
        async with session.client("ec2", region_name=region) as ec2:
            paginator = ec2.get_paginator("describe_instance_types")
            async for page in paginator.paginate():
                types.extend(page.get("InstanceTypes", []))
        return types

    async def _spot_prices(self, session: aioboto3.Session, region: str) -> dict[str, float]:
        """Cheapest current spot price per instance type, across the region's AZs.

        ``DescribeSpotPriceHistory`` with ``StartTime=now`` returns the price in
        force right now in each availability zone. We keep the minimum: the offer
        is a region, and a caller who asks for spot in that region can be placed
        in the AZ that quotes it.
        """
        prices: dict[str, float] = {}
        now = datetime.now(UTC)
        async with session.client("ec2", region_name=region) as ec2:
            paginator = ec2.get_paginator("describe_spot_price_history")
            async for page in paginator.paginate(ProductDescriptions=["Linux/UNIX"], StartTime=now, EndTime=now):
                for entry in page.get("SpotPriceHistory", []):
                    instance_type = entry["InstanceType"]
                    price = float(entry["SpotPrice"])
                    if price < prices.get(instance_type, float("inf")):
                        prices[instance_type] = price
        return prices

    async def _on_demand_prices(self, session: aioboto3.Session, region: str) -> dict[str, float]:
        """Linux/shared-tenancy on-demand rate card for one region, in one walk.

        The alternative — a ``GetProducts`` call per instance type, as v1 does — is
        roughly 900 round trips per region. Filtering by ``regionCode`` and paginating
        the whole rate card is the same data in a couple of dozen.
        """
        prices: dict[str, float] = {}
        filters = [
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            {"Type": "TERM_MATCH", "Field": "licenseModel", "Value": "No License required"},
            {"Type": "TERM_MATCH", "Field": "marketoption", "Value": "OnDemand"},
        ]
        async with session.client("pricing", region_name=PRICING_REGION) as pricing:
            paginator = pricing.get_paginator("get_products")
            async for page in paginator.paginate(ServiceCode="AmazonEC2", Filters=filters):
                for raw in page.get("PriceList", []):
                    product = json.loads(raw) if isinstance(raw, str) else raw
                    instance_type = (product.get("product", {}).get("attributes", {})).get("instanceType")
                    price = _hourly(product)
                    if instance_type and price:
                        prices[instance_type] = price
        return prices


class _Gpu(NamedTuple):
    name: str | None
    count: int
    vram: float | None
    manufacturer: str | None


def _gpu(raw: dict[str, Any]) -> _Gpu:
    """The instance's accelerators, as AWS spells them.

    ``GpuInfo.Gpus`` is a list because an instance could in principle mix models;
    none does today, so the first entry names the card and the counts add up.
    ``MemoryInfo.SizeInMiB`` there is already per card — it is passed through, and
    the shared catalog is left to normalize the name.
    """
    gpus = (raw.get("GpuInfo") or {}).get("Gpus") or []
    if not gpus:
        return _Gpu(None, 0, None, None)

    first = gpus[0]
    count = sum(int(gpu.get("Count") or 0) for gpu in gpus)
    vram = float((first.get("MemoryInfo") or {}).get("SizeInMiB") or 0) / MIB or None
    return _Gpu(first.get("Name") or None, count, vram, first.get("Manufacturer") or None)


def _hourly(product: dict[str, Any]) -> float | None:
    terms = (product.get("terms") or {}).get("OnDemand") or {}
    for term in terms.values():
        for dimension in (term.get("priceDimensions") or {}).values():
            price = (dimension.get("pricePerUnit") or {}).get("USD")
            if price and float(price) > 0:
                return float(price)
    return None
