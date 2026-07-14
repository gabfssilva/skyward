import re
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

BASE_URL = "https://infrahub-api.nexgencloud.com/v1"
FLAVORS_PATH = "/core/flavors"
PRICEBOOK_PATH = "/pricebook"

CPU_VCPU_RATE = "vCPU (cpu-only-flavors)"
CPU_RAM_RATE = "RAM (cpu-only-flavors)"

_DROPPED_TOKENS = frozenset({"pcie", "nvlink", "sxm", "sxm4", "sxm5", "ib", "se", "k8s", "sm", "n2"})
_MEMORY_TOKEN = re.compile(r"^\d+gb?$")


class HyperstackProvider:
    """Hyperstack (NexGen Cloud) — a fixed fleet of flavors with a published pricebook.

    Flavors and prices change on the order of weeks, not minutes, so the catalog is
    cached for an hour; stock is the only fast-moving dimension and it is exposed as
    a hint in ``specific`` rather than as a price.
    """

    kind: ClassVar[str] = "hyperstack"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(hours=1)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("hyperstack requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=30,
            headers={"api_key": self._api_key, "Accept": "application/json"},
        ) as client:
            flavors_response = await client.get(FLAVORS_PATH)
            flavors_response.raise_for_status()
            pricebook_response = await client.get(PRICEBOOK_PATH)
            pricebook_response.raise_for_status()

        groups = flavors_response.json().get("data", [])
        prices = _price_map(pricebook_response.json())

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for group in groups:
            for flavor in group.get("flavors", []):
                gpu = str(flavor.get("gpu") or "")
                gpu_count = int(flavor.get("gpu_count") or 0)
                cpus = int(flavor.get("cpu") or 0)
                memory_gb = float(flavor.get("ram") or 0)
                spot = gpu.lower().endswith("-spot")
                price = _hourly_price(gpu, gpu_count, cpus, memory_gb, prices)
                region = flavor.get("region_name") or group.get("region_name")
                features = flavor.get("features") or {}

                yield Offer(
                    id=str(flavor["id"]),
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    instance_type=str(flavor["name"]),
                    accelerator=_accelerator(gpu),
                    accelerator_count=gpu_count,
                    cpus=cpus,
                    memory_gb=memory_gb,
                    region=region,
                    disk_gb=float(flavor.get("disk") or 0),
                    spot_price=price if spot else None,
                    on_demand_price=None if spot else price,
                    available=None,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "flavor_name": flavor["name"],
                        "region": region,
                        "gpu": gpu or None,
                        "stock_available": bool(flavor.get("stock_available", True)),
                        "network_optimised": bool(features.get("network_optimised")),
                        "ephemeral_gb": flavor.get("ephemeral"),
                    },
                )


def _price_map(pricebook: Any) -> dict[str, float]:
    if not isinstance(pricebook, list):
        return {}
    prices: dict[str, float] = {}
    for entry in pricebook:
        name = entry.get("name")
        raw = entry.get("value")
        if not name or raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        prices[name.upper()] = value
    return prices


def _hourly_price(
    gpu: str,
    gpu_count: int,
    cpus: int,
    memory_gb: float,
    prices: dict[str, float],
) -> float | None:
    if gpu and gpu_count:
        per_gpu = prices.get(gpu.upper())
        if per_gpu is None or per_gpu <= 0:
            return None
        return round(per_gpu * gpu_count, 6)

    vcpu_rate = prices.get(CPU_VCPU_RATE.upper(), 0.0)
    ram_rate = prices.get(CPU_RAM_RATE.upper(), 0.0)
    total = vcpu_rate * cpus + ram_rate * memory_gb
    return round(total, 6) if total > 0 else None


def _accelerator(gpu: str) -> str | None:
    if not gpu:
        return None
    tokens = [
        token
        for token in gpu.lower().removesuffix("-spot").split("-")
        if token and token not in _DROPPED_TOKENS and not _MEMORY_TOKEN.match(token)
    ]
    return "-".join(tokens) or None
