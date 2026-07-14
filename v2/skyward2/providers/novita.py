import re
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

BASE_URL = "https://api.novita.ai/gpu-instance/openapi/v1"
PRODUCTS_PATH = "/products"

PRICE_DIVISOR = 100_000
MEMORY_SUFFIX = re.compile(r"\s+\d+GB\b")
QUALIFIER = re.compile(r"\s*\(.*\)\s*$")
VENDOR_PREFIX = re.compile(r"^((?i:nvidia|geforce)\s+)+")


class NovitaProvider:
    """Novita.ai — a fixed GPU fleet with a small, slow-moving product catalog.

    The product list itself barely changes; only ``inventoryState`` and the spot
    price move, so a 15 minute TTL keeps availability roughly honest without
    re-querying for every question.
    """

    kind: ClassVar[str] = "novita"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=15)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("novita requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def offers(self) -> AsyncIterator[Offer]:
        params: dict[str, Any] = {}
        if cluster_id := self._config.get("cluster_id"):
            params["clusterId"] = cluster_id

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
            response = await client.get(
                PRODUCTS_PATH,
                params=params,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            products = response.json().get("data", [])

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for product in products:
            regions = product.get("regions") or [None]
            for region in regions:
                yield self._to_offer(product, region, now, expires_at)

    def _to_offer(
        self,
        product: dict[str, Any],
        region: str | None,
        now: datetime,
        expires_at: datetime,
    ) -> Offer:
        raw_name = str(product.get("name") or "unknown")
        billing = product.get("billingMethods") or []
        spot = _price(product.get("spotPrice")) if "spot" in billing else None
        return Offer(
            id=f"{product['id']}:{region or 'any'}",
            provider_id=self._id,
            provider_name=self._name,
            kind=self.kind,
            instance_type=raw_name,
            accelerator=_accelerator(raw_name),
            accelerator_count=1,
            cpus=int(product.get("cpuPerGpu") or 0),
            memory_gb=float(product.get("memoryPerGpu") or 0),
            region=region,
            disk_gb=float(product.get("diskPerGpu") or 0),
            spot_price=spot,
            on_demand_price=_price(product.get("price")),
            available=_available(product),
            fetched_at=now,
            expires_at=expires_at,
            specific={
                "product_id": str(product["id"]),
                "cluster_name": region,
                "billing_methods": billing,
                "inventory_state": product.get("inventoryState"),
                "available_deploy": product.get("availableDeploy"),
                "min_rootfs_gb": product.get("minRootFS"),
                "max_rootfs_gb": product.get("maxRootFS"),
            },
        )


def _price(raw: str | int | float | None) -> float | None:
    """Novita quotes prices as integer strings in 1/100,000 USD per hour."""
    match raw:
        case str() as s if s.strip():
            try:
                return float(s) / PRICE_DIVISOR
            except ValueError:
                return None
        case int() | float() as n:
            return n / PRICE_DIVISOR
        case _:
            return None


def _available(product: dict[str, Any]) -> int | None:
    if product.get("availableDeploy") is False:
        return 0
    match product.get("inventoryState"):
        case "none":
            return 0
        case _:
            return None


def _accelerator(raw: str) -> str | None:
    name = QUALIFIER.sub("", raw)
    name = MEMORY_SUFFIX.sub("", name).strip()
    name = VENDOR_PREFIX.sub("", name)
    if not name:
        return None
    return name.lower().replace(" ", "-")
