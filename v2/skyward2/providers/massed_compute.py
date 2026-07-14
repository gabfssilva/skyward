import re
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

BASE_URL = "https://vm.massedcompute.com/api/v1"
INVENTORY_PATH = "/gpu-inventory"

PRODUCT_PATTERN = re.compile(r"^gpu_(\d+)x_(.+)$", re.IGNORECASE)
VARIANT_SUFFIXES = ("_spot", "_low_ram", "_high_ram")


class MassedComputeProvider:
    """Massed Compute — a fixed US fleet with a small, slow-moving product catalog.

    Prices are published per product and change on the order of weeks, but the
    per-product capacity counters move whenever someone rents a box, so the TTL
    is minutes rather than hours: the numbers a user sorts by are stable, the
    availability they act on is not.
    """

    kind: ClassVar[str] = "massed_compute"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=10)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("massed_compute requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
            response = await client.get(
                INVENTORY_PATH,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            inventory: dict[str, Any] = response.json().get("gpu_inventory", {})

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for product, item in inventory.items():
            instance_type = item["instance_type"]
            specs = instance_type["specs"]
            price = float(instance_type["price_cents_per_hour"]) / 100.0
            spot = _is_spot(product)
            accelerator, count = _gpu(product)
            capacity = int(item.get("capacity_available") or 0)
            regions = [region["name"] for region in item.get("regions_with_capacity_available", [])]

            for region in regions or [None]:
                yield Offer(
                    id=f"{product}:{region}" if region else product,
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    instance_type=product,
                    accelerator=accelerator,
                    accelerator_count=count,
                    cpus=int(specs["vcpu_count"]),
                    memory_gb=float(specs["memory_gib"]),
                    region=region,
                    disk_gb=float(specs["storage_gb"]),
                    spot_price=price if spot else None,
                    on_demand_price=None if spot else price,
                    available=capacity,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "product_name": product,
                        "description": instance_type.get("description"),
                        "spot": spot,
                        "regions_with_capacity": regions,
                        "price_cents_per_hour": instance_type["price_cents_per_hour"],
                    },
                )


def _is_spot(product: str) -> bool:
    return product.lower().endswith("_spot")


def _gpu(product: str) -> tuple[str | None, int]:
    """Pull the GPU token and its count out of a product id (``gpu_8x_a6000_spot``).

    A Massed Compute product id is not a GPU name — the count, the rental
    variant and the model are packed into one string, and unpacking that is
    provider knowledge. Naming the GPU is not: the token comes out raw
    (``h100_nvl``, ``a6000``) and the shared vocabulary canonicalizes it.
    """
    match = PRODUCT_PATTERN.match(product)
    if not match:
        return None, 0

    count = int(match.group(1))
    model = match.group(2).lower()

    changed = True
    while changed:
        changed = False
        for suffix in VARIANT_SUFFIXES:
            if model.endswith(suffix):
                model = model[: -len(suffix)]
                changed = True

    return model or None, count
