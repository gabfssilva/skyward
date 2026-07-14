from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

BASE_URL = "https://dashboard.tensordock.com"
LOCATIONS_PATH = "/api/v2/locations"

DEFAULT_STORAGE_GB = 100


class TensorDockProvider:
    """TensorDock — a marketplace of independent hosts, so its catalog is volatile.

    Hosts join, leave and get rented out by other tenants without notice, so the
    TTL is short (5 minutes): the GPU count a location advertises is a snapshot
    of somebody else's fleet, not a reservation.

    Pricing is à la carte — TensorDock bills a per-GPU rate plus per-vCPU,
    per-GB-RAM and per-GB-storage rates. An Offer here is the *whole* available
    slice of one GPU model at one location: all ``max_count`` GPUs, the maximum
    vCPU and RAM a VM of that shape may claim, and ``storage_gb`` of disk. Its
    ``on_demand_price`` is the sum of those four line items, so the price, the
    ``accelerator_count``, the ``cpus`` and the ``memory_gb`` of an Offer all
    describe the same single machine.
    """

    kind: ClassVar[str] = "tensordock"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_token",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=5)

    def __init__(self, provider_id: str, name: str, api_token: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_token = api_token
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_token = credentials.get("api_token")
        if not api_token:
            raise CapabilityMismatchError("tensordock requires an api_token credential", provider=name)
        return cls(provider_id, name, api_token, config)

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
            response = await client.get(
                LOCATIONS_PATH,
                headers={"Authorization": f"Bearer {self._api_token}", "Accept": "application/json"},
            )
            response.raise_for_status()
            locations = response.json().get("data", {}).get("locations", [])

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl
        storage_gb = int(self._config.get("storage_gb", DEFAULT_STORAGE_GB))

        for location in locations:
            for gpu in location.get("gpus", []):
                model = str(gpu.get("v0Name") or "")
                count = int(gpu.get("max_count") or 0)
                if not model or count < 1:
                    continue

                resources = gpu.get("resources") or {}
                pricing = gpu.get("pricing") or {}
                cpus = int(resources.get("max_vcpus") or 0)
                memory_gb = float(resources.get("max_ram_gb") or 0)
                disk_gb = float(min(storage_gb, int(resources.get("max_storage_gb") or storage_gb)))

                price = (
                    count * float(gpu.get("price_per_hr") or 0)
                    + cpus * float(pricing.get("per_vcpu_hr") or 0)
                    + memory_gb * float(pricing.get("per_gb_ram_hr") or 0)
                    + disk_gb * float(pricing.get("per_gb_storage_hr") or 0)
                )

                yield Offer(
                    id=f"{location['id']}:{model}",
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    instance_type=str(gpu.get("displayName") or model),
                    accelerator=_accelerator(model),
                    accelerator_count=count,
                    cpus=cpus,
                    memory_gb=memory_gb,
                    region=_region(location),
                    disk_gb=disk_gb,
                    spot_price=None,
                    on_demand_price=price,
                    available=count,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "location_id": location["id"],
                        "gpu_model": model,
                        "tier": location.get("tier"),
                        "price_per_gpu_hr": gpu.get("price_per_hr"),
                        "per_vcpu_hr": pricing.get("per_vcpu_hr"),
                        "per_gb_ram_hr": pricing.get("per_gb_ram_hr"),
                        "per_gb_storage_hr": pricing.get("per_gb_storage_hr"),
                        "max_vcpus": resources.get("max_vcpus"),
                        "max_ram_gb": resources.get("max_ram_gb"),
                        "max_storage_gb": resources.get("max_storage_gb"),
                        "network_features": gpu.get("network_features"),
                    },
                )


_MODELS = (
    ("h100", "h100"),
    ("h200", "h200"),
    ("a100", "a100"),
    ("l40s", "l40s"),
    ("l40", "l40"),
    ("v100", "v100"),
)


def _accelerator(model: str) -> str | None:
    """Normalize a TensorDock GPU id like ``geforcertx4090-pcie-24gb`` to ``rtx-4090``."""
    if not model:
        return None
    head = model.split("-", 1)[0].lower()
    for needle, name in _MODELS:
        if needle in head:
            return name
    if "rtx" in head:
        return f"rtx-{head.split('rtx', 1)[1]}"
    return head


def _region(location: Mapping[str, Any]) -> str | None:
    parts = [location.get("city"), location.get("stateprovince"), location.get("country")]
    return ", ".join(p for p in parts if p) or None
