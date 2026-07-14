import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

BASE_URL = "https://api.scaleway.com"
PRODUCTS_PATH = "/instance/v1/zones/{zone}/products/servers"
AVAILABILITY_PATH = "/instance/v1/zones/{zone}/products/servers/availability"

DEFAULT_ZONES = (
    "fr-par-1",
    "fr-par-2",
    "fr-par-3",
    "nl-ams-1",
    "nl-ams-2",
    "nl-ams-3",
    "pl-waw-1",
    "pl-waw-2",
    "pl-waw-3",
)

GIB = 1024**3
GB = 1_000_000_000


class ScalewayProvider:
    """Scaleway — a fixed fleet, but the offer carries a per-zone availability flag.

    The catalog itself (server types, hourly prices) barely moves, so the TTL is
    driven by the volatile half of the offer: a GPU type flips between
    ``available``, ``scarce`` and ``shortage`` within a zone over minutes, and
    that flag is what ``available`` encodes — hence ten minutes, not hours.
    """

    kind: ClassVar[str] = "scaleway"
    credential_fields: ClassVar[tuple[str, ...]] = ("secret_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=10)

    def __init__(self, provider_id: str, name: str, secret_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._secret_key = secret_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        secret_key = credentials.get("secret_key")
        if not secret_key:
            raise CapabilityMismatchError("scaleway requires a secret_key credential", provider=name)
        return cls(provider_id, name, secret_key, config)

    @property
    def _zones(self) -> tuple[str, ...]:
        return tuple(self._config.get("zones") or DEFAULT_ZONES)

    async def _fetch_zone(self, client: httpx.AsyncClient, zone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        products, availability = await asyncio.gather(
            client.get(PRODUCTS_PATH.format(zone=zone)),
            client.get(AVAILABILITY_PATH.format(zone=zone)),
        )
        products.raise_for_status()
        availability.raise_for_status()
        return products.json().get("servers", {}), availability.json().get("servers", {})

    async def offers(self) -> AsyncIterator[Offer]:
        zones = self._zones
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=30,
            headers={"X-Auth-Token": self._secret_key, "Accept": "application/json"},
        ) as client:
            results = await asyncio.gather(*(self._fetch_zone(client, zone) for zone in zones))

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for zone, (server_types, availability) in zip(zones, results, strict=True):
            for commercial_type, spec in server_types.items():
                gpu_info = spec.get("gpu_info") or {}
                gpu_name = gpu_info.get("gpu_name")
                gpu_count = int(spec.get("gpu") or 0)
                gpu_memory_gb = float(gpu_info.get("gpu_memory") or 0) / GIB or None
                state = (availability.get(commercial_type) or {}).get("availability")
                hourly = spec.get("hourly_price")

                yield Offer(
                    id=f"scaleway-{zone}-{commercial_type}",
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    instance_type=commercial_type,
                    accelerator=gpu_name if gpu_count else None,
                    accelerator_count=gpu_count,
                    vram=gpu_memory_gb if gpu_count else None,
                    cpus=int(spec.get("ncpus") or 0),
                    memory_gb=float(spec.get("ram") or 0) / GIB,
                    region=zone,
                    disk_gb=_disk_gb(spec),
                    spot_price=None,
                    on_demand_price=float(hourly) if hourly else None,
                    available=_available(state),
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "commercial_type": commercial_type,
                        "zone": zone,
                        "arch": spec.get("arch"),
                        "availability": state,
                        "gpu_name": gpu_name,
                        "gpu_memory_gb": gpu_memory_gb,
                        "monthly_price": spec.get("monthly_price"),
                        "end_of_service": spec.get("end_of_service"),
                        "boot_types": (spec.get("capabilities") or {}).get("boot_types"),
                        "block_storage": (spec.get("capabilities") or {}).get("block_storage"),
                        "scratch_storage_max_size": spec.get("scratch_storage_max_size"),
                    },
                )


def _disk_gb(spec: dict[str, Any]) -> float | None:
    local = (spec.get("volumes_constraint") or {}).get("max_size") or 0
    scratch = spec.get("scratch_storage_max_size") or 0
    size = local or scratch
    return float(size) / GB if size else None


def _available(state: str | None) -> int | None:
    match state:
        case "available" | "scarce":
            return 1
        case "shortage":
            return 0
        case _:
            return None
