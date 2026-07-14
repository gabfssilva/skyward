import re
from collections.abc import AsyncIterator, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

BASE_URL = "https://backendprod.jarvislabs.net"
SERVER_META_PATH = "/misc/server_meta"

EUROPE_REGION = "europe-01"
EUROPE_GPU_COUNTS = (1, 8)

_MEMORY_SUFFIX = re.compile(r"[-_]?\d+GB$", re.IGNORECASE)


class JarvisLabsProvider:
    """Jarvis Labs — an owned fleet, so the catalog is a price list, not a market.

    Its offers TTL is fifteen minutes: prices are set by Jarvis Labs and barely
    move, and only the free-device counts drift as instances come and go.

    The regional backends each mirror the whole fleet, so one call to the India
    backend returns Europe and Chennai too — there is no per-region fan-out.
    """

    kind: ClassVar[str] = "jarvislabs"
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
            raise CapabilityMismatchError("jarvislabs requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
            response = await client.get(
                SERVER_META_PATH,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            entries = response.json().get("server_meta", [])

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for entry in entries:
            gpu_type = str(entry.get("gpu_type") or "")
            if not gpu_type:
                continue

            region = str(entry.get("region") or "")
            workload = entry.get("workload_type")
            unit_price = entry.get("price_per_hour")
            unit_spot = entry.get("spot_price")
            cpus_per_gpu = int(entry.get("cpus_per_gpu") or 0)
            ram_per_gpu = float(entry.get("ram_per_gpu") or 0)
            free = int(entry.get("effective_num_free_devices") or entry.get("num_free_devices") or 0)
            vram = entry.get("vram")

            for count in _gpu_counts(entry, region):
                yield Offer(
                    id=f"{region}:{workload or 'default'}:{gpu_type}:{count}",
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    instance_type=f"{gpu_type}x{count}",
                    accelerator=_accelerator(gpu_type),
                    accelerator_count=count,
                    cpus=cpus_per_gpu * count,
                    memory_gb=ram_per_gpu * count,
                    region=region,
                    disk_gb=None,
                    spot_price=float(unit_spot) * count if unit_spot is not None else None,
                    on_demand_price=float(unit_price) * count if unit_price is not None else None,
                    available=free // count,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "gpu_type": gpu_type,
                        "num_gpus": count,
                        "region": region,
                        "workload_type": workload,
                        "vram_gb": int(vram) if vram else None,
                        "architecture": entry.get("arc"),
                        "spot_only_server": entry.get("spot_only_server"),
                        "reserved_pricing": entry.get("reserved_pricing"),
                    },
                )


def _gpu_counts(entry: Mapping[str, Any], region: str) -> Iterable[int]:
    """GPU counts a single machine of this type can be rented with.

    Jarvis Labs picks the count at provision time, so one catalog entry becomes
    one offer per rentable count. Europe machines only accept 1 or 8 GPUs; the
    other regions expose their per-machine maximum in ``num_gpus``.
    """
    if region == EUROPE_REGION:
        return EUROPE_GPU_COUNTS
    max_gpus = int(entry.get("num_gpus") or 1)
    return range(1, max_gpus + 1)


def _accelerator(gpu_type: str) -> str:
    return _MEMORY_SUFFIX.sub("", gpu_type).lower().replace("_", "-")
