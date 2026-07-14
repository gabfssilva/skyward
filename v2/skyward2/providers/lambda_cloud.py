import re
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

BASE_URL = "https://cloud.lambda.ai/api/v1"
INSTANCE_TYPES_PATH = "/instance-types"

_GPU_DESCRIPTION = re.compile(r"(?:Tesla\s+|NVIDIA\s+)?([\w.-]+)\s*\((\d+)\s*GB", re.IGNORECASE)


class LambdaProvider:
    """Lambda Cloud — a fixed fleet with a published price list, but scarce capacity.

    Its TTL is short not because prices move (they almost never do) but because
    ``available`` encodes ``regions_with_capacity_available``, which flips within
    minutes as instances are grabbed; the price half of the offer would happily
    cache for a day.
    """

    kind: ClassVar[str] = "lambda"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=5)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("lambda requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
            response = await client.get(
                INSTANCE_TYPES_PATH,
                auth=httpx.BasicAuth(self._api_key, ""),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            catalog: dict[str, Any] = response.json().get("data", {})

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for type_name, entry in catalog.items():
            info = entry["instance_type"]
            specs = info["specs"]
            accelerator, vram_gb = _accelerator(info.get("gpu_description"))
            regions = [region["name"] for region in entry.get("regions_with_capacity_available", [])]

            for region in regions or [None]:
                yield Offer(
                    id=f"{type_name}:{region}" if region else type_name,
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    instance_type=type_name,
                    accelerator=accelerator,
                    accelerator_count=int(specs.get("gpus") or 0),
                    cpus=int(specs.get("vcpus") or 0),
                    memory_gb=float(specs.get("memory_gib") or 0),
                    region=region,
                    disk_gb=float(specs.get("storage_gib") or 0),
                    spot_price=None,
                    on_demand_price=int(info["price_cents_per_hour"]) / 100.0,
                    available=1 if region else 0,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "instance_type_name": type_name,
                        "description": info.get("description"),
                        "gpu_description": info.get("gpu_description"),
                        "vram_gb": vram_gb,
                        "regions_with_capacity": regions,
                    },
                )


def _accelerator(gpu_description: str | None) -> tuple[str | None, int | None]:
    if not gpu_description or gpu_description == "N/A":
        return None, None
    if match := _GPU_DESCRIPTION.match(gpu_description):
        return match.group(1).lower(), int(match.group(2))
    return gpu_description.lower().replace(" ", "-"), None
