from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

BASE_URL = "https://console.vast.ai"
SEARCH_PATH = "/api/v0/bundles/"


class VastAIProvider:
    """Vast.ai — a marketplace, so its catalog moves by the minute.

    Its offers TTL is short for that reason: a bundle that was there five
    minutes ago is often gone. Providers with a fixed fleet (AWS) can cache for
    hours; that is exactly why the TTL belongs to the provider and not to the
    cache.
    """

    kind: ClassVar[str] = "vastai"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=2)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("vastai requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    def _query(self) -> dict[str, Any]:
        return {
            "verified": {"eq": bool(self._config.get("verified_only", False))},
            "rentable": {"eq": True},
            "reliability2": {"gte": float(self._config.get("min_reliability", 0.9))},
            "num_gpus": {"gte": 1},
            "order": [["score", "desc"]],
            "type": "on-demand",
            "limit": int(self._config.get("limit", 500)),
        }

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
            response = await client.post(
                SEARCH_PATH,
                json=self._query(),
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            bundles = response.json().get("offers", [])

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for bundle in bundles:
            gpus = int(bundle.get("num_gpus") or 0)
            on_demand = bundle.get("dph_total")
            gpu_name = bundle.get("gpu_name")
            gpu_ram = bundle.get("gpu_ram")
            yield Offer(
                id=str(bundle["id"]),
                provider_id=self._id,
                provider_name=self._name,
                kind=self.kind,
                instance_type=str(gpu_name or "cpu"),
                accelerator=gpu_name,
                accelerator_count=gpus,
                vram=float(gpu_ram) / 1024 if gpu_ram else None,
                cpus=int(bundle.get("cpu_cores_effective") or 0),
                memory_gb=float(bundle.get("cpu_ram") or 0) / 1024,
                region=bundle.get("geolocation"),
                disk_gb=float(bundle.get("disk_space") or 0),
                spot_price=bundle.get("min_bid"),
                on_demand_price=float(on_demand) if on_demand is not None else None,
                available=gpus,
                fetched_at=now,
                expires_at=expires_at,
                specific={
                    "gpu_name": gpu_name,
                    "machine_id": bundle.get("machine_id"),
                    "cuda_max_good": bundle.get("cuda_max_good"),
                    "reliability": bundle.get("reliability2"),
                    "direct_port_count": bundle.get("direct_port_count"),
                },
            )
