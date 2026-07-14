import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

GRAPHQL_URL = "https://api.runpod.io/graphql"

GPU_TYPES_QUERY = """
query GpuTypes($secureCloud: Boolean) {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    maxGpuCount
    maxGpuCountSecureCloud
    maxGpuCountCommunityCloud
    lowestPrice(input: {gpuCount: 1, secureCloud: $secureCloud}) {
      minimumBidPrice
      uninterruptablePrice
      stockStatus
      totalCount
      rentedCount
      minVcpu
      minMemory
    }
  }
}
"""

CLOUDS: tuple[tuple[str, bool], ...] = (("SECURE", True), ("COMMUNITY", False))


class RunPodProvider:
    """RunPod — a two-tier GPU cloud: SECURE (datacenter fleet) and COMMUNITY (peer hosts).

    Its catalog is a small fixed list of GPU types, but ``lowestPrice`` tracks live
    host supply and interruptible bids, so the TTL is minutes rather than hours.
    """

    kind: ClassVar[str] = "runpod"
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
            raise CapabilityMismatchError("runpod requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def _gpu_types(self, client: httpx.AsyncClient, secure: bool) -> list[dict[str, Any]]:
        response = await client.post(
            GRAPHQL_URL,
            json={"query": GPU_TYPES_QUERY, "variables": {"secureCloud": secure}},
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        payload = response.json()
        if errors := payload.get("errors"):
            raise RuntimeError(f"runpod graphql error: {errors}")
        return payload.get("data", {}).get("gpuTypes") or []

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(timeout=30) as client:
            catalogs = await asyncio.gather(*(self._gpu_types(client, secure) for _, secure in CLOUDS))

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for (cloud, secure), gpu_types in zip(CLOUDS, catalogs, strict=True):
            for gpu in gpu_types:
                if not gpu.get("secureCloud" if secure else "communityCloud"):
                    continue

                price = gpu.get("lowestPrice") or {}
                bid = price.get("minimumBidPrice")
                on_demand = price.get("uninterruptablePrice")
                if bid is None and on_demand is None:
                    continue

                gpu_id = str(gpu["id"])
                display = str(gpu.get("displayName") or gpu_id)
                vram_gb = int(gpu.get("memoryInGb") or 0)
                vcpus = int(price.get("minVcpu") or 0)
                memory_gb = float(price.get("minMemory") or 0)
                total = price.get("totalCount")
                rented = int(price.get("rentedCount") or 0)
                available = max(int(total) - rented, 0) if total else None

                max_count = int(
                    gpu.get("maxGpuCountSecureCloud" if secure else "maxGpuCountCommunityCloud")
                    or gpu.get("maxGpuCount")
                    or 1
                )

                for count in range(1, max_count + 1):
                    yield Offer(
                        id=f"{gpu_id}:{cloud}:{count}x",
                        provider_id=self._id,
                        provider_name=self._name,
                        kind=self.kind,
                        instance_type=f"{gpu_id}:{cloud}",
                        accelerator=display,
                        accelerator_count=count,
                        vram=float(vram_gb) or None,
                        cpus=vcpus * count,
                        memory_gb=memory_gb * count,
                        spot_price=round(bid * count, 4) if bid is not None else None,
                        on_demand_price=round(on_demand * count, 4) if on_demand is not None else None,
                        available=available,
                        fetched_at=now,
                        expires_at=expires_at,
                        specific={
                            "gpu_type_id": gpu_id,
                            "cloud_type": cloud,
                            "gpu_display_name": display,
                            "gpu_memory_gb": vram_gb,
                            "stock_status": price.get("stockStatus"),
                        },
                    )
