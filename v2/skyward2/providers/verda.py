import re
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

BASE_URL = "https://api.verda.com/v1"
TOKEN_PATH = "/oauth2/token"
INSTANCE_TYPES_PATH = "/instance-types"
AVAILABILITY_PATH = "/instance-availability"

GPU_ALIASES = {"tesla-v100": "v100"}


class VerdaProvider:
    """Verda — a fixed fleet with a published price list, so the catalog is stable.

    Prices only change when Verda changes them; what actually moves is regional
    availability, which is why the TTL is 15 minutes rather than hours or minutes.
    """

    kind: ClassVar[str] = "verda"
    credential_fields: ClassVar[tuple[str, ...]] = ("client_id", "client_secret")
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=15)

    def __init__(
        self,
        provider_id: str,
        name: str,
        client_id: str,
        client_secret: str,
        config: Mapping[str, Any],
    ) -> None:
        self._id = provider_id
        self._name = name
        self._client_id = client_id
        self._client_secret = client_secret
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        client_id = credentials.get("client_id")
        client_secret = credentials.get("client_secret")
        if not client_id or not client_secret:
            raise CapabilityMismatchError(
                "verda requires client_id and client_secret credentials", provider=name
            )
        return cls(provider_id, name, client_id, client_secret, config)

    async def _token(self, client: httpx.AsyncClient) -> str:
        response = await client.post(
            TOKEN_PATH,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        response.raise_for_status()
        return response.json()["access_token"]

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
            headers = {"Authorization": f"Bearer {await self._token(client)}"}

            types_response = await client.get(INSTANCE_TYPES_PATH, headers=headers)
            types_response.raise_for_status()
            instance_types = types_response.json() or []

            on_demand_regions = await _availability(client, headers, is_spot=False)
            spot_regions = await _availability(client, headers, is_spot=True)

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for entry in instance_types:
            name = str(entry["instance_type"])
            gpu = entry.get("gpu") or {}
            gpu_count = int(gpu.get("number_of_gpus") or 0)
            total_vram = float((entry.get("gpu_memory") or {}).get("size_in_gigabytes") or 0)
            on_demand_price = _price(entry.get("price_per_hour"))
            spot_price = _price(entry.get("spot_price"))

            regions = on_demand_regions.get(name, frozenset()) | spot_regions.get(name, frozenset())
            for region in sorted(regions) or [None]:
                spot_here = spot_price if region is None or region in spot_regions.get(name, frozenset()) else None
                on_demand_here = (
                    on_demand_price
                    if region is None or region in on_demand_regions.get(name, frozenset())
                    else None
                )
                yield Offer(
                    id=f"{name}:{region}" if region else name,
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    instance_type=name,
                    accelerator=_accelerator(gpu.get("description")),
                    accelerator_count=gpu_count,
                    cpus=int((entry.get("cpu") or {}).get("number_of_cores") or 0),
                    memory_gb=float((entry.get("memory") or {}).get("size_in_gigabytes") or 0),
                    region=region,
                    disk_gb=None,
                    spot_price=spot_here,
                    on_demand_price=on_demand_here,
                    available=None,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "gpu_description": gpu.get("description"),
                        "gpu_memory_gb_per_card": total_vram / gpu_count if gpu_count else None,
                        "storage": (entry.get("storage") or {}).get("description"),
                        "supported_os": entry.get("supported_os", []),
                        "location_code": region,
                        "spot_available": region in spot_regions.get(name, frozenset()),
                    },
                )


async def _availability(
    client: httpx.AsyncClient, headers: Mapping[str, str], is_spot: bool
) -> dict[str, frozenset[str]]:
    response = await client.get(
        AVAILABILITY_PATH, headers=dict(headers), params={"is_spot": str(is_spot).lower()}
    )
    response.raise_for_status()
    by_type: dict[str, set[str]] = {}
    for region in response.json() or []:
        code = region["location_code"]
        for instance_type in region.get("availabilities", []):
            by_type.setdefault(instance_type, set()).add(code)
    return {k: frozenset(v) for k, v in by_type.items()}


def _price(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        return float(str(raw))
    except ValueError:
        return None


def _accelerator(description: str | None) -> str | None:
    if not description:
        return None
    match = re.match(r"^\d+x\s+(.+?)(?:\s+\d+GB)?$", description)
    if not match:
        return None
    model = re.sub(r"\s+SXM\d+", "", match.group(1)).lower().replace(" ", "-")
    return GPU_ALIASES.get(model, model)
