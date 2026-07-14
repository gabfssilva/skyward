import re
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

BASE_URL = "https://api.vultr.com/v2"
PLANS_PATH = "/plans"
METAL_PLANS_PATH = "/plans-metal"
PAGE_SIZE = 500

VRAM_SUFFIX = re.compile(r"(\d+)vram$")


class VultrProvider:
    """Vultr — a fixed fleet with a published plan list.

    Its offers TTL is hours, not minutes: the catalog is a price list that changes
    when Vultr ships a new plan, not a live market like Vast.ai. The endpoints carry
    no per-region availability at all, so there is nothing here that goes stale fast.
    """

    kind: ClassVar[str] = "vultr"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(hours=6)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("vultr requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def _pages(self, client: httpx.AsyncClient, path: str, key: str) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"per_page": PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            response = await client.get(path, params=params)
            response.raise_for_status()
            payload = response.json()
            for plan in payload.get(key, []):
                yield plan
            cursor = payload.get("meta", {}).get("links", {}).get("next") or None
            if not cursor:
                return

    async def offers(self) -> AsyncIterator[Offer]:
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}
        async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=30) as client:
            now = datetime.now(UTC)
            expires_at = now + self.offers_ttl

            async for plan in self._pages(client, PLANS_PATH, "plans"):
                for offer in self._offers(plan, cpus=int(plan.get("vcpu_count") or 0), bare_metal=False, now=now, expires_at=expires_at):
                    yield offer

            async for plan in self._pages(client, METAL_PLANS_PATH, "plans_metal"):
                for offer in self._offers(plan, cpus=int(plan.get("cpu_count") or 0), bare_metal=True, now=now, expires_at=expires_at):
                    yield offer

    def _offers(
        self,
        plan: dict[str, Any],
        *,
        cpus: int,
        bare_metal: bool,
        now: datetime,
        expires_at: datetime,
    ) -> Iterator[Offer]:
        plan_id = str(plan["id"])
        gpu_type = plan.get("gpu_type") or None
        whole, fraction = _accelerator_count(plan, gpu_type)
        total_vram = float(plan.get("gpu_vram_gb") or 0) or float(_vram_from_id(plan_id))
        disks = max(int(plan.get("disk_count") or 1), 1)
        preemptible = bool(plan.get("deploy_preemptible"))
        locations: list[str | None] = list(plan.get("locations") or []) or [None]

        specific = {
            "plan_type": plan.get("type"),
            "bare_metal": bare_metal,
            "gpu_brand": plan.get("gpu_brand"),
            "gpu_type": plan.get("gpu_type"),
            "gpu_vram_gb": plan.get("gpu_vram_gb") or _vram_from_id(plan_id) or None,
            "gpu_fraction": fraction,
            "bandwidth_gb": plan.get("bandwidth"),
            "disk_type": plan.get("disk_type") or (plan.get("type") if bare_metal else None),
            "disk_count": disks,
            "monthly_cost": plan.get("monthly_cost"),
            "deploy_ondemand": bool(plan.get("deploy_ondemand", True)),
            "deploy_preemptible": preemptible,
        }

        for region in locations:
            cost = _cost(plan, region)
            yield Offer(
                id=f"vultr-{plan_id}-{region or 'any'}",
                provider_id=self._id,
                provider_name=self._name,
                kind=self.kind,
                instance_type=plan_id,
                accelerator=gpu_type,
                accelerator_count=whole,
                vram=_vram_per_card(total_vram, whole),
                cpus=cpus,
                memory_gb=float(plan.get("ram") or 0) / 1024,
                region=region,
                disk_gb=float(plan.get("disk") or 0) * disks,
                spot_price=cost.get("hourly_cost_preemptible") if preemptible else None,
                on_demand_price=cost.get("hourly_cost"),
                fetched_at=now,
                expires_at=expires_at,
                specific=specific,
            )


def _cost(plan: dict[str, Any], region: str | None) -> dict[str, Any]:
    override = (plan.get("location_cost") or {}).get(region) if region else None
    return override or plan


def _vram_per_card(total_vram: float, whole: int) -> float | None:
    """Vultr reports ``gpu_vram_gb`` as the total across the plan's cards.

    An 8x H100 plan says 640; a card has 80. A sliced plan has no whole card to
    divide by and its VRAM already is the slice, so it passes through as it is.
    """
    if not total_vram:
        return None
    return total_vram / whole if whole else total_vram


def _vram_from_id(plan_id: str) -> int:
    match = VRAM_SUFFIX.search(plan_id)
    return int(match.group(1)) if match else 0


def _accelerator_count(plan: dict[str, Any], accelerator: str | None) -> tuple[int, float | None]:
    """Whole GPUs attached, plus the fraction when the plan is a GPU slice.

    Vultr reports ``gpu_count`` as ``"1/8"`` on sliced Cloud GPU plans. A slice gets
    ``0`` whole GPUs — it must not compete with a plan that really hands over a
    device — and carries its share in ``specific``.
    """
    if not accelerator:
        return 0, None

    match plan.get("gpu_count"):
        case int(count):
            return count, float(count)
        case str(count) if "/" in count:
            numerator, _, denominator = count.partition("/")
            return 0, int(numerator) / int(denominator)
        case str(count) if count.isdigit():
            return int(count), float(count)

    return 1, 1.0
