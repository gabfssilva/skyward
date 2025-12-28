"""Cloud instance pricing data with daily cache.

Fetches pricing from Vantage (AWS, Azure, GCP) and caches daily.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache, reduce
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import httpx

from skyward.cache import cached

if TYPE_CHECKING:
    from skyward.types import Instance

logger = logging.getLogger("skyward.pricing")

Provider = Literal["aws", "azure", "gcp", "digitalocean"]

VANTAGE_ENDPOINTS: dict[Provider, str] = {
    "aws": "https://instances.vantage.sh/instances.json",
    "azure": "https://instances.vantage.sh/azure/instances.json",
    "gcp": "https://instances.vantage.sh/gcp/instances.json",
}

DEFAULT_REGIONS: dict[Provider, str] = {
    "aws": "us-east-1",
    "azure": "eastus",
    "gcp": "us-central1",
    "digitalocean": "nyc1",
}

_CACHE_TTL = timedelta(days=1)


# =============================================================================
# Models
# =============================================================================


@dataclass(frozen=True, slots=True)
class InstancePricing:
    """Pricing for a cloud instance type."""

    instance_type: str
    provider: Provider
    region: str
    vcpu: int
    memory_gb: float
    gpu_count: int = 0
    ondemand: float | None = None
    spot_avg: float | None = None
    spot_min: float | None = None
    spot_max: float | None = None
    interrupt_frequency: str | None = None  # "<5%", "5-10%", "10-15%", "15-20%", ">20%"
    spot_savings_pct: int | None = None

    @property
    def spot_available(self) -> bool:
        return self.spot_avg is not None

    @property
    def spot_resilient(self) -> bool:
        return self.interrupt_frequency in ("<5%", "5-10%")

    def hourly_cost(self, spot: bool = False) -> float | None:
        return self.spot_avg if spot else self.ondemand


@dataclass(frozen=True, slots=True)
class FleetCost:
    """Cost breakdown for a fleet of instances."""

    total_hourly: float
    spot_hourly: float
    ondemand_hourly: float
    spot_count: int
    ondemand_count: int
    spot_pct: float
    savings_vs_ondemand: float
    savings_pct: float
    instances: tuple[tuple[str, bool, float], ...]

    @property
    def total_count(self) -> int:
        return self.spot_count + self.ondemand_count

    def format_summary(self) -> str:
        return (
            f"{self.total_count} instances: "
            f"{self.spot_count} spot ({self.spot_pct:.0f}%), "
            f"{self.ondemand_count} on-demand | "
            f"${self.total_hourly:.2f}/hr "
            f"(saving ${self.savings_vs_ondemand:.2f}/hr, {self.savings_pct:.0f}%)"
        )


# =============================================================================
# Parsing helpers (pure functions)
# =============================================================================


def _safe_float(value: str | float | None) -> float | None:
    """Parse float, None on failure."""
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _safe_int(value: str | int | float | None, default: int = 0) -> int:
    """Parse int, default on failure."""
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def _extract_linux_pricing(inst: dict[str, Any], region: str) -> dict[str, Any]:
    """Extract linux pricing dict from instance data."""
    pricing: dict[str, Any] = inst.get("pricing", {}).get(region, {}).get("linux", {})
    return pricing


def _parse_savings_pct(value: str | int | None) -> int | None:
    """Parse savings percentage from various formats."""
    if value is None:
        return None
    if isinstance(value, str):
        return _safe_int(value.replace("%", ""))
    return _safe_int(value)


# =============================================================================
# Data fetching with daily cache
# =============================================================================


@cached(namespace="pricing.aws", ttl=_CACHE_TTL)
def _fetch_aws_data() -> list[dict[str, Any]]:
    """Fetch AWS pricing data from Vantage."""
    logger.info("Fetching AWS pricing data from Vantage...")
    data: list[dict[str, Any]] = httpx.get(VANTAGE_ENDPOINTS["aws"], timeout=30).json()
    return data


@cached(namespace="pricing.azure", ttl=_CACHE_TTL)
def _fetch_azure_data() -> list[dict[str, Any]]:
    """Fetch Azure pricing data from Vantage."""
    logger.info("Fetching Azure pricing data from Vantage...")
    data: list[dict[str, Any]] = httpx.get(VANTAGE_ENDPOINTS["azure"], timeout=30).json()
    return data


@cached(namespace="pricing.gcp", ttl=_CACHE_TTL)
def _fetch_gcp_data() -> list[dict[str, Any]]:
    """Fetch GCP pricing data from Vantage."""
    logger.info("Fetching GCP pricing data from Vantage...")
    data: list[dict[str, Any]] = httpx.get(VANTAGE_ENDPOINTS["gcp"], timeout=30).json()
    return data


_FETCHERS: dict[Provider, Callable[[], list[dict[str, Any]]]] = {
    "aws": _fetch_aws_data,
    "azure": _fetch_azure_data,
    "gcp": _fetch_gcp_data,
}


@lru_cache(maxsize=128)
def _get_instance_type_key(inst_type: str) -> str:
    """Normalize instance type for lookup."""
    return inst_type.lower().strip()


def _get_type_from_inst(inst: dict[str, Any]) -> str:
    """Extract instance type name from various provider formats."""
    return inst.get("instance_type") or inst.get("name") or inst.get("pretty_name") or ""


def _build_pricing_index(
    data: list[dict[str, Any]], provider: Provider
) -> dict[str, dict[str, Any]]:
    """Build instance_type -> data index for fast lookup."""
    return {_get_instance_type_key(_get_type_from_inst(inst)): inst for inst in data}


# Memory-cached indexes (rebuilt when underlying data changes)
_pricing_indexes: dict[Provider, dict[str, dict[str, Any]]] = {}


def _get_pricing_index(provider: Provider) -> dict[str, dict[str, Any]]:
    """Get or build pricing index for provider."""
    if provider not in _pricing_indexes:
        fetcher = _FETCHERS.get(provider)
        if fetcher is None:
            raise ValueError(f"Unsupported provider: {provider}")
        data = fetcher()
        _pricing_indexes[provider] = _build_pricing_index(data, provider)
    return _pricing_indexes[provider]


# =============================================================================
# Public API
# =============================================================================


def get_instance_pricing(
    instance_type: str,
    provider: Provider = "aws",
    region: str | None = None,
) -> InstancePricing | None:
    """Get pricing for a specific instance type."""
    region = region or DEFAULT_REGIONS.get(provider, "us-east-1")

    try:
        index = _get_pricing_index(provider)
    except Exception as e:
        logger.warning(f"Failed to fetch pricing data for {provider}: {e}")
        return None

    inst = index.get(_get_instance_type_key(instance_type))
    if inst is None:
        return None

    linux = _extract_linux_pricing(inst, region)

    return InstancePricing(
        instance_type=instance_type,
        provider=provider,
        region=region,
        vcpu=_safe_int(inst.get("vCPU") or inst.get("vcpu")),
        memory_gb=_safe_float(inst.get("memory")) or 0.0,
        gpu_count=_safe_int(inst.get("GPU")),
        ondemand=_safe_float(linux.get("ondemand")),
        spot_avg=_safe_float(linux.get("spot_avg")),
        spot_min=_safe_float(linux.get("spot_min")),
        spot_max=_safe_float(linux.get("spot_max")),
        interrupt_frequency=linux.get("pct_interrupt"),
        spot_savings_pct=_parse_savings_pct(linux.get("pct_savings_od")),
    )


class _CostAccumulator(TypedDict):
    """Accumulator for fleet cost calculation."""

    spot_total: float
    ondemand_total: float
    spot_count: int
    ondemand_count: int
    all_ondemand: float
    details: list[tuple[str, bool, float]]


def calculate_fleet_cost(
    instances: list[tuple[str, bool]],
    region: str = "us-east-1",
    provider: Provider = "aws",
) -> FleetCost | None:
    """Calculate total fleet cost from (instance_type, is_spot) pairs."""
    if not instances:
        return None

    # Fetch pricing for all instances, filter None
    priced: list[tuple[str, bool, InstancePricing]] = [
        (t, s, p)
        for t, s in instances
        if (p := get_instance_pricing(t, provider, region)) is not None
    ]

    if not priced:
        return None

    # Calculate costs using reduce
    def accumulate(
        acc: _CostAccumulator, item: tuple[str, bool, InstancePricing]
    ) -> _CostAccumulator:
        inst_type, is_spot, pricing = item
        price = pricing.spot_avg if is_spot and pricing.spot_avg else pricing.ondemand or 0

        return {
            "spot_total": acc["spot_total"] + (price if is_spot else 0),
            "ondemand_total": acc["ondemand_total"] + (price if not is_spot else 0),
            "spot_count": acc["spot_count"] + (1 if is_spot else 0),
            "ondemand_count": acc["ondemand_count"] + (1 if not is_spot else 0),
            "all_ondemand": acc["all_ondemand"] + (pricing.ondemand or 0),
            "details": acc["details"] + [(inst_type, is_spot, price)],
        }

    initial: _CostAccumulator = {
        "spot_total": 0.0,
        "ondemand_total": 0.0,
        "spot_count": 0,
        "ondemand_count": 0,
        "all_ondemand": 0.0,
        "details": [],
    }

    result = reduce(accumulate, priced, initial)

    total = result["spot_total"] + result["ondemand_total"]
    total_count = result["spot_count"] + result["ondemand_count"]
    spot_pct = (result["spot_count"] / total_count * 100) if total_count else 0
    savings = result["all_ondemand"] - total
    savings_pct = (savings / result["all_ondemand"] * 100) if result["all_ondemand"] else 0

    return FleetCost(
        total_hourly=total,
        spot_hourly=result["spot_total"],
        ondemand_hourly=result["ondemand_total"],
        spot_count=result["spot_count"],
        ondemand_count=result["ondemand_count"],
        spot_pct=spot_pct,
        savings_vs_ondemand=savings,
        savings_pct=savings_pct,
        instances=tuple(result["details"]),
    )


def calculate_fleet_cost_from_instances(
    instances: tuple[Instance, ...],
    region: str = "us-east-1",
    provider: Provider = "aws",
) -> FleetCost | None:
    """Calculate fleet cost from Instance objects."""
    fleet_input = [
        (inst.get_meta("instance_type", ""), inst.spot)
        for inst in instances
        if inst.get_meta("instance_type")
    ]
    return calculate_fleet_cost(fleet_input, region, provider)


def format_fleet_log(
    instances: tuple[Instance, ...],
    region: str = "us-east-1",
    provider: Provider = "aws",
) -> str | None:
    """Format fleet cost for logging."""
    cost = calculate_fleet_cost_from_instances(instances, region, provider)
    return cost.format_summary() if cost else None
