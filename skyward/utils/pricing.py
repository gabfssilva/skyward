"""Cloud instance pricing data with daily cache.

Fetches pricing from Vantage (AWS, Azure, GCP) and caches daily.
"""

from __future__ import annotations

import json as json_mod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Any, Literal
from urllib.request import urlopen

from .cache import cached

Provider = Literal["aws", "azure", "gcp"]

VANTAGE_ENDPOINTS: dict[Provider, str] = {
    "aws": "https://instances.vantage.sh/instances.json",
    "azure": "https://instances.vantage.sh/azure/instances.json",
    "gcp": "https://instances.vantage.sh/gcp/instances.json",
}

DEFAULT_REGIONS: dict[Provider, str] = {
    "aws": "us-east-1",
    "azure": "eastus",
    "gcp": "us-central1",
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


def _fetch_json(url: str) -> list[dict[str, Any]]:
    with urlopen(url, timeout=30) as resp:  # noqa: S310
        return json_mod.loads(resp.read())


@cached(namespace="pricing.aws", ttl=_CACHE_TTL)
def _fetch_aws_data() -> list[dict[str, Any]]:
    return _fetch_json(VANTAGE_ENDPOINTS["aws"])


@cached(namespace="pricing.azure", ttl=_CACHE_TTL)
def _fetch_azure_data() -> list[dict[str, Any]]:
    return _fetch_json(VANTAGE_ENDPOINTS["azure"])


@cached(namespace="pricing.gcp", ttl=_CACHE_TTL)
def _fetch_gcp_data() -> list[dict[str, Any]]:
    return _fetch_json(VANTAGE_ENDPOINTS["gcp"])


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
    except Exception:
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
