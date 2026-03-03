"""Runtime catalog feed — fetches provider offers and caches locally.

Each provider's GPU offers are fetched on demand, cached as JSON in
``~/.skyward/cache/catalog/``, and refreshed when the per-provider TTL
expires.  Marketplace providers (VastAI, RunPod, TensorDock) use short
TTLs (minutes) while cloud providers (AWS, GCP) use longer ones (hours/days).

Usage::

    from skyward.offers.feed import ensure_fresh, refresh

    await ensure_fresh(providers=["aws", "vastai"])   # only if stale
    await refresh(providers=["vastai"])                # force re-fetch
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from skyward.infra.cache import CACHE_DIR
from skyward.observability.logger import logger

# ---------------------------------------------------------------------------
# Cache paths & per-provider TTLs
# ---------------------------------------------------------------------------

_CATALOG_DIR = CACHE_DIR / "catalog"
_META_FILE = _CATALOG_DIR / "meta.json"

_PROVIDER_TTL: dict[str, timedelta] = {
    "vastai": timedelta(minutes=5),
    "runpod": timedelta(minutes=15),
    "tensordock": timedelta(minutes=15),
    "hyperstack": timedelta(hours=6),
    "verda": timedelta(hours=6),
    "aws": timedelta(days=1),
    "gcp": timedelta(days=1),
}

ALL_PROVIDERS: tuple[str, ...] = tuple(_PROVIDER_TTL)

# ---------------------------------------------------------------------------
# GPU name normalization (moved from scripts/fetch_catalog.py)
# ---------------------------------------------------------------------------

# Hyperstack consumer: "geforcertx5090-pcie-32gb"
_HYPERSTACK_RE = re.compile(
    r"^(?:geforce)?-?(rtx)?-?([a-z]?\d+[a-z]?\d*)-.*$", re.IGNORECASE,
)
# Hyperstack datacenter: "A100-80G-PCIe", "H100-80G-PCIe-NVLink"
_HYPERSTACK_DC_RE = re.compile(r"^([A-Z]+\d+[A-Z]?)-\d+G(?:-.+)?$")


def _normalize_gpu_name(raw: str) -> str:
    """Normalize a provider GPU name to a canonical form."""
    name = raw.strip()

    # Strip count prefix: "1x ", "2x "
    name = re.sub(r"^\d+x\s+", "", name)

    # Strip "-spot" suffix
    name = re.sub(r"-spot$", "", name)

    # Strip vendor prefixes
    for prefix in (
        "AMD Instinct ", "NVIDIA GeForce ", "NVIDIA Tesla ", "NVIDIA ", "AMD ",
    ):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Hyperstack datacenter format: "A100-80G-PCIe" → "A100"
    if m := _HYPERSTACK_DC_RE.match(name):
        name = m.group(1)
    # Hyperstack consumer format: "geforcertx5090-pcie-32gb" → "RTX 5090"
    elif m := _HYPERSTACK_RE.match(name):
        prefix_part = (m.group(1) or "").upper()
        model_part = m.group(2).upper()
        name = f"{prefix_part} {model_part}".strip() if prefix_part else model_part

    # Dashed workstation names: "RTX-A6000" → "RTX A6000"
    name = re.sub(r"^(RTX)-([A-Z])", r"\1 \2", name)

    # Strip VRAM suffix: " 80GB", " 48GB"
    name = re.sub(r"\s+\d+\s*GB\s*$", "", name, flags=re.IGNORECASE)

    # Strip form factor suffix: " PCIe", " SXM", " SXM4", " OAM", " NVL", " NVLink"
    name = re.sub(
        r"\s*[-\s]?(?:PCIe|SXM\d?|OAM|NVL|NVLink)\s*$", "", name,
        flags=re.IGNORECASE,
    )

    return name.strip()


def _parse_vram_from_name(name: str) -> float:
    """Extract VRAM in GB from GPU display name (e.g. '... 24GB' -> 24.0)."""
    match = re.search(r"(\d+)\s*GB", name, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Private data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Accelerator:
    name: str
    vram: float
    manufacturer: str = ""
    architecture: str = ""
    cuda_min: str = ""
    cuda_max: str = ""


@dataclass(frozen=True, slots=True)
class _Spec:
    accelerator: _Accelerator
    vcpus: float
    memory_gb: float


@dataclass(frozen=True, slots=True)
class _Offer:
    spec: _Spec
    accelerator_count: int
    instance_type: str
    region: str
    spot_price: float | None = None
    on_demand_price: float | None = None
    billing_unit: str = "hour"
    specific: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Spec builder (enriches from accelerators SPECS)
# ---------------------------------------------------------------------------


def _make_spec(
    raw_name: str, vram: float, vcpus: float = 0, memory_gb: float = 0,
) -> _Spec:
    from skyward.accelerators.catalog import SPECS, get_gpu_vram_gb

    name = _normalize_gpu_name(raw_name)
    spec = SPECS.get(name)
    if vram <= 0 and spec:
        vram = float(get_gpu_vram_gb(name))
    accel = _Accelerator(
        name=name,
        vram=round(vram, 1),
        manufacturer=spec.get("manufacturer", "") if spec else "",
        architecture=spec.get("architecture", "") if spec else "",
        cuda_min=spec.get("cuda", {}).get("min", "") if spec else "",
        cuda_max=spec.get("cuda", {}).get("max", "") if spec else "",
    )
    return _Spec(accelerator=accel, vcpus=vcpus, memory_gb=round(memory_gb, 1))


# ---------------------------------------------------------------------------
# Provider fetchers
# ---------------------------------------------------------------------------


async def _fetch_vastai() -> list[_Offer]:
    from skyward.providers.vastai.client import VastAIClient, get_api_key

    api_key = get_api_key()
    async with VastAIClient(api_key) as client:
        raw_offers = await client.search_offers(limit=5000)

    offers: list[_Offer] = []
    for raw in raw_offers:
        gpu_name: str = raw.get("gpu_name", "")
        if not gpu_name:
            continue
        num_gpus = raw.get("num_gpus") or 1
        total_vram_gb = (raw.get("gpu_ram") or 0) / 1024
        per_gpu_vram = total_vram_gb / num_gpus if num_gpus > 0 else 0
        offers.append(_Offer(
            spec=_make_spec(
                gpu_name, per_gpu_vram,
                vcpus=float(raw.get("cpu_cores") or 0),
                memory_gb=(raw.get("cpu_ram") or 0) / 1024,
            ),
            accelerator_count=num_gpus,
            instance_type=f"{_normalize_gpu_name(gpu_name)}x{num_gpus}",
            region=raw.get("geolocation", "unknown"),
            spot_price=_safe_float(raw.get("min_bid")),
            on_demand_price=_safe_float(raw.get("dph_total")),
            billing_unit="hour",
        ))
    return offers


async def _fetch_runpod() -> list[_Offer]:
    from skyward.providers.runpod.client import RunPodClient, get_api_key
    from skyward.providers.runpod.config import RunPod

    api_key = get_api_key()
    async with RunPodClient(api_key, config=RunPod()) as client:
        gpu_types = await client.get_gpu_types()

    offers: list[_Offer] = []
    for gpu in gpu_types:
        display_name: str = gpu.get("displayName", "")
        if not display_name:
            continue
        lowest = gpu.get("lowestPrice") or {}
        spot = _safe_float(lowest.get("minimumBidPrice"))
        on_demand = _safe_float(lowest.get("uninterruptablePrice"))
        if spot is None and on_demand is None:
            continue
        gpu_id = gpu.get("id", display_name)
        offers.append(_Offer(
            spec=_make_spec(display_name, float(gpu.get("memoryInGb", 0))),
            accelerator_count=1,
            instance_type=gpu_id,
            region="global",
            spot_price=spot,
            on_demand_price=on_demand,
            billing_unit="hour",
            specific={"gpu_type_id": gpu_id},
        ))
    return offers


async def _fetch_hyperstack() -> list[_Offer]:
    from skyward.providers.hyperstack.client import HyperstackClient, get_api_key

    api_key = get_api_key()
    async with HyperstackClient(api_key) as client:
        flavors = await client.list_flavors()
        pricebook = await client.get_pricebook()

    prices: dict[str, float] = {}
    for entry in pricebook:
        name = entry.get("name", "")
        value = entry.get("value")
        if name and value is not None:
            prices[name] = float(value)

    grouped: dict[str, dict[str, Any]] = {}
    for flavor in flavors:
        gpu_name: str = flavor.get("gpu", "")
        if not gpu_name:
            continue
        flavor_name: str = flavor.get("name", "")
        is_spot = flavor_name.endswith("-spot")
        base_name = flavor_name.removesuffix("-spot")

        gpu_count = flavor.get("gpu_count", 1)
        per_gpu_price = prices.get(gpu_name)
        total_price = per_gpu_price * gpu_count if per_gpu_price is not None else None

        if base_name not in grouped:
            grouped[base_name] = {
                "gpu_name": gpu_name,
                "gpu_count": gpu_count,
                "vcpus": float(flavor.get("cpu", 0)),
                "memory_gb": float(flavor.get("ram", 0)),
                "region": flavor.get("region_name", "unknown"),
                "spot_price": None,
                "on_demand_price": None,
            }

        if is_spot:
            grouped[base_name]["spot_price"] = total_price
        else:
            grouped[base_name]["on_demand_price"] = total_price

    offers: list[_Offer] = []
    for base_name, g in grouped.items():
        offers.append(_Offer(
            spec=_make_spec(
                g["gpu_name"], 0,
                vcpus=g["vcpus"],
                memory_gb=g["memory_gb"],
            ),
            accelerator_count=g["gpu_count"],
            instance_type=base_name,
            region=g["region"],
            spot_price=g["spot_price"],
            on_demand_price=g["on_demand_price"],
            billing_unit="hour",
            specific={"flavor_name": base_name, "region": g["region"]},
        ))
    return offers


async def _fetch_tensordock() -> list[_Offer]:
    from skyward.providers.tensordock.client import TensorDockClient

    async with TensorDockClient(api_token="unused") as client:
        locations = await client.list_locations()

    offers: list[_Offer] = []
    for location in locations:
        loc_id = location.get("id", "")
        region = f"{location.get('city', '')}, {location.get('country', '')}"
        for gpu in location.get("gpus", []):
            display_name: str = gpu.get("displayName", "")
            resources = gpu.get("resources", {})
            gpu_model = gpu.get("v0Name", display_name)
            hourly_rate = _safe_float(gpu.get("price_per_hr"))
            offers.append(_Offer(
                spec=_make_spec(
                    display_name, _parse_vram_from_name(display_name),
                    vcpus=float(resources.get("max_vcpus", 0)),
                    memory_gb=float(resources.get("max_ram_gb", 0)),
                ),
                accelerator_count=1,
                instance_type=gpu_model,
                region=region,
                on_demand_price=hourly_rate,
                billing_unit="second",
                specific={
                    "location_id": loc_id,
                    "gpu_model": gpu_model,
                    "gpu_count": 1,
                    "vcpus": int(resources.get("max_vcpus", 0)),
                    "ram_gb": int(resources.get("max_ram_gb", 0)),
                    "hourly_rate": hourly_rate,
                },
            ))
    return offers


async def _fetch_verda() -> list[_Offer]:
    import httpx

    from skyward.providers.verda.client import (
        VERDA_API_BASE,
        VerdaClient,
        _OAuth2,
        get_credentials,
    )

    client_id, client_secret = get_credentials()
    auth = _OAuth2(client_id, client_secret, f"{VERDA_API_BASE}/oauth2/token")
    async with httpx.AsyncClient(
        base_url=VERDA_API_BASE, auth=auth, timeout=30,
    ) as http:
        client = VerdaClient(http)
        instance_types = await client.list_instance_types()
        on_demand_avail = await client.get_availability(is_spot=False)

    offers: list[_Offer] = []
    for itype in instance_types:
        gpu_info = itype.get("gpu")
        gpu_mem_info = itype.get("gpu_memory")
        cpu_info = itype.get("cpu", {})
        mem_info = itype.get("memory", {})
        itype_name = itype.get("instance_type", "")

        gpu_desc = gpu_info.get("description", "") if gpu_info else ""
        gpu_count = gpu_info.get("number_of_gpus", 0) if gpu_info else 0
        gpu_memory = gpu_mem_info.get("size_in_gigabytes", 0) if gpu_mem_info else 0

        if gpu_count == 0:
            continue

        per_gpu_vram = float(gpu_memory) / gpu_count if gpu_count > 0 else 0

        available_regions = [
            r for r, types in on_demand_avail.items()
            if itype_name in types
        ]

        for region in available_regions or ["unknown"]:
            offers.append(_Offer(
                spec=_make_spec(
                    gpu_desc, per_gpu_vram,
                    vcpus=float(cpu_info.get("number_of_cores", 0)),
                    memory_gb=float(mem_info.get("size_in_gigabytes", 0)),
                ),
                accelerator_count=gpu_count,
                instance_type=itype_name,
                region=region,
                spot_price=_safe_float(itype.get("spot_price")),
                on_demand_price=_safe_float(itype.get("price_per_hour")),
                billing_unit="hour",
                specific={"os_image": itype.get("os_image", "ubuntu-22.04")},
            ))
    return offers


async def _fetch_aws() -> list[_Offer]:
    from skyward.providers.aws.instances import (
        _fetch_ondemand_price,
        _fetch_spot_price,
        list_gpu_instances,
    )

    region = "us-east-1"
    gpu_instances = await list_gpu_instances(region)

    async def _with_pricing(res: Any) -> _Offer | None:
        spot, ondemand = await asyncio.gather(
            _fetch_spot_price(res.instance_type, region),
            _fetch_ondemand_price(res.instance_type, region),
        )
        per_gpu_vram = res.gpu_vram_gb / res.gpu_count if res.gpu_count > 0 else 0
        return _Offer(
            spec=_make_spec(
                res.gpu_model, per_gpu_vram,
                vcpus=float(res.vcpus),
                memory_gb=res.memory_gb,
            ),
            accelerator_count=res.gpu_count,
            instance_type=res.instance_type,
            region=region,
            spot_price=spot,
            on_demand_price=ondemand,
            billing_unit="second",
            specific={"architecture": res.architecture},
        )

    sem = asyncio.Semaphore(10)

    async def _bounded(res: Any) -> _Offer | None:
        async with sem:
            return await _with_pricing(res)

    results = await asyncio.gather(
        *(_bounded(r) for r in gpu_instances),
        return_exceptions=True,
    )
    return [r for r in results if isinstance(r, _Offer)]


async def _fetch_gcp() -> list[_Offer]:
    from concurrent.futures import ThreadPoolExecutor

    from skyward.providers.gcp.instances import (
        _BUILTIN_GPU_FAMILIES,
        estimate_vram,
        parse_builtin_gpu_count,
    )

    project, zone = _resolve_gcp_project(), "us-central1-a"

    from google.cloud import compute_v1

    machines_client = compute_v1.MachineTypesClient()
    loop = asyncio.get_event_loop()
    pool = ThreadPoolExecutor(max_workers=4)

    machines = await loop.run_in_executor(
        pool,
        lambda: list(machines_client.list(
            request=compute_v1.ListMachineTypesRequest(project=project, zone=zone),
        )),
    )

    gpu_family_accel: dict[str, str] = {
        "a2": "nvidia-tesla-a100",
        "a3": "nvidia-h100",
        "a4": "nvidia-h200",
        "g2": "nvidia-l4",
    }
    gpu_family_model: dict[str, str] = {
        "a2": "A100", "a3": "H100", "a4": "H200", "g2": "L4",
    }

    offers: list[_Offer] = []
    for mt in machines:
        family = mt.name.split("-")[0]
        if family not in _BUILTIN_GPU_FAMILIES:
            continue

        accel_type = gpu_family_accel.get(family, "")
        gpu_count = parse_builtin_gpu_count(mt.name)
        gpu_model: str = gpu_family_model.get(family) or family.upper()
        per_gpu_vram = float(estimate_vram(accel_type))

        uses_guest = family not in _BUILTIN_GPU_FAMILIES
        offers.append(_Offer(
            spec=_make_spec(
                gpu_model, per_gpu_vram,
                vcpus=float(mt.guest_cpus),
                memory_gb=mt.memory_mb / 1024,
            ),
            accelerator_count=gpu_count,
            instance_type=mt.name,
            region=zone,
            billing_unit="second",
            specific={
                "machine_type": mt.name,
                "uses_guest_accelerators": uses_guest,
                "accelerator_type": accel_type,
                "gpu_count": gpu_count,
                "gpu_model": gpu_model,
                "gpu_vram_gb": int(per_gpu_vram),
                "vcpus": mt.guest_cpus,
                "memory_gb": round(mt.memory_mb / 1024, 1),
            },
        ))

    pool.shutdown(wait=False)
    return offers


def _resolve_gcp_project() -> str:
    import os

    if p := os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return p
    if p := os.environ.get("GCLOUD_PROJECT"):
        return p

    import google.auth
    _, project = google.auth.default()
    if project:
        return project

    raise RuntimeError("Cannot resolve GCP project. Set GOOGLE_CLOUD_PROJECT.")


# ---------------------------------------------------------------------------
# Fetcher registry
# ---------------------------------------------------------------------------

type _Fetcher = Callable[[], Coroutine[Any, Any, list[_Offer]]]

_FETCHERS: dict[str, _Fetcher] = {
    "vastai": _fetch_vastai,
    "runpod": _fetch_runpod,
    "hyperstack": _fetch_hyperstack,
    "tensordock": _fetch_tensordock,
    "verda": _fetch_verda,
    "aws": _fetch_aws,
    "gcp": _fetch_gcp,
}


# ---------------------------------------------------------------------------
# JSON serialization (denormalized — each provider file is self-contained)
# ---------------------------------------------------------------------------


def _offer_to_dict(offer: _Offer) -> dict[str, Any]:
    return {
        "instance_type": offer.instance_type,
        "gpu": offer.spec.accelerator.name,
        "gpu_count": offer.accelerator_count,
        "gpu_vram": offer.spec.accelerator.vram,
        "vcpus": offer.spec.vcpus,
        "memory_gb": offer.spec.memory_gb,
        "region": offer.region,
        "spot_price": offer.spot_price,
        "on_demand_price": offer.on_demand_price,
        "billing_unit": offer.billing_unit,
        "manufacturer": offer.spec.accelerator.manufacturer,
        "architecture": offer.spec.accelerator.architecture,
        "cuda_min": offer.spec.accelerator.cuda_min,
        "cuda_max": offer.spec.accelerator.cuda_max,
        "specific": offer.specific,
    }


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

_log = logger.bind(component="catalog-feed")


def _read_meta() -> dict[str, str]:
    try:
        return json.loads(_META_FILE.read_text()) if _META_FILE.exists() else {}
    except Exception:
        return {}


def _write_meta(meta: dict[str, str]) -> None:
    _CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _META_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    tmp.rename(_META_FILE)


def provider_path(name: str) -> Path:
    """Return the cache file path for a given provider."""
    return _CATALOG_DIR / f"{name}.json"


def cached_provider_paths() -> list[Path]:
    """Return paths of all cached provider JSON files on disk."""
    if not _CATALOG_DIR.exists():
        return []
    return sorted(
        p for p in _CATALOG_DIR.glob("*.json") if p.name != "meta.json"
    )


def _is_stale(name: str, meta: dict[str, str]) -> bool:
    ts_str = meta.get(name)
    if not ts_str:
        return True
    try:
        fetched_at = datetime.fromisoformat(ts_str)
        ttl = _PROVIDER_TTL.get(name, timedelta(hours=6))
        return datetime.now(UTC) - fetched_at > ttl
    except ValueError:
        return True


def _write_provider_cache(name: str, offers: list[_Offer]) -> None:
    _CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    path = provider_path(name)
    data = [_offer_to_dict(o) for o in offers]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def _fetch_one(name: str) -> list[_Offer] | None:
    fetcher = _FETCHERS.get(name)
    if fetcher is None:
        _log.warning("Unknown provider {p}, skipping", p=name)
        return None
    try:
        offers = await fetcher()
        _log.info("Fetched {n} offers from {p}", n=len(offers), p=name)
        return offers
    except Exception as exc:
        _log.warning("Failed to fetch {p}: {err}", p=name, err=exc)
        return None


async def ensure_fresh(
    providers: Sequence[str] | None = None,
    *,
    force: bool = False,
) -> None:
    """Ensure the on-disk cache is fresh for the given providers.

    Only fetches providers whose TTL has expired (or all if *force* is True).
    Providers that fail to fetch are silently skipped — their existing cache
    (even if stale) is preserved.

    Parameters
    ----------
    providers
        Which providers to consider.  ``None`` means all known providers.
    force
        Bypass TTL check and always re-fetch.
    """
    targets = list(providers) if providers is not None else list(ALL_PROVIDERS)
    meta = _read_meta()

    stale = [p for p in targets if force or _is_stale(p, meta)]
    if not stale:
        return

    _log.info("Refreshing catalog for {providers}", providers=stale)

    results = await asyncio.gather(*(_fetch_one(p) for p in stale))

    for name, result in zip(stale, results, strict=True):
        if result is not None:
            _write_provider_cache(name, result)
            meta[name] = datetime.now(UTC).isoformat()

    _write_meta(meta)


async def refresh(providers: Sequence[str] | None = None) -> None:
    """Force-refresh the on-disk catalog cache.

    Parameters
    ----------
    providers
        Which providers to refresh.  ``None`` refreshes all known providers.
    """
    await ensure_fresh(providers=providers, force=True)
