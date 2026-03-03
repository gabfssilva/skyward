"""Fetch GPU instance catalog from all Skyward providers.

Queries raw APIs of each supported provider, normalizes into a relational
JSON structure (specs, providers, offers), and writes to docs/catalog/.

Accelerator metadata (manufacturer, architecture, CUDA compatibility) is
enriched inline from ``skyward.accelerators.catalog.SPECS`` — the single
source of truth for hardware specs.

Usage:
    uv run python scripts/fetch_catalog.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CATALOG_DIR = Path(__file__).resolve().parent.parent / "docs" / "catalog"
OFFERS_DIR = CATALOG_DIR / "offers"

PROVIDER_NAMES: dict[str, str] = {
    "aws": "AWS",
    "gcp": "GCP",
    "vastai": "Vast.ai",
    "runpod": "RunPod",
    "hyperstack": "Hyperstack",
    "tensordock": "TensorDock",
    "verda": "Verda",
}


# ---------------------------------------------------------------------------
# GPU name normalization
# ---------------------------------------------------------------------------

# Hyperstack dashed format: "a40-pcie-48gb", "geforcertx5090-pcie-32gb"
_HYPERSTACK_RE = re.compile(
    r"^(?:geforce)?-?(rtx)?-?([a-z]?\d+[a-z]?\d*)-.*$", re.IGNORECASE,
)
# Hyperstack datacenter: "A100-80G-PCIe", "H100-80G-PCIe-NVLink"
_HYPERSTACK_DC_RE = re.compile(
    r"^([A-Z]+\d+[A-Z]?)-\d+G(?:-.+)?$",
)


def _normalize_gpu_name(raw: str) -> str:
    """Normalize a provider GPU name to a canonical form."""
    name = raw.strip()

    # Strip count prefix: "1x ", "2x "
    name = re.sub(r"^\d+x\s+", "", name)

    # Strip "-spot" suffix
    name = re.sub(r"-spot$", "", name)

    # Strip vendor prefixes
    for prefix in ("AMD Instinct ", "NVIDIA GeForce ", "NVIDIA Tesla ",
                    "NVIDIA ", "AMD "):
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
    name = re.sub(r"\s*[-\s]?(?:PCIe|SXM\d?|OAM|NVL|NVLink)\s*$", "", name,
                  flags=re.IGNORECASE)

    return name.strip()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Accelerator:
    name: str
    vram: float
    manufacturer: str = ""
    architecture: str = ""
    cuda_min: str = ""
    cuda_max: str = ""

    @property
    def id(self) -> str:
        raw = f"{self.name}:{self.vram}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class Spec:
    accelerator: Accelerator
    vcpus: float
    memory_gb: float

    @property
    def id(self) -> str:
        raw = f"{self.accelerator.id}:{self.vcpus}:{self.memory_gb}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class Offer:
    spec: Spec
    accelerator_count: int
    instance_type: str
    region: str
    spot_price: float | None = None
    on_demand_price: float | None = None
    billing_unit: str = "hour"
    specific: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    raw_name: str, vram: float, vcpus: float = 0, memory_gb: float = 0,
) -> Spec:
    """Create a Spec with an Accelerator enriched from SPECS."""
    from skyward.accelerators.catalog import SPECS, get_gpu_vram_gb

    name = _normalize_gpu_name(raw_name)
    spec = SPECS.get(name)
    if vram <= 0 and spec:
        vram = float(get_gpu_vram_gb(name))
    accel = Accelerator(
        name=name,
        vram=round(vram, 1),
        manufacturer=spec.get("manufacturer", "") if spec else "",
        architecture=spec.get("architecture", "") if spec else "",
        cuda_min=spec.get("cuda", {}).get("min", "") if spec else "",
        cuda_max=spec.get("cuda", {}).get("max", "") if spec else "",
    )
    return Spec(accelerator=accel, vcpus=vcpus, memory_gb=round(memory_gb, 1))


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
# Provider fetchers
# ---------------------------------------------------------------------------


async def fetch_vastai() -> list[Offer]:
    from skyward.providers.vastai.client import VastAIClient, get_api_key

    api_key = get_api_key()
    async with VastAIClient(api_key) as client:
        raw_offers = await client.search_offers(limit=5000)

    offers: list[Offer] = []
    for raw in raw_offers:
        gpu_name: str = raw.get("gpu_name", "")
        if not gpu_name:
            continue
        num_gpus = raw.get("num_gpus") or 1
        total_vram_gb = (raw.get("gpu_ram") or 0) / 1024
        per_gpu_vram = total_vram_gb / num_gpus if num_gpus > 0 else 0
        offers.append(Offer(
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


async def fetch_runpod() -> list[Offer]:
    from skyward.providers.runpod.client import RunPodClient, get_api_key
    from skyward.providers.runpod.config import RunPod

    api_key = get_api_key()
    async with RunPodClient(api_key, config=RunPod()) as client:
        gpu_types = await client.get_gpu_types()

    offers: list[Offer] = []
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
        offers.append(Offer(
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


async def fetch_hyperstack() -> list[Offer]:
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

    # Group by base flavor name (strip "-spot") to merge spot + on-demand prices
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

    offers: list[Offer] = []
    for base_name, g in grouped.items():
        offers.append(Offer(
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


async def fetch_tensordock() -> list[Offer]:
    from skyward.providers.tensordock.client import TensorDockClient

    async with TensorDockClient(api_token="unused") as client:
        locations = await client.list_locations()

    offers: list[Offer] = []
    for location in locations:
        loc_id = location.get("id", "")
        region = f"{location.get('city', '')}, {location.get('country', '')}"
        for gpu in location.get("gpus", []):
            display_name: str = gpu.get("displayName", "")
            resources = gpu.get("resources", {})
            gpu_model = gpu.get("v0Name", display_name)
            hourly_rate = _safe_float(gpu.get("price_per_hr"))
            offers.append(Offer(
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


async def fetch_verda() -> list[Offer]:
    import httpx

    from skyward.providers.verda.client import (
        VERDA_API_BASE,
        VerdaClient,
        _OAuth2,
        get_credentials,
    )

    client_id, client_secret = get_credentials()
    auth = _OAuth2(client_id, client_secret, f"{VERDA_API_BASE}/oauth2/token")
    async with httpx.AsyncClient(base_url=VERDA_API_BASE, auth=auth, timeout=30) as http:
        client = VerdaClient(http)
        instance_types = await client.list_instance_types()
        on_demand_avail = await client.get_availability(is_spot=False)
        spot_avail = await client.get_availability(is_spot=True)

    all_regions: set[str] = set()
    for regions in (*on_demand_avail.values(), *spot_avail.values()):
        all_regions |= set(regions) if isinstance(regions, frozenset) else set()

    offers: list[Offer] = []
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
            offers.append(Offer(
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


async def fetch_aws() -> list[Offer]:
    from skyward.providers.aws.instances import (
        _fetch_ondemand_price,
        _fetch_spot_price,
        list_gpu_instances,
    )

    region = "us-east-1"
    gpu_instances = await list_gpu_instances(region)

    async def _with_pricing(res: Any) -> Offer | None:
        spot, ondemand = await asyncio.gather(
            _fetch_spot_price(res.instance_type, region),
            _fetch_ondemand_price(res.instance_type, region),
        )
        per_gpu_vram = res.gpu_vram_gb / res.gpu_count if res.gpu_count > 0 else 0
        return Offer(
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

    async def _bounded(res: Any) -> Offer | None:
        async with sem:
            return await _with_pricing(res)

    results = await asyncio.gather(
        *(_bounded(r) for r in gpu_instances),
        return_exceptions=True,
    )
    return [r for r in results if isinstance(r, Offer)]


async def fetch_gcp() -> list[Offer]:
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

    offers: list[Offer] = []
    for mt in machines:
        family = mt.name.split("-")[0]
        if family not in _BUILTIN_GPU_FAMILIES:
            continue

        accel_type = gpu_family_accel.get(family, "")
        gpu_count = parse_builtin_gpu_count(mt.name)
        gpu_model: str = gpu_family_model.get(family) or family.upper()
        per_gpu_vram = float(estimate_vram(accel_type))

        uses_guest = family not in _BUILTIN_GPU_FAMILIES
        offers.append(Offer(
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
# Output serialization
# ---------------------------------------------------------------------------


def _serialize_accelerator(accel: Accelerator) -> dict[str, Any]:
    d = asdict(accel)
    d["id"] = accel.id
    return d


def _serialize_spec(spec: Spec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "accelerator_id": spec.accelerator.id,
        "vcpus": spec.vcpus,
        "memory_gb": spec.memory_gb,
    }


def _serialize_offer(offer: Offer) -> dict[str, Any]:
    return {
        "spec_id": offer.spec.id,
        "accelerator_count": offer.accelerator_count,
        "instance_type": offer.instance_type,
        "region": offer.region,
        "spot_price": offer.spot_price,
        "on_demand_price": offer.on_demand_price,
        "billing_unit": offer.billing_unit,
        "specific": offer.specific,
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


FETCHERS: dict[str, Any] = {
    "vastai": fetch_vastai,
    "runpod": fetch_runpod,
    "hyperstack": fetch_hyperstack,
    "tensordock": fetch_tensordock,
    "verda": fetch_verda,
    "aws": fetch_aws,
    "gcp": fetch_gcp,
}


async def main() -> None:
    results = await asyncio.gather(
        *(fn() for fn in FETCHERS.values()),
        return_exceptions=True,
    )

    all_specs: dict[str, Spec] = {}
    all_accelerators: dict[str, Accelerator] = {}
    metadata: dict[str, Any] = {
        "updated_at": datetime.now(UTC).isoformat(),
        "providers": {},
    }

    for provider_id, result in zip(FETCHERS, results, strict=False):
        if isinstance(result, BaseException):
            metadata["providers"][provider_id] = {
                "status": "error",
                "error": str(result),
            }
            print(f"  [SKIP] {provider_id}: {result}", file=sys.stderr)
            continue

        provider_offers: list[Offer] = result
        metadata["providers"][provider_id] = {
            "status": "ok",
            "count": len(provider_offers),
        }
        print(f"  [OK]   {provider_id}: {len(provider_offers)} offers")

        for offer in provider_offers:
            all_specs[offer.spec.id] = offer.spec
            all_accelerators[offer.spec.accelerator.id] = offer.spec.accelerator

        _write_json(
            OFFERS_DIR / f"{provider_id}.json",
            [_serialize_offer(o) for o in provider_offers],
        )

    _write_json(
        CATALOG_DIR / "accelerators.json",
        sorted(
            [_serialize_accelerator(a) for a in all_accelerators.values()],
            key=lambda a: (a["manufacturer"], a["architecture"], a["name"]),
        ),
    )

    _write_json(
        CATALOG_DIR / "specs.json",
        sorted(
            [_serialize_spec(s) for s in all_specs.values()],
            key=lambda s: (s["accelerator_id"], s["vcpus"]),
        ),
    )

    _write_json(
        CATALOG_DIR / "providers.json",
        [
            {"id": pid, "name": PROVIDER_NAMES[pid]}
            for pid in FETCHERS
            if metadata["providers"].get(pid, {}).get("status") == "ok"
        ],
    )

    _write_json(CATALOG_DIR / "metadata.json", metadata)

    ok = sum(1 for p in metadata["providers"].values() if p["status"] == "ok")
    total = len(FETCHERS)
    print(f"\nCatalog written to {CATALOG_DIR}")
    print(
        f"Providers: {ok}/{total} succeeded, "
        f"{len(all_accelerators)} accelerators, {len(all_specs)} specs"
    )


if __name__ == "__main__":
    asyncio.run(main())
