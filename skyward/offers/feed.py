"""Catalog feed utilities — GPU normalization, internal models, and cache I/O.

Shared helpers used by the offer repository and conversion layer.  Provider
offers are now fetched directly via ``provider.offers()`` instead of the
per-provider fetchers that previously lived here.

Cache structure::

    ~/.skyward/cache/catalog/
    ├── accelerators.json     # shared accelerator metadata
    ├── specs.json            # shared spec (accelerator + vcpus + memory_gb)
    └── offers/
        ├── aws.json          # per-provider offers (references spec_id)
        ├── tensordock.json
        └── ...
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skyward.infra.cache import CACHE_DIR

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

CATALOG_DIR = CACHE_DIR / "catalog"
_OFFERS_DIR = CATALOG_DIR / "offers"

# ---------------------------------------------------------------------------
# GPU name normalization
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
    cpu_architecture: str = "x86_64"


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
# ID computation (same MD5 formula as the original fetch_catalog.py script)
# ---------------------------------------------------------------------------


def _accel_id(accel: _Accelerator) -> str | None:
    if not accel.name:
        return None
    return hashlib.md5(f"{accel.name}:{accel.vram}".encode()).hexdigest()[:12]


def _spec_id(spec: _Spec) -> str:
    aid = _accel_id(spec.accelerator) or "cpu"
    return hashlib.md5(f"{aid}:{spec.vcpus}:{spec.memory_gb}:{spec.cpu_architecture}".encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Spec builder (enriches from accelerators SPECS)
# ---------------------------------------------------------------------------


_CPU_ACCELERATOR = _Accelerator(name="", vram=0)


def _make_spec(
    raw_name: str, vram: float, vcpus: float = 0, memory_gb: float = 0,
    cpu_architecture: str = "x86_64",
) -> _Spec:
    if not raw_name.strip():
        return _Spec(accelerator=_CPU_ACCELERATOR, vcpus=vcpus, memory_gb=round(memory_gb, 1), cpu_architecture=cpu_architecture)

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
    return _Spec(accelerator=accel, vcpus=vcpus, memory_gb=round(memory_gb, 1), cpu_architecture=cpu_architecture)


# ---------------------------------------------------------------------------
# Normalized JSON serialization
# ---------------------------------------------------------------------------


def _serialize_accelerator(accel: _Accelerator) -> dict[str, Any]:
    return {
        "id": _accel_id(accel),
        "name": accel.name,
        "vram": accel.vram,
        "manufacturer": accel.manufacturer,
        "architecture": accel.architecture,
        "cuda_min": accel.cuda_min,
        "cuda_max": accel.cuda_max,
    }


def _serialize_spec(spec: _Spec) -> dict[str, Any]:
    return {
        "id": _spec_id(spec),
        "accelerator_id": _accel_id(spec.accelerator),
        "vcpus": spec.vcpus,
        "memory_gb": round(spec.memory_gb, 1),
        "cpu_architecture": spec.cpu_architecture,
    }


def _serialize_offer(offer: _Offer) -> dict[str, Any]:
    return {
        "spec_id": _spec_id(offer.spec),
        "accelerator_count": offer.accelerator_count,
        "instance_type": offer.instance_type,
        "region": offer.region,
        "spot_price": offer.spot_price,
        "on_demand_price": offer.on_demand_price,
        "billing_unit": offer.billing_unit,
        "specific": offer.specific,
    }


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.rename(path)


def _merge_globals(fresh: dict[str, list[_Offer]]) -> None:
    """Merge new accelerators/specs from freshly-fetched offers into global files."""
    accels: dict[str, dict[str, Any]] = {}
    specs: dict[str, dict[str, Any]] = {}

    accels_file = CATALOG_DIR / "accelerators.json"
    if accels_file.exists():
        for a in json.loads(accels_file.read_text()):
            accels[a["id"]] = a

    specs_file = CATALOG_DIR / "specs.json"
    if specs_file.exists():
        for s in json.loads(specs_file.read_text()):
            specs[s["id"]] = s

    for offers in fresh.values():
        for offer in offers:
            aid = _accel_id(offer.spec.accelerator)
            sid = _spec_id(offer.spec)
            if aid and aid not in accels:
                accels[aid] = _serialize_accelerator(offer.spec.accelerator)
            if sid not in specs:
                specs[sid] = _serialize_spec(offer.spec)

    _write_json(
        accels_file,
        sorted(accels.values(), key=lambda a: (a["manufacturer"], a["name"])),
    )
    _write_json(
        specs_file,
        sorted(specs.values(), key=lambda s: (s["accelerator_id"], s["vcpus"])),
    )
