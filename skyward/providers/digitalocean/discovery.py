"""DigitalOcean instance type discovery and parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skyward.utils.cache import cached
from skyward.types import InstanceSpec

if TYPE_CHECKING:
    from pydo import Client

    from skyward.accelerators import AcceleratorSpec


def normalize_gpu_model(model: str | None) -> str | None:
    """Normalize DigitalOcean GPU model name.

    Args:
        model: Raw GPU model string from API (e.g., "nvidia_h100").

    Returns:
        Normalized model name (e.g., "H100") or None.
    """
    if not model:
        return None

    model = model.lower()
    for prefix in ("nvidia_", "amd_"):
        if model.startswith(prefix):
            model = model[len(prefix) :]
            break

    return model.upper()


def parse_droplet_size(size: dict[str, Any]) -> InstanceSpec:
    """Parse DigitalOcean size to InstanceSpec.

    Args:
        size: Size dict from DigitalOcean API.

    Returns:
        InstanceSpec with normalized fields.
    """
    gpu_info = size.get("gpu_info") or {}

    accelerator = None
    accelerator_count = 0
    accelerator_memory_gb = 0.0

    if gpu_info:
        accelerator = normalize_gpu_model(gpu_info.get("model"))
        accelerator_count = gpu_info.get("count", 0)
        vram = gpu_info.get("vram") or {}
        accelerator_memory_gb = vram.get("amount", 0)

    return InstanceSpec(
        name=size["slug"],
        vcpu=size.get("vcpus", 0),
        memory_gb=size.get("memory", 0) / 1024,
        accelerator=accelerator,
        accelerator_count=accelerator_count,
        accelerator_memory_gb=accelerator_memory_gb,
        price_on_demand=size.get("price_hourly"),
        price_spot=None,  # DigitalOcean does not have spot pricing
        billing_increment_minutes=None,  # per-second billing (60s min)
        metadata={
            "regions": size.get("regions", []),
            "price_monthly": size.get("price_monthly"),
        },
    )


def get_gpu_image(accelerator: AcceleratorSpec, accelerator_count: int = 1) -> str:
    """Get the appropriate GPU image for the accelerator type.

    Args:
        accelerator: Accelerator specification.
        accelerator_count: Number of GPUs.

    Returns:
        DigitalOcean image slug (e.g., "gpu-h100x1-base").
    """
    acc_str = accelerator.accelerator if hasattr(accelerator, "accelerator") else str(accelerator)

    if acc_str and "MI3" in acc_str.upper():
        return "gpu-amd-base"

    if accelerator_count == 8:
        return "gpu-h100x8-base"

    return "gpu-h100x1-base"


@cached(namespace="digitalocean")
def fetch_available_instances(client: Client) -> tuple[InstanceSpec, ...]:
    """Fetch all available droplet sizes from DigitalOcean API.

    Results are cached for 24 hours.

    Args:
        client: Authenticated pydo Client.

    Returns:
        Tuple of InstanceSpec sorted by accelerator, count, vcpu, memory.
    """
    sizes: list[dict[str, Any]] = []

    page = 1
    while True:
        resp = client.sizes.list(page=page, per_page=100)
        sizes.extend(resp.get("sizes", []))

        links = resp.get("links", {})
        pages = links.get("pages", {})
        if not pages.get("next"):
            break
        page += 1

    specs = [parse_droplet_size(s) for s in sizes if s.get("available", True)]

    return tuple(
        sorted(
            specs,
            key=lambda s: (
                s.Accelerator or "",
                s.accelerator_count,
                s.vcpu,
                s.memory_gb,
            ),
        )
    )
