"""Convert CatalogOffer → runtime Offer with per-provider specific deserialization."""

from __future__ import annotations

import json
from typing import Any, cast

from skyward.accelerators.spec import Accelerator
from skyward.api.model import InstanceType, Offer
from skyward.api.spec import Architecture

from .model import CatalogOffer


def to_offer(co: CatalogOffer) -> Offer:
    """Convert a CatalogOffer from the SQLite catalog into a runtime Offer."""
    accelerator = Accelerator(name=co.gpu, count=co.gpu_count)

    it = InstanceType(
        name=co.instance_type,
        accelerator=accelerator,
        vcpus=co.vcpus,
        memory_gb=co.memory_gb,
        architecture=cast(Architecture, "x86_64"),
        specific=None,
    )

    raw: dict[str, Any] | None = json.loads(co.specific) if co.specific else None
    specific = _build_specific(co.provider, raw)

    billing_unit = co.billing_unit
    if billing_unit not in ("second", "minute", "hour"):
        billing_unit = "hour"

    return Offer(
        id=f"{co.provider}-{co.region}-{co.instance_type}",
        instance_type=it,
        spot_price=co.spot_price,
        on_demand_price=co.on_demand_price,
        billing_unit=cast(Any, billing_unit),
        specific=specific,
    )


def _build_specific(provider: str, raw: dict[str, Any] | None) -> Any:
    """Dispatch to per-provider specific builder."""
    match provider:
        case "aws":
            return _aws_specific(raw)
        case "gcp":
            return _gcp_specific(raw)
        case "vastai":
            return 0
        case "runpod":
            return _runpod_specific(raw)
        case "hyperstack":
            return raw or {}
        case "tensordock":
            return _tensordock_specific(raw)
        case "verda":
            return _verda_specific(raw)
        case _:
            return raw


def _aws_specific(_raw: dict[str, Any] | None) -> Any:
    from skyward.providers.aws.provider import AWSOfferSpecific

    return AWSOfferSpecific(ami="")


def _gcp_specific(raw: dict[str, Any] | None) -> Any:
    from skyward.providers.gcp.instances import ResolvedMachine

    if raw is None:
        return ResolvedMachine(
            machine_type="", uses_guest_accelerators=False,
            accelerator_type="", gpu_count=0, gpu_model="",
            gpu_vram_gb=0, vcpus=0, memory_gb=0.0,
        )
    return ResolvedMachine(
        machine_type=raw.get("machine_type", ""),
        uses_guest_accelerators=raw.get("uses_guest_accelerators", False),
        accelerator_type=raw.get("accelerator_type", ""),
        gpu_count=raw.get("gpu_count", 0),
        gpu_model=raw.get("gpu_model", ""),
        gpu_vram_gb=raw.get("gpu_vram_gb", 0),
        vcpus=raw.get("vcpus", 0),
        memory_gb=raw.get("memory_gb", 0.0),
    )


def _runpod_specific(raw: dict[str, Any] | None) -> Any:
    from skyward.providers.runpod.provider import RunPodOfferData

    if raw is None:
        return RunPodOfferData(gpu_type_id="", min_cuda=None)
    return RunPodOfferData(
        gpu_type_id=raw.get("gpu_type_id", ""),
        min_cuda=raw.get("min_cuda"),
    )


def _tensordock_specific(raw: dict[str, Any] | None) -> Any:
    from skyward.providers.tensordock.provider import TensorDockOfferData

    if raw is None:
        return TensorDockOfferData(
            location_id="", gpu_model="", gpu_count=0,
            vcpus=0, ram_gb=0, hourly_rate=0.0,
        )
    return TensorDockOfferData(
        location_id=raw.get("location_id", ""),
        gpu_model=raw.get("gpu_model", ""),
        gpu_count=raw.get("gpu_count", 0),
        vcpus=raw.get("vcpus", 0),
        ram_gb=raw.get("ram_gb", 0),
        hourly_rate=raw.get("hourly_rate", 0.0),
    )


def _verda_specific(raw: dict[str, Any] | None) -> str:
    if raw is None:
        return "ubuntu-22.04"
    return raw.get("os_image", "ubuntu-22.04")
