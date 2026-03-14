"""Catalog offer model — flattened view of accelerator + spec + offer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CatalogOffer:
    """A single offer from the catalog.

    Represents the pre-joined view across accelerators, specs, and offers tables.
    All fields correspond to columns in the ``catalog`` SQLite VIEW.
    """

    provider: str
    instance_type: str
    region: str
    accelerator_name: str
    accelerator_count: float
    accelerator_memory_gb: float
    manufacturer: str
    architecture: str
    cuda_min: str
    cuda_max: str
    vcpus: float
    memory_gb: float
    cpu_architecture: str
    spot_price: float | None
    on_demand_price: float | None
    billing_unit: str
    specific: str | None
