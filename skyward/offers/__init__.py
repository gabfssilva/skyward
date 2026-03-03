"""Queryable GPU offer catalog backed by SQLite.

Loads the static JSON catalog (accelerators, specs, offers) into an in-memory
SQLite database and exposes a fluent query builder for filtering and ranking.

Usage::

    import skyward as sky

    repo = await sky.offers.OfferRepository.create()

    # A100 cheapest spot offer
    offer = repo.accelerator("A100").vram(80).spot().cheapest()

    # Top 5 Hopper with >= 4 GPUs
    offers = repo.architecture("Hopper").gpus(4).cheapest(5)

    # Raw SQL
    offers = repo.query("SELECT * FROM catalog WHERE gpu LIKE 'H%' ORDER BY spot_price")
"""

from __future__ import annotations

from .conversion import to_offer
from .model import CatalogOffer
from .query import OfferQuery
from .repository import OfferRepository

__all__ = [
    "CatalogOffer",
    "OfferQuery",
    "OfferRepository",
    "to_offer",
]
