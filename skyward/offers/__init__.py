"""Queryable GPU offer catalog backed by SQLite.

Fetches provider offers on demand, caches locally in ``~/.skyward/cache/catalog/``,
and loads into an in-memory SQLite database with a fluent query builder.

Usage::

    import skyward as sky

    repo = await sky.offers.OfferRepository.create()

    # A100 cheapest spot offer
    offer = repo.accelerator("A100").vram(80).spot().cheapest()

    # Top 5 Hopper with >= 4 GPUs
    offers = repo.architecture("Hopper").gpus(4).cheapest(5)

    # Raw SQL
    offers = repo.query("SELECT * FROM catalog WHERE gpu LIKE 'H%' ORDER BY spot_price")

    # Force-refresh the cache
    await sky.offers.refresh(providers=["vastai"])
"""

from __future__ import annotations

from .conversion import to_offer
from .feed import ensure_fresh, refresh
from .model import CatalogOffer
from .query import OfferQuery
from .repository import OfferRepository

__all__ = [
    "CatalogOffer",
    "OfferQuery",
    "OfferRepository",
    "ensure_fresh",
    "refresh",
    "to_offer",
]
