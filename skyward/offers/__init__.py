"""Queryable GPU offer catalog backed by persistent SQLite.

Persists to ``~/.skyward/offers.db``. Fetches provider offers on demand and
exposes a fluent query builder.

Usage::

    import skyward as sky

    repo = await sky.offers.OfferRepository.create()

    # A100 cheapest spot offer
    offer = await repo.accelerator("A100").accelerator_memory(80).spot().cheapest()

    # Fluent select
    offers = await repo.select().architecture("Hopper").order_by("spot_price ASC").limit(5).all()

    # Raw SQL
    offers = repo.query("SELECT * FROM catalog WHERE accelerator_name LIKE 'H%' ORDER BY spot_price")
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
