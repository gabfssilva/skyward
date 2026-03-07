"""Queryable GPU offer catalog backed by SQLite.

Fetches provider offers on demand via ``provider.offers()``, caches locally in
``~/.skyward/cache/catalog/``, and loads into an in-memory SQLite database with
a fluent query builder.

Usage::

    import skyward as sky

    repo = await sky.offers.OfferRepository.create()

    # A100 cheapest spot offer
    offer = repo.accelerator("A100").accelerator_memory(80).spot().cheapest()

    # Top 5 Hopper with >= 4 accelerators
    offers = repo.architecture("Hopper").accelerator_count(4).cheapest(5)

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
