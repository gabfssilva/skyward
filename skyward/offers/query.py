"""Chainable query builder over the catalog SQLite VIEW."""

from __future__ import annotations

import sqlite3
from typing import Any, overload

from .model import CatalogOffer


class OfferQuery:
    """Fluent builder that accumulates WHERE clauses and compiles to SQL on terminal call.

    Filters are chainable and additive (AND). Terminal methods execute the query.
    """

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        self._clauses: list[str] = []
        self._params: list[Any] = []

    # ── chainable filters ────────────────────────────────────

    def accelerator(self, name: str) -> OfferQuery:
        """Filter by GPU name (exact match, e.g. ``"A100"``)."""
        self._clauses.append("gpu = ?")
        self._params.append(name)
        return self

    def provider(self, name: str) -> OfferQuery:
        """Filter by provider id (e.g. ``"aws"``, ``"vastai"``)."""
        self._clauses.append("provider = ?")
        self._params.append(name)
        return self

    def region(self, region: str) -> OfferQuery:
        """Filter by region (exact match)."""
        self._clauses.append("region = ?")
        self._params.append(region)
        return self

    def architecture(self, arch: str) -> OfferQuery:
        """Filter by GPU architecture (e.g. ``"Hopper"``, ``"Ampere"``)."""
        self._clauses.append("architecture = ?")
        self._params.append(arch)
        return self

    def manufacturer(self, name: str) -> OfferQuery:
        """Filter by manufacturer (e.g. ``"NVIDIA"``, ``"AMD"``)."""
        self._clauses.append("manufacturer = ?")
        self._params.append(name)
        return self

    def vram(self, min_gb: float) -> OfferQuery:
        """Filter by minimum VRAM per GPU in GB."""
        self._clauses.append("vram >= ?")
        self._params.append(min_gb)
        return self

    def vcpus(self, min_vcpus: float) -> OfferQuery:
        """Filter by minimum vCPUs."""
        self._clauses.append("vcpus >= ?")
        self._params.append(min_vcpus)
        return self

    def memory(self, min_gb: float) -> OfferQuery:
        """Filter by minimum system memory in GB."""
        self._clauses.append("memory_gb >= ?")
        self._params.append(min_gb)
        return self

    def gpus(self, min_count: int) -> OfferQuery:
        """Filter by minimum GPU count per node."""
        self._clauses.append("gpu_count >= ?")
        self._params.append(min_count)
        return self

    def spot(self) -> OfferQuery:
        """Only offers with spot pricing."""
        self._clauses.append("spot_price IS NOT NULL")
        return self

    def on_demand(self) -> OfferQuery:
        """Only offers with on-demand pricing."""
        self._clauses.append("on_demand_price IS NOT NULL")
        return self

    def max_price(self, usd_hr: float) -> OfferQuery:
        """Filter by maximum price per hour (spot preferred, falls back to on-demand)."""
        self._clauses.append("COALESCE(spot_price, on_demand_price) <= ?")
        self._params.append(usd_hr)
        return self

    def where(self, clause: str, *params: Any) -> OfferQuery:
        """Append a raw SQL WHERE clause (e.g. ``"gpu LIKE ?", "H%"``)."""
        self._clauses.append(clause)
        self._params.extend(params)
        return self

    # ── terminals ────────────────────────────────────────────

    @overload
    def cheapest(self) -> CatalogOffer | None: ...
    @overload
    def cheapest(self, n: int) -> list[CatalogOffer]: ...

    def cheapest(self, n: int | None = None) -> CatalogOffer | list[CatalogOffer] | None:
        """Return the cheapest offer(s), ordered by lowest available price.

        Without arguments returns a single ``CatalogOffer | None``.
        With ``n`` returns the ``n`` cheapest as a list.
        """
        order = "COALESCE(spot_price, on_demand_price) ASC NULLS LAST"
        if n is None:
            rows = self._execute(order_by=order, limit=1)
            return rows[0] if rows else None
        return self._execute(order_by=order, limit=n)

    def all(self) -> list[CatalogOffer]:
        """Execute the query and return all matching offers."""
        return self._execute()

    def first(self) -> CatalogOffer | None:
        """Execute the query and return the first match, or ``None``."""
        rows = self._execute(limit=1)
        return rows[0] if rows else None

    # ── internals ────────────────────────────────────────────

    def _execute(self, *, order_by: str | None = None, limit: int | None = None) -> list[CatalogOffer]:
        sql = "SELECT * FROM catalog"
        if self._clauses:
            sql += " WHERE " + " AND ".join(self._clauses)
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"
        rows = self._db.execute(sql, self._params).fetchall()
        return [CatalogOffer(**dict(zip(row.keys(), row, strict=True))) for row in rows]
