"""Chainable query builder over the catalog SQLite VIEW."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, overload

from .model import CatalogOffer

if TYPE_CHECKING:
    from .repository import OfferRepository


class OfferQuery:
    """Fluent builder that accumulates WHERE clauses and compiles to SQL on terminal call.

    Filters are chainable and additive (AND). Terminal methods execute the query
    asynchronously via the repository's ThreadPoolRunner.
    """

    def __init__(self, db: sqlite3.Connection, repo: OfferRepository) -> None:
        self._db = db
        self._repo = repo
        self._clauses: list[str] = []
        self._params: list[Any] = []
        self._clause_params: list[tuple[str, Any | None]] = []
        self._order_by: str | None = None
        self._limit: int | None = None

    # ── chainable filters ─────────────────────────────────────

    def _add_filter(self, clause: str, param: Any) -> None:
        self._clauses.append(clause)
        self._params.append(param)
        self._clause_params.append((clause, param))

    def accelerator(self, name: str) -> OfferQuery:
        """Filter by accelerator name (exact match, e.g. ``"A100"``)."""
        self._add_filter("accelerator_name = ?", name)
        return self

    def provider(self, name: str) -> OfferQuery:
        """Filter by provider id (e.g. ``"aws"``, ``"vastai"``)."""
        self._add_filter("provider = ?", name)
        return self

    def region(self, region: str) -> OfferQuery:
        """Filter by region (exact match)."""
        self._add_filter("region = ?", region)
        return self

    def architecture(self, arch: str) -> OfferQuery:
        """Filter by accelerator architecture (e.g. ``"Hopper"``, ``"Ampere"``)."""
        self._add_filter("architecture = ?", arch)
        return self

    def manufacturer(self, name: str) -> OfferQuery:
        """Filter by manufacturer (e.g. ``"NVIDIA"``, ``"AMD"``)."""
        self._add_filter("manufacturer = ?", name)
        return self

    def accelerator_memory(self, min_gb: float, max_gb: float | None = None) -> OfferQuery:
        """Filter by accelerator memory in GB. Single arg = minimum, two args = range."""
        if max_gb is None:
            self._add_filter("accelerator_memory_gb >= ?", min_gb)
        else:
            self._clauses.append("accelerator_memory_gb >= ? AND accelerator_memory_gb <= ?")
            self._params.extend([min_gb, max_gb])
            self._clause_params.append(("accelerator_memory_range", (min_gb, max_gb)))
        return self

    def vcpus(self, exact_or_min: float, max_vcpus: float | None = None) -> OfferQuery:
        """Filter by vCPUs. Single arg = exact match, two args = range."""
        if max_vcpus is None:
            self._add_filter("vcpus = ?", exact_or_min)
        else:
            self._clauses.append("vcpus >= ? AND vcpus <= ?")
            self._params.extend([exact_or_min, max_vcpus])
            self._clause_params.append(("vcpus_range", (exact_or_min, max_vcpus)))
        return self

    def memory(self, exact_or_min: float, max_gb: float | None = None) -> OfferQuery:
        """Filter by system memory in GB. Single arg = exact match, two args = range."""
        if max_gb is None:
            self._add_filter("memory_gb = ?", exact_or_min)
        else:
            self._clauses.append("memory_gb >= ? AND memory_gb <= ?")
            self._params.extend([exact_or_min, max_gb])
            self._clause_params.append(("memory_range", (exact_or_min, max_gb)))
        return self

    def accelerator_count(self, exact_or_min: float, max_count: float | None = None) -> OfferQuery:
        """Filter by accelerator count. Single arg = exact match, two args = range."""
        if max_count is None:
            self._add_filter("accelerator_count = ?", exact_or_min)
        else:
            self._clauses.append("accelerator_count >= ? AND accelerator_count <= ?")
            self._params.extend([exact_or_min, max_count])
            self._clause_params.append(("accelerator_count_range", (exact_or_min, max_count)))
        return self

    def cpu_only(self) -> OfferQuery:
        """Only CPU instances (no accelerator)."""
        self._clauses.append("accelerator_name = ''")
        self._clause_params.append(("cpu_only", True))
        return self

    def allocation(self, strategy: str) -> OfferQuery:
        """Filter by allocation strategy.

        ``"spot"`` requires spot pricing.  ``"on-demand"`` requires on-demand
        pricing.  ``"spot-if-available"`` and ``"cheapest"`` accept any offer.
        """
        match strategy:
            case "spot":
                self._clauses.append("spot_price IS NOT NULL")
            case "on-demand":
                self._clauses.append("on_demand_price IS NOT NULL")
        return self

    def spot(self) -> OfferQuery:
        """Only offers with spot pricing."""
        return self.allocation("spot")

    def on_demand(self) -> OfferQuery:
        """Only offers with on-demand pricing."""
        return self.allocation("on-demand")

    def max_price(self, usd_hr: float) -> OfferQuery:
        """Filter by maximum price per hour (spot preferred, falls back to on-demand)."""
        self._clauses.append("COALESCE(spot_price, on_demand_price) <= ?")
        self._params.append(usd_hr)
        return self

    def where(self, clause: str, *params: Any) -> OfferQuery:
        """Append a raw SQL WHERE clause (e.g. ``"accelerator_name LIKE ?", "H%"``)."""
        self._clauses.append(clause)
        self._params.extend(params)
        return self

    # ── ordering and limiting ─────────────────────────────────

    def order_by(self, expr: str) -> OfferQuery:
        """Set the ORDER BY expression for this query."""
        self._order_by = expr
        return self

    def limit(self, n: int) -> OfferQuery:
        """Set the maximum number of rows to return."""
        self._limit = n
        return self

    # ── terminals ─────────────────────────────────────────────

    @overload
    async def cheapest(self) -> CatalogOffer | None: ...
    @overload
    async def cheapest(self, n: int) -> list[CatalogOffer]: ...

    async def cheapest(self, n: int | None = None) -> CatalogOffer | list[CatalogOffer] | None:
        """Return the cheapest offer(s), ordered by lowest available price.

        Without arguments returns a single ``CatalogOffer | None``.
        With ``n`` returns the ``n`` cheapest as a list.
        """
        order = "COALESCE(spot_price, on_demand_price) ASC NULLS LAST"
        if n is None:
            rows = await self._execute(order_by=order, limit=1)
            return rows[0] if rows else None
        return await self._execute(order_by=order, limit=n)

    async def all(self) -> list[CatalogOffer]:
        """Execute the query and return all matching offers."""
        return await self._execute()

    async def first(self) -> CatalogOffer | None:
        """Execute the query and return the first match, or ``None``."""
        rows = await self._execute(limit=1)
        return rows[0] if rows else None

    async def count(self) -> int:
        """Return the total number of matching offers (ignores limit/order)."""
        sql = "SELECT COUNT(*) FROM catalog"
        if self._clauses:
            sql += " WHERE " + " AND ".join(self._clauses)
        params = tuple(self._params)
        rows = self._db.execute(sql, params).fetchone()
        return rows[0] if rows else 0

    # ── internals ─────────────────────────────────────────────

    _FILTER_MAP: dict[str, str] = {
        "accelerator_name = ?": "accelerator",
        "accelerator_count = ?": "accelerator_count",
        "provider = ?": "provider",
        "vcpus = ?": "vcpus",
        "memory_gb = ?": "memory_gb",
        "region = ?": "region",
        "cpu_only": "cpu_only",
    }

    def _extract_filters(self) -> dict[str, Any]:
        """Extract structured filters from accumulated clauses."""
        filters: dict[str, Any] = {}
        for clause, param in self._clause_params:
            if key := self._FILTER_MAP.get(clause):
                filters[key] = param
        return filters

    async def _execute(self, *, order_by: str | None = None, limit: int | None = None) -> list[CatalogOffer]:
        effective_order = order_by or self._order_by
        effective_limit = limit or self._limit

        sql = "SELECT * FROM catalog"
        if self._clauses:
            sql += " WHERE " + " AND ".join(self._clauses)
        if effective_order:
            sql += f" ORDER BY {effective_order}"
        if effective_limit:
            sql += f" LIMIT {effective_limit}"
        params = tuple(self._params)
        rows = await self._repo._run_query(sql, params)
        if not rows:
            await self._repo._fetch_and_persist(self._extract_filters())
            rows = await self._repo._run_query(sql, params)
        return rows
