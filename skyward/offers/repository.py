"""SQLite-backed GPU offer repository — loads cached catalog into an in-memory database."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from skyward.infra.threaded import ThreadPoolRunner

from .model import CatalogOffer
from .query import OfferQuery

_SCHEMA = """\
CREATE TABLE accelerators (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    vram REAL NOT NULL,
    manufacturer TEXT NOT NULL DEFAULT '',
    architecture TEXT NOT NULL DEFAULT '',
    cuda_min TEXT NOT NULL DEFAULT '',
    cuda_max TEXT NOT NULL DEFAULT ''
);

CREATE TABLE specs (
    id TEXT PRIMARY KEY,
    accelerator_id TEXT REFERENCES accelerators(id),
    vcpus REAL NOT NULL,
    memory_gb REAL NOT NULL,
    cpu_architecture TEXT NOT NULL DEFAULT 'x86_64'
);

CREATE TABLE offers (
    provider TEXT NOT NULL,
    spec_id TEXT NOT NULL REFERENCES specs(id),
    accelerator_count INTEGER NOT NULL,
    instance_type TEXT NOT NULL,
    region TEXT NOT NULL,
    spot_price REAL,
    on_demand_price REAL,
    billing_unit TEXT NOT NULL DEFAULT 'hour',
    specific TEXT
);

CREATE VIEW catalog AS
SELECT
    o.provider,
    o.instance_type,
    o.region,
    COALESCE(a.name, '') AS accelerator_name,
    o.accelerator_count,
    COALESCE(a.vram, 0) AS accelerator_memory_gb,
    COALESCE(a.manufacturer, '') AS manufacturer,
    COALESCE(a.architecture, '') AS architecture,
    COALESCE(a.cuda_min, '') AS cuda_min,
    COALESCE(a.cuda_max, '') AS cuda_max,
    s.vcpus,
    s.memory_gb,
    s.cpu_architecture,
    o.spot_price,
    o.on_demand_price,
    o.billing_unit,
    o.specific
FROM offers o
JOIN specs s ON o.spec_id = s.id
LEFT JOIN accelerators a ON s.accelerator_id = a.id;

CREATE INDEX idx_offers_spec ON offers(spec_id);
CREATE INDEX idx_specs_accel ON specs(accelerator_id);
"""


class OfferRepository:
    """SQLite-backed queryable catalog of GPU offers across all providers.

    Pre-load cached offer data into an in-memory SQLite database.
    Support fluent query building and raw SQL against a ``catalog`` VIEW.

    Examples
    --------
    >>> repo = await sky.offers([sky.AWS(), sky.VastAI()])

    >>> # Fluent API
    >>> cheapest = repo.accelerator("A100").spot().cheapest()
    >>> top5 = repo.architecture("Hopper").accelerator_count(4).cheapest(5)

    >>> # Raw SQL
    >>> offers = repo.query(
    ...     "SELECT * FROM catalog WHERE accelerator_name LIKE 'H%' ORDER BY spot_price"
    ... )
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        providers: dict[str, Any] | None = None,
        runner: ThreadPoolRunner | None = None,
    ) -> None:
        self._db = db
        self._db.row_factory = sqlite3.Row
        self._providers = providers or {}
        self._runner = runner or ThreadPoolRunner(workers=1)

    @staticmethod
    async def create(
        providers: Sequence[Any] | None = None,
    ) -> OfferRepository:
        """Load catalog into an in-memory SQLite database.

        Starts with whatever is cached on disk and fetches on-demand
        when queries return no results.

        Parameters
        ----------
        providers
            Provider instances to use for on-demand fetching.
        """
        from .feed import CATALOG_DIR

        runner = ThreadPoolRunner(workers=1)
        db_repo = await asyncio.to_thread(_load, CATALOG_DIR, None)
        provider_map = {p.name: p for p in providers} if providers else {}
        return OfferRepository(db_repo._db, providers=provider_map, runner=runner)

    # ── filter entry points ──────────────────────────────────

    def accelerator(self, name: str) -> OfferQuery:
        """Start a query filtered by GPU name."""
        return OfferQuery(self._db, self).accelerator(name)

    def provider(self, name: str) -> OfferQuery:
        """Start a query filtered by provider id."""
        return OfferQuery(self._db, self).provider(name)

    def region(self, region: str) -> OfferQuery:
        """Start a query filtered by region."""
        return OfferQuery(self._db, self).region(region)

    def architecture(self, arch: str) -> OfferQuery:
        """Start a query filtered by GPU architecture."""
        return OfferQuery(self._db, self).architecture(arch)

    def manufacturer(self, name: str) -> OfferQuery:
        """Start a query filtered by manufacturer."""
        return OfferQuery(self._db, self).manufacturer(name)

    def cpu_only(self) -> OfferQuery:
        """Start a query for CPU-only instances (no accelerator)."""
        return OfferQuery(self._db, self).cpu_only()

    # ── async query execution ───────────────────────────────

    async def _run_query(self, sql: str, params: tuple[Any, ...]) -> list[CatalogOffer]:
        run_fn = self._runner.as_async(self._run_query_sync)
        return await run_fn(sql, params)

    def _run_query_sync(self, sql: str, params: tuple[Any, ...]) -> list[CatalogOffer]:
        rows = self._db.execute(sql, params).fetchall()
        return [CatalogOffer(**dict(zip(row.keys(), row, strict=True))) for row in rows]

    async def _fetch_and_persist(self, filters: dict[str, Any]) -> None:
        if not self._providers:
            return

        from skyward.api.spec import Image, PoolSpec
        from skyward.offers.conversion import _offer_from_runtime
        from skyward.offers.feed import (
            _serialize_accelerator,
            _serialize_offer,
            _serialize_spec,
        )

        provider_filter = filters.get("provider")
        accel_name = filters.get("accelerator")

        accel: Accelerator | None = None
        if accel_name:
            from skyward.accelerators.spec import Accelerator

            accel = Accelerator(name=accel_name)

        spec = PoolSpec(
            nodes=1,
            accelerator=accel,
            region=filters.get("region", ""),
            vcpus=filters.get("vcpus"),
            memory_gb=filters.get("memory_gb"),
            image=Image(),
        )

        targets = (
            {provider_filter: self._providers[provider_filter]}
            if provider_filter and provider_filter in self._providers
            else self._providers
        )

        for name, provider in targets.items():
            feed_offers = []
            try:
                async for offer in provider.offers(spec):
                    feed_offers.append(_offer_from_runtime(offer, name))
            except Exception:
                continue

            if not feed_offers:
                continue

            def _persist(offers: list[Any] = feed_offers, provider_name: str = name) -> None:
                for fo in offers:
                    accel_id: str | None = None
                    if fo.spec.accelerator.name:
                        a = _serialize_accelerator(fo.spec.accelerator)
                        accel_id = a["id"]
                        self._db.execute(
                            "INSERT OR IGNORE INTO accelerators VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (a["id"], a["name"], a["vram"], a.get("manufacturer", ""),
                             a.get("architecture", ""), a.get("cuda_min", ""), a.get("cuda_max", "")),
                        )

                    s = _serialize_spec(fo.spec)
                    self._db.execute(
                        "INSERT OR IGNORE INTO specs VALUES (?, ?, ?, ?, ?)",
                        (s["id"], accel_id, s["vcpus"], s["memory_gb"],
                         s.get("cpu_architecture", "x86_64")),
                    )

                    o = _serialize_offer(fo)
                    specific = o.get("specific")
                    self._db.execute(
                        "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (provider_name, o["spec_id"], o["accelerator_count"],
                         o["instance_type"], o["region"],
                         o.get("spot_price"), o.get("on_demand_price"),
                         o.get("billing_unit", "hour"),
                         json.dumps(specific) if specific is not None else None),
                    )
                self._db.commit()

            persist_fn = self._runner.as_async(_persist)
            await persist_fn()

    # ── raw SQL ──────────────────────────────────────────────

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[CatalogOffer]:
        """Execute raw SQL against the catalog database.

        The pre-joined ``catalog`` VIEW is available with columns:
        provider, instance_type, region, accelerator_name, accelerator_count,
        accelerator_memory_gb, manufacturer, architecture, cuda_min, cuda_max,
        vcpus, memory_gb, spot_price, on_demand_price, billing_unit, specific.
        """
        rows = self._db.execute(sql, params).fetchall()
        return [CatalogOffer(**dict(zip(row.keys(), row, strict=True))) for row in rows]

    # ── lifecycle ────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._db.close()

    def __enter__(self) -> OfferRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ── private helpers ──────────────────────────────────────────


def _load(catalog_dir: Path, providers: list[str] | None = None) -> OfferRepository:
    """Build an in-memory SQLite DB from normalized catalog JSON files."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.executescript(_SCHEMA)

    accels_file = catalog_dir / "accelerators.json"
    if accels_file.exists():
        for a in json.loads(accels_file.read_text()):
            db.execute(
                "INSERT OR IGNORE INTO accelerators VALUES (?, ?, ?, ?, ?, ?, ?)",
                (a["id"], a["name"], a["vram"], a.get("manufacturer", ""),
                 a.get("architecture", ""), a.get("cuda_min", ""), a.get("cuda_max", "")),
            )

    specs_file = catalog_dir / "specs.json"
    if specs_file.exists():
        for s in json.loads(specs_file.read_text()):
            db.execute(
                "INSERT OR IGNORE INTO specs VALUES (?, ?, ?, ?, ?)",
                (s["id"], s["accelerator_id"], s["vcpus"], s["memory_gb"],
                 s.get("cpu_architecture", "x86_64")),
            )

    offers_dir = catalog_dir / "offers"
    if offers_dir.exists():
        provider_files = (
            [offers_dir / f"{p}.json" for p in providers if (offers_dir / f"{p}.json").exists()]
            if providers is not None
            else sorted(offers_dir.glob("*.json"))
        )
        for pf in provider_files:
            provider = pf.stem
            for o in json.loads(pf.read_text()):
                specific = o.get("specific")
                db.execute(
                    "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (provider, o["spec_id"], o["accelerator_count"], o["instance_type"],
                     o["region"], o.get("spot_price"), o.get("on_demand_price"),
                     o.get("billing_unit", "hour"),
                     json.dumps(specific) if specific is not None else None),
                )

    db.commit()
    return OfferRepository(db)
