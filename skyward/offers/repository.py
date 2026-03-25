"""SQLite-backed GPU offer repository — persistent catalog on disk."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skyward.infra.threaded import ThreadPoolRunner

from .model import CatalogOffer
from .query import OfferQuery

_DEFAULT_DB_PATH = Path.home() / ".skyward" / "offers.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS accelerators (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    vram REAL NOT NULL,
    manufacturer TEXT NOT NULL DEFAULT '',
    architecture TEXT NOT NULL DEFAULT '',
    cuda_min TEXT NOT NULL DEFAULT '',
    cuda_max TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS specs (
    id TEXT PRIMARY KEY,
    accelerator_id TEXT REFERENCES accelerators(id),
    vcpus REAL NOT NULL,
    memory_gb REAL NOT NULL,
    cpu_architecture TEXT NOT NULL DEFAULT 'x86_64'
);

CREATE TABLE IF NOT EXISTS offers (
    provider TEXT NOT NULL,
    spec_id TEXT NOT NULL REFERENCES specs(id),
    accelerator_count REAL NOT NULL,
    instance_type TEXT NOT NULL,
    region TEXT NOT NULL,
    spot_price REAL,
    on_demand_price REAL,
    billing_unit TEXT NOT NULL DEFAULT 'hour',
    specific JSON
);

CREATE VIEW IF NOT EXISTS catalog AS
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

CREATE TABLE IF NOT EXISTS fetch_log (
    provider TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_offers_spec ON offers(spec_id);
CREATE INDEX IF NOT EXISTS idx_specs_accel ON specs(accelerator_id);
"""

_MARKETPLACE_PROVIDERS = frozenset({"vastai", "runpod", "tensordock", "jarvislabs"})
_TTL_MARKETPLACE = 3600
_TTL_CATALOG = 86400

_SUMMARY_SQL = """\
SELECT
    accelerator_name AS gpu,
    provider,
    COUNT(*) AS offers,
    ROUND(AVG(spot_price), 2) AS avg_spot,
    ROUND(AVG(on_demand_price), 2) AS avg_on_demand
FROM catalog
WHERE accelerator_name != ''
{where}
GROUP BY accelerator_name, provider
ORDER BY accelerator_name, avg_spot ASC NULLS LAST
"""


class OfferRepository:
    """SQLite-backed queryable catalog of GPU offers across all providers.

    Persists to ``~/.skyward/offers.db`` by default. Supports fluent query
    building via ``select()`` and raw SQL via ``query()``.
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
        *,
        db_path: Path | str = _DEFAULT_DB_PATH,
    ) -> OfferRepository:
        """Open (or create) the persistent offer database.

        Parameters
        ----------
        providers
            Provider instances for on-demand fetching.
        db_path
            Path to the SQLite database. Defaults to ``~/.skyward/offers.db``.
            Pass ``":memory:"`` for in-memory (tests).
        """
        runner = ThreadPoolRunner(workers=1)
        db = await asyncio.to_thread(_open_db, db_path)
        await asyncio.to_thread(_migrate_json, db)
        provider_map = {p.name: p for p in providers} if providers else {}
        return OfferRepository(db, providers=provider_map, runner=runner)

    # ── public query entry points ─────────────────────────────

    def select(self) -> OfferQuery:
        """Start a new query against the catalog."""
        return OfferQuery(self._db, self)

    def accelerator(self, name: str) -> OfferQuery:
        """Start a query filtered by GPU name."""
        return self.select().accelerator(name)

    def provider(self, name: str) -> OfferQuery:
        """Start a query filtered by provider id."""
        return self.select().provider(name)

    def region(self, region: str) -> OfferQuery:
        """Start a query filtered by region."""
        return self.select().region(region)

    def architecture(self, arch: str) -> OfferQuery:
        """Start a query filtered by GPU architecture."""
        return self.select().architecture(arch)

    def manufacturer(self, name: str) -> OfferQuery:
        """Start a query filtered by manufacturer."""
        return self.select().manufacturer(name)

    def cpu_only(self) -> OfferQuery:
        """Start a query for CPU-only instances (no accelerator)."""
        return self.select().cpu_only()

    # ── aggregation ───────────────────────────────────────────

    def summary(self, accelerator: str | None = None) -> list[dict[str, Any]]:
        """Return average pricing per GPU per provider.

        Parameters
        ----------
        accelerator
            Optional GPU name filter.
        """
        where = "AND accelerator_name = ?" if accelerator else ""
        sql = _SUMMARY_SQL.format(where=where)
        params = (accelerator,) if accelerator else ()
        rows = self._db.execute(sql, params).fetchall()
        return [dict(zip(row.keys(), row, strict=True)) for row in rows]

    # ── staleness ──────────────────────────────────────────────

    def stale_providers(self) -> list[str]:
        """Return provider names whose cached offers are expired."""
        rows = self._db.execute("SELECT provider, fetched_at FROM fetch_log").fetchall()
        last_fetch = {r[0]: datetime.fromisoformat(r[1]) for r in rows}
        now = datetime.now(UTC)
        stale = []
        for name in self._providers:
            fetched = last_fetch.get(name)
            if not fetched:
                stale.append(name)
                continue
            ttl = _TTL_MARKETPLACE if name in _MARKETPLACE_PROVIDERS else _TTL_CATALOG
            if (now - fetched).total_seconds() > ttl:
                stale.append(name)
        return stale

    async def refresh(self) -> int:
        """Fetch only stale providers. Returns count of new offers."""
        stale = self.stale_providers()
        if not stale:
            return 0

        from skyward.offers.conversion import _offer_from_runtime

        total = 0

        for name in stale:
            provider_instance = self._providers[name]
            feed_offers = []
            try:
                async for offer in provider_instance.offers():
                    feed_offers.append(_offer_from_runtime(offer, name))
            except Exception:
                continue

            if not feed_offers:
                continue

            self._persist_offers(feed_offers, name)
            total += len(feed_offers)

        return total

    # ── bulk fetch ────────────────────────────────────────────

    async def fetch_all(self) -> int:
        """Fetch offers from all configured providers and persist.

        Returns
        -------
        int
            Total number of offers fetched.
        """
        if not self._providers:
            return 0

        from skyward.offers.conversion import _offer_from_runtime

        total = 0

        for name, provider_instance in self._providers.items():
            feed_offers = []
            try:
                async for offer in provider_instance.offers():
                    feed_offers.append(_offer_from_runtime(offer, name))
            except Exception:
                continue

            if not feed_offers:
                continue

            self._persist_offers(feed_offers, name)
            total += len(feed_offers)

        return total

    # ── async query execution ─────────────────────────────────

    async def _run_query(self, sql: str, params: tuple[Any, ...]) -> list[CatalogOffer]:
        run_fn = self._runner.as_async(self._run_query_sync)
        return await run_fn(sql, params)

    def _run_query_sync(self, sql: str, params: tuple[Any, ...]) -> list[CatalogOffer]:
        rows = self._db.execute(sql, params).fetchall()
        return [CatalogOffer(**dict(zip(row.keys(), row, strict=True))) for row in rows]

    async def _fetch_and_persist(self, filters: dict[str, Any]) -> None:
        if not self._providers:
            return

        from skyward.offers.conversion import _offer_from_runtime

        provider_filter = filters.get("provider")

        targets = (
            {provider_filter: self._providers[provider_filter]}
            if provider_filter and provider_filter in self._providers
            else self._providers
        )

        for name, provider_instance in targets.items():
            feed_offers = []
            try:
                async for offer in provider_instance.offers():
                    feed_offers.append(_offer_from_runtime(offer, name))
            except Exception:
                continue

            if not feed_offers:
                continue

            self._persist_offers(feed_offers, name)

    def _persist_offers(self, feed_offers: list[Any], provider_name: str) -> None:
        from skyward.offers.feed import (
            _serialize_accelerator,
            _serialize_offer,
            _serialize_spec,
        )

        self._db.execute("DELETE FROM offers WHERE provider = ?", (provider_name,))

        for fo in feed_offers:
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
        self._db.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?, ?)",
            (provider_name, datetime.now(UTC).isoformat()),
        )
        self._db.commit()

    # ── raw SQL ───────────────────────────────────────────────

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[CatalogOffer]:
        """Execute raw SQL against the catalog database."""
        rows = self._db.execute(sql, params).fetchall()
        return [CatalogOffer(**dict(zip(row.keys(), row, strict=True))) for row in rows]

    # ── lifecycle ─────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._db.close()

    def __enter__(self) -> OfferRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ── private helpers ───────────────────────────────────────────


def _open_db(path: Path | str) -> sqlite3.Connection:
    """Open or create the SQLite database with idempotent schema."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), check_same_thread=False)
    db.executescript(_SCHEMA)
    return db


def _migrate_json(db: sqlite3.Connection) -> None:
    """One-time migration: import cached JSON files into the database if empty."""
    row = db.execute("SELECT COUNT(*) FROM offers").fetchone()
    if row[0] > 0:
        return

    from .feed import CATALOG_DIR

    accels_file = CATALOG_DIR / "accelerators.json"
    if accels_file.exists():
        for a in json.loads(accels_file.read_text()):
            db.execute(
                "INSERT OR IGNORE INTO accelerators VALUES (?, ?, ?, ?, ?, ?, ?)",
                (a["id"], a["name"], a["vram"], a.get("manufacturer", ""),
                 a.get("architecture", ""), a.get("cuda_min", ""), a.get("cuda_max", "")),
            )

    specs_file = CATALOG_DIR / "specs.json"
    if specs_file.exists():
        for s in json.loads(specs_file.read_text()):
            db.execute(
                "INSERT OR IGNORE INTO specs VALUES (?, ?, ?, ?, ?)",
                (s["id"], s["accelerator_id"], s["vcpus"], s["memory_gb"],
                 s.get("cpu_architecture", "x86_64")),
            )

    offers_dir = CATALOG_DIR / "offers"
    if offers_dir.exists():
        for pf in sorted(offers_dir.glob("*.json")):
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
