"""SQLite-backed GPU offer repository — loads cached catalog into an in-memory database."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    accelerator_id TEXT NOT NULL REFERENCES accelerators(id),
    vcpus REAL NOT NULL,
    memory_gb REAL NOT NULL
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
    a.name AS gpu,
    o.accelerator_count AS gpu_count,
    a.vram,
    a.manufacturer,
    a.architecture,
    a.cuda_min,
    a.cuda_max,
    s.vcpus,
    s.memory_gb,
    o.spot_price,
    o.on_demand_price,
    o.billing_unit,
    o.specific
FROM offers o
JOIN specs s ON o.spec_id = s.id
JOIN accelerators a ON s.accelerator_id = a.id;

CREATE INDEX idx_offers_spec ON offers(spec_id);
CREATE INDEX idx_specs_accel ON specs(accelerator_id);
"""


class OfferRepository:
    """SQLite-backed queryable view over the GPU offer catalog.

    Usage::

        repo = await OfferRepository.create()

        offer = repo.accelerator("A100").vram(80).spot().cheapest()

        offers = repo.architecture("Hopper").gpus(4).cheapest(5)

        offers = repo.query("SELECT * FROM catalog WHERE gpu LIKE 'H%' ORDER BY spot_price")
    """

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        self._db.row_factory = sqlite3.Row

    @staticmethod
    async def create(providers: Sequence[str] | None = None) -> OfferRepository:
        """Load catalog into an in-memory SQLite database.

        Refreshes stale provider caches before loading.  When *providers*
        is ``None``, loads all cached provider files on disk.

        Parameters
        ----------
        providers
            Which providers to ensure are fresh and load.
        """
        from .feed import cached_provider_paths, ensure_fresh, provider_path

        await ensure_fresh(providers=providers)

        files = (
            [provider_path(p) for p in providers if provider_path(p).exists()]
            if providers is not None
            else cached_provider_paths()
        )
        return await asyncio.to_thread(_load, files)

    # ── filter entry points ──────────────────────────────────

    def accelerator(self, name: str) -> OfferQuery:
        """Start a query filtered by GPU name."""
        return OfferQuery(self._db).accelerator(name)

    def provider(self, name: str) -> OfferQuery:
        """Start a query filtered by provider id."""
        return OfferQuery(self._db).provider(name)

    def region(self, region: str) -> OfferQuery:
        """Start a query filtered by region."""
        return OfferQuery(self._db).region(region)

    def architecture(self, arch: str) -> OfferQuery:
        """Start a query filtered by GPU architecture."""
        return OfferQuery(self._db).architecture(arch)

    def manufacturer(self, name: str) -> OfferQuery:
        """Start a query filtered by manufacturer."""
        return OfferQuery(self._db).manufacturer(name)

    # ── raw SQL ──────────────────────────────────────────────

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[CatalogOffer]:
        """Execute raw SQL against the catalog database.

        The pre-joined ``catalog`` VIEW is available with columns:
        provider, instance_type, region, gpu, gpu_count, vram,
        manufacturer, architecture, cuda_min, cuda_max, vcpus,
        memory_gb, spot_price, on_demand_price, billing_unit, specific.
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


def _load(provider_files: list[Path]) -> OfferRepository:
    """Build an in-memory SQLite DB from denormalized per-provider JSON files."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.executescript(_SCHEMA)

    seen_accels: set[str] = set()
    seen_specs: set[str] = set()

    for path in provider_files:
        provider = path.stem
        try:
            raw_offers: list[dict[str, Any]] = json.loads(path.read_text())
        except Exception:
            continue

        for o in raw_offers:
            gpu = o.get("gpu", "")
            vram = o.get("gpu_vram", 0.0)
            vcpus = o.get("vcpus", 0.0)
            memory_gb = o.get("memory_gb", 0.0)

            accel_id = hashlib.md5(f"{gpu}:{vram}".encode()).hexdigest()[:12]
            spec_id = hashlib.md5(
                f"{accel_id}:{vcpus}:{memory_gb}".encode(),
            ).hexdigest()[:12]

            if accel_id not in seen_accels:
                seen_accels.add(accel_id)
                db.execute(
                    "INSERT OR IGNORE INTO accelerators VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        accel_id, gpu, vram,
                        o.get("manufacturer", ""),
                        o.get("architecture", ""),
                        o.get("cuda_min", ""),
                        o.get("cuda_max", ""),
                    ),
                )

            if spec_id not in seen_specs:
                seen_specs.add(spec_id)
                db.execute(
                    "INSERT OR IGNORE INTO specs VALUES (?, ?, ?, ?)",
                    (spec_id, accel_id, vcpus, memory_gb),
                )

            specific = o.get("specific")
            db.execute(
                "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    provider,
                    spec_id,
                    o.get("gpu_count", 1),
                    o.get("instance_type", ""),
                    o.get("region", ""),
                    o.get("spot_price"),
                    o.get("on_demand_price"),
                    o.get("billing_unit", "hour"),
                    json.dumps(specific) if specific is not None else None,
                ),
            )

    db.commit()
    return OfferRepository(db)
