"""SQLite-backed GPU offer repository — loads the JSON catalog into an in-memory database."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from .model import CatalogOffer
from .query import OfferQuery

_DEFAULT_CATALOG_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "catalog"

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
    on_demand_price REAL
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
    o.on_demand_price
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
    async def create(catalog_dir: Path | None = None) -> OfferRepository:
        """Load catalog JSONs into an in-memory SQLite database."""
        return await asyncio.to_thread(_load, catalog_dir or _DEFAULT_CATALOG_DIR)

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
        memory_gb, spot_price, on_demand_price.
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


def _load(catalog_dir: Path) -> OfferRepository:
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.executescript(_SCHEMA)

    with open(catalog_dir / "accelerators.json") as f:
        db.executemany(
            "INSERT INTO accelerators VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(a["id"], a["name"], a["vram"], a["manufacturer"], a["architecture"], a["cuda_min"], a["cuda_max"]) for a in json.load(f)],
        )

    with open(catalog_dir / "specs.json") as f:
        db.executemany(
            "INSERT INTO specs VALUES (?, ?, ?, ?)",
            [(s["id"], s["accelerator_id"], s["vcpus"], s["memory_gb"]) for s in json.load(f)],
        )

    offers_dir = catalog_dir / "offers"
    for provider_file in sorted(offers_dir.glob("*.json")):
        provider = provider_file.stem
        with open(provider_file) as f:
            db.executemany(
                "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (provider, o["spec_id"], o["accelerator_count"], o["instance_type"], o["region"], o.get("spot_price"), o.get("on_demand_price"))
                    for o in json.load(f)
                ],
            )

    db.commit()
    return OfferRepository(db)
