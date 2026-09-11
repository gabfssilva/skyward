from __future__ import annotations

import asyncio
import hashlib
import re
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import datetime

from skyward.server.persistence.db import transaction
from skyward.server.persistence.store import after, digest, now
from skyward.server.persistence.tables import BlobRow, ChunkRow, FunctionRow
from skyward.shared.errors import HashMismatchError, NotFoundError
from skyward.shared.observability import logger
from skyward.shared.schemas import Function, Page

logger = logger.bind(component="blobs")

CODEC = "cloudpickle+lz4"

CACHE_ITEM = 1024 * 1024
"""The largest blob the read cache keeps."""

CACHE_BYTES = 64 * 1024 * 1024
"""The most bytes the read cache holds before it forgets the least recently read."""


class BlobStore:
    """Content addressed by its hash.

    Writing is idempotent for free: the same bytes have the same name, so a client
    that retries an upload is not uploading a second time, it is discovering the
    first one is already there. That is also why nothing here needs a revision or
    a lock — content that could change would not be content-addressed.

    The insert says so to the database as well. Two callers uploading the same
    bytes at the same time both find it missing and both write, and a conflict
    there is the same non-event as the check that missed it.

    The bytes are kept as chunks cut where the content says, each named by the hash
    of its own bytes, so content shared between blobs — a large argument sent again
    with small differences — is stored once. A manifest commits with its chunks in
    one transaction, so no blob ever names a chunk that is not stored.
    """

    def __init__(self) -> None:
        self._legacy: bool | None = None
        self._recent: OrderedDict[str, bytes] = OrderedDict()
        self._recent_bytes = 0

    async def _draining(self) -> bool:
        if self._legacy is None:
            rows = await BlobRow.raw("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'blobs_legacy'").run()
            self._legacy = bool(rows)
        return self._legacy

    async def exists(self, sha256: str) -> bool:
        if await BlobRow.exists().where(BlobRow.sha256 == sha256):
            return True
        return await self._draining() and bool(await BlobRow.raw("SELECT 1 FROM blobs_legacy WHERE sha256 = {}", sha256).run())

    async def put(self, sha256: str, blob: bytes) -> bool:
        if (actual := await digest(blob)) != sha256:
            raise HashMismatchError(f"blob hashes to {actual}, not {sha256}", expected=sha256, actual=actual)

        if await self.exists(sha256):
            return False

        hashed = await _offloaded(len(blob), _hashed, blob)
        chunks = await _missing(hashed)
        async with transaction():
            await _insert_chunks(chunks)
            await _insert_manifests([BlobRow(sha256=sha256, size_bytes=len(blob), chunks=_manifest(hashed), created_at=now())])
        return True

    async def get(self, sha256: str) -> bytes:
        if (cached := self._recent.get(sha256)) is not None:
            self._recent.move_to_end(sha256)
            return cached

        blob = await self._read(sha256)
        if len(blob) <= CACHE_ITEM:
            self._recent[sha256] = blob
            self._recent_bytes += len(blob)
            while self._recent_bytes > CACHE_BYTES:
                _, evicted = self._recent.popitem(last=False)
                self._recent_bytes -= len(evicted)
        return blob

    async def _read(self, sha256: str) -> bytes:
        row = await BlobRow.select(BlobRow.chunks).where(BlobRow.sha256 == sha256).first()
        if row is None:
            if await self._draining() and (legacy := await BlobRow.raw("SELECT data FROM blobs_legacy WHERE sha256 = {}", sha256).run()):
                return legacy[0]["data"]
            raise NotFoundError(f"no such blob: {sha256}")

        manifest: bytes = row["chunks"]
        order = [manifest[offset : offset + 32].hex() for offset in range(0, len(manifest), 32)]
        distinct = list(dict.fromkeys(order))
        found: dict[str, bytes] = {}
        for batch in _batches(distinct):
            rows = await ChunkRow.select(ChunkRow.sha256, ChunkRow.data).where(ChunkRow.sha256.is_in(list(batch)))
            found.update((chunk["sha256"], chunk["data"]) for chunk in rows)

        if len(found) < len(distinct):
            raise RuntimeError(f"blob {sha256} names a chunk that is not stored")
        return await _offloaded(sum(len(data) for data in found.values()), _joined, order, found)

    async def store(self, blob: bytes) -> str:
        """Put content whose hash the caller has not bothered to compute."""
        sha256 = await digest(blob)
        await self.put(sha256, blob)
        return sha256

    async def rechunk(self) -> None:
        """Convert the whole blobs in ``blobs_legacy`` into chunks, then drop the table.

        Runs in the background from the daemon's start. Each batch commits its chunks
        and manifests and deletes the legacy rows it converted in one transaction, so it
        is restartable: a daemon stopped halfway resumes on its next start. When nothing
        is left the table is dropped and the file vacuumed to reclaim the space.
        """
        if not await self._draining():
            return

        started = time.monotonic()
        [totals] = await BlobRow.raw("SELECT count(*) AS count, coalesce(sum(length(data)), 0) AS bytes FROM blobs_legacy").run()
        logger.info("rechunking {} legacy blobs ({} bytes)", totals["count"], totals["bytes"])

        converted = 0
        while rows := await BlobRow.raw("SELECT sha256, data, created_at FROM blobs_legacy ORDER BY rowid LIMIT 16").run():
            manifests: list[BlobRow] = []
            chunks: dict[str, bytes] = {}
            for row in rows:
                data: bytes = row["data"]
                created_at: datetime = row["created_at"]
                hashed = await _offloaded(len(data), _hashed, data)
                chunks.update(await _missing([(sha, piece) for sha, piece in hashed if sha not in chunks]))
                manifests.append(BlobRow(sha256=row["sha256"], size_bytes=len(data), chunks=_manifest(hashed), created_at=created_at))

            names = [row["sha256"] for row in rows]
            placeholders = ", ".join("{}" for _ in names)
            async with transaction():
                await _insert_chunks(list(chunks.items()))
                await _insert_manifests(manifests)
                await BlobRow.raw(f"DELETE FROM blobs_legacy WHERE sha256 IN ({placeholders})", *names).run()
            converted += len(rows)

        self._legacy = False
        for statement in ("DROP TABLE blobs_legacy", "VACUUM", "PRAGMA wal_checkpoint(TRUNCATE)"):
            await BlobRow.raw(statement).run()

        [stored] = await BlobRow.raw("SELECT coalesce(sum(length(data)), 0) AS bytes FROM chunks").run()
        logger.info(
            "rechunked {} blobs: {} legacy bytes, {} chunk bytes stored, in {:.1f}s",
            converted,
            totals["bytes"],
            stored["bytes"],
            time.monotonic() - started,
        )


class FunctionStore:
    """The code, registered once and named by every task that calls it.

    A function is a blob plus the little that is worth knowing about it without
    unpickling it. Separating the two is what lets a thousand tasks over an hour
    ship a hash instead of a pickle.
    """

    def __init__(self, blobs: BlobStore) -> None:
        self._blobs = blobs

    async def exists(self, sha256: str) -> bool:
        return await FunctionRow.exists().where(FunctionRow.sha256 == sha256)

    async def get(self, sha256: str) -> Function:
        row = await FunctionRow.objects().where(FunctionRow.sha256 == sha256).first()
        if row is None:
            raise NotFoundError(f"no such function: {sha256}")
        return _to_function(row)

    async def list(self, cursor: str | None, limit: int) -> Page[Function]:
        query = FunctionRow.objects()
        if pivot := await after(cursor, FunctionRow.sha256, FunctionRow.created_at):
            query = query.where(FunctionRow.created_at > pivot)

        rows = await query.order_by(FunctionRow.created_at).limit(limit)
        items = tuple(_to_function(row) for row in rows)
        return Page(items=items, next_cursor=items[-1].sha256 if items and len(items) == limit else None)

    async def register(self, sha256: str, blob: bytes, name: str | None) -> tuple[Function, bool]:
        if await self.exists(sha256):
            return await self.get(sha256), False

        await self._blobs.put(sha256, blob)
        await FunctionRow.insert(
            FunctionRow(
                sha256=sha256,
                size_bytes=len(blob),
                codec=CODEC,
                name=name,
                created_at=now(),
            ),
        ).on_conflict(action="DO NOTHING").run()

        return await self.get(sha256), True


def _to_function(row: FunctionRow) -> Function:
    return Function(
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        codec=row.codec,
        created_at=row.created_at,
        name=row.name,
    )


_BOUNDARY = re.compile(rb"\xa7\x3c|\x5e\xd1")
_MIN = 2 * 1024
_MAX = 256 * 1024
_OFFLOAD = 64 * 1024
_BATCH = 500


async def _offloaded[**P, R](size: int, work: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    return work(*args, **kwargs) if size < _OFFLOAD else await asyncio.to_thread(work, *args, **kwargs)


async def _missing(hashed: Sequence[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    pieces = dict(hashed)
    stored: set[str] = set()
    for batch in _batches(list(pieces)):
        stored.update(row["sha256"] for row in await ChunkRow.select(ChunkRow.sha256).where(ChunkRow.sha256.is_in(list(batch))))
    missing = [(sha, piece) for sha, piece in pieces.items() if sha not in stored]
    return await _offloaded(sum(len(piece) for _, piece in missing), _packed, missing)


async def _insert_chunks(chunks: Sequence[tuple[str, bytes]]) -> None:
    for batch in _batches(chunks):
        await ChunkRow.insert(*(ChunkRow(sha256=sha, data=data) for sha, data in batch)).on_conflict(action="DO NOTHING").run()


async def _insert_manifests(manifests: Sequence[BlobRow]) -> None:
    for batch in _batches(manifests):
        await BlobRow.insert(*batch).on_conflict(action="DO NOTHING").run()


def _batches[T](items: Sequence[T]) -> list[Sequence[T]]:
    return [items[offset : offset + _BATCH] for offset in range(0, len(items), _BATCH)]


def _manifest(hashed: Sequence[tuple[str, bytes]]) -> bytes:
    return b"".join(bytes.fromhex(sha) for sha, _ in hashed)


def _split(blob: bytes) -> list[bytes]:
    pieces: list[bytes] = []
    start = 0
    for match in _BOUNDARY.finditer(blob):
        end = match.end()
        if end - start < _MIN:
            continue
        while end - start > _MAX:
            pieces.append(blob[start : start + _MAX])
            start += _MAX
        pieces.append(blob[start:end])
        start = end
    while len(blob) - start > _MAX:
        pieces.append(blob[start : start + _MAX])
        start += _MAX
    if start < len(blob) or not pieces:
        pieces.append(blob[start:])
    return pieces


def _hashed(blob: bytes) -> list[tuple[str, bytes]]:
    return [(hashlib.sha256(piece).hexdigest(), piece) for piece in _split(blob)]


def _packed(pieces: Sequence[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    return [(sha, zlib.compress(piece, 6)) for sha, piece in pieces]


def _joined(order: Sequence[str], found: dict[str, bytes]) -> bytes:
    return b"".join(zlib.decompress(found[sha]) for sha in order)
