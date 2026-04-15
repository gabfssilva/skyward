"""Blob service: atomic content writes + DB row persistence.

Bytes are written to a temporary file, moved into their sharded destination
via ``os.replace``, and the ``blobs`` row is written inside the same
``store.tx()`` transaction so DB metadata and filesystem content commit as
one unit. Path layout is ``{root}/{xx}/{id:08x}.bin`` where ``xx`` is the
first two hex chars of the zero-padded id — bounds per-directory entry
count to 256 buckets.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from skyward.server.host.domain import BlobId, BlobKind
from skyward.server.host.store import Store


class BlobEvicted(Exception):  # noqa: N818
    """Raised by :meth:`Blobs.read` when the blob row has ``evicted_at`` set."""


@dataclass(frozen=True, slots=True)
class Blobs:
    """Persist blob bytes atomically alongside their DB metadata row.

    Parameters
    ----------
    store
        Backing :class:`Store` for metadata.
    root
        Filesystem root that owns the sharded blob tree.
    """

    store: Store
    root: Path

    async def put(self, data: bytes, *, kind: BlobKind) -> BlobId:
        """Write ``data`` atomically and insert the matching ``blobs`` row.

        Parameters
        ----------
        data
            Raw bytes to persist.
        kind
            Either ``"payload"`` or ``"result"``.

        Returns
        -------
        BlobId
            The autoincrement id of the inserted row.
        """
        sha = hashlib.sha256(data).hexdigest()
        size = len(data)
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.root, prefix=".blob-", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            async with self.store.tx() as tx:
                cursor = await tx.execute(
                    "INSERT INTO blobs (path, size, sha256, kind, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (str(tmp_path), size, sha, kind, time.time()),
                )
                blob_id = int(cursor.lastrowid or 0)
                final = self._path_for(blob_id)
                final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp_path, final)
                await tx.execute(
                    "UPDATE blobs SET path = ? WHERE id = ?",
                    (str(final), blob_id),
                )
            return blob_id
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    async def read(self, id: BlobId) -> bytes:
        """Return the bytes for *id*.

        Raises
        ------
        FileNotFoundError
            If no row exists for *id*.
        BlobEvicted
            If the row's ``evicted_at`` column is populated.
        """
        row = await self.store.get_blob(id)
        match row:
            case None:
                raise FileNotFoundError(f"Blob {id} not found")
            case blob if blob.evicted_at is not None:
                raise BlobEvicted(f"Blob {id} evicted at {blob.evicted_at}")
            case blob:
                return Path(blob.path).read_bytes()

    async def evict(self, id: BlobId) -> None:
        """Mark *id* evicted and best-effort unlink the on-disk file."""
        row = await self.store.get_blob(id)
        match row:
            case None:
                return
            case blob if blob.evicted_at is not None:
                return
            case blob:
                await self.store.evict_blob(id)
                Path(blob.path).unlink(missing_ok=True)

    async def gc(self, *, ttl: timedelta) -> int:
        """Evict live blobs whose parents are terminal older than ``ttl``.

        A blob is eligible when at least one parent execution referencing it
        (via ``payload_blob``) is terminal with ``finished_at`` older than
        ``now - ttl``, **and** no non-terminal execution references it, **and**
        no non-terminal result references it (via ``result_blob``).

        Parameters
        ----------
        ttl
            Minimum age past a parent's ``finished_at`` before eviction.

        Returns
        -------
        int
            Number of blobs evicted in this pass.
        """
        cutoff = time.time() - ttl.total_seconds()
        async with self.store.tx() as tx:
            rows = await tx.fetch_all(
                """
                SELECT b.id AS id, b.path AS path
                FROM blobs b
                WHERE b.evicted_at IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM task_executions e
                      WHERE e.payload_blob = b.id
                        AND (
                            e.status_tag NOT IN ('succeeded','failed','interrupted','cancelled')
                            OR e.finished_at IS NULL
                            OR e.finished_at >= ?
                        )
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM task_results r
                      WHERE r.result_blob = b.id
                        AND (
                            r.status_tag NOT IN ('succeeded','failed','interrupted')
                            OR r.finished_at IS NULL
                            OR r.finished_at >= ?
                        )
                  )
                  AND (
                      EXISTS (
                          SELECT 1 FROM task_executions e
                          WHERE e.payload_blob = b.id
                            AND e.status_tag IN ('succeeded','failed','interrupted','cancelled')
                            AND e.finished_at IS NOT NULL
                            AND e.finished_at < ?
                      )
                      OR EXISTS (
                          SELECT 1 FROM task_results r
                          WHERE r.result_blob = b.id
                            AND r.status_tag IN ('succeeded','failed','interrupted')
                            AND r.finished_at IS NOT NULL
                            AND r.finished_at < ?
                      )
                  )
                """,
                (cutoff, cutoff, cutoff, cutoff),
            )
            now_ts = time.time()
            for row in rows:
                await tx.execute(
                    "UPDATE blobs SET evicted_at = ? WHERE id = ?",
                    (now_ts, int(row["id"])),
                )
        for row in rows:
            Path(str(row["path"])).unlink(missing_ok=True)
        return len(rows)

    def _path_for(self, id: BlobId) -> Path:
        hex_id = f"{id:08x}"
        return self.root / hex_id[:2] / f"{hex_id}.bin"

    async def gc_orphans(self) -> int:
        """Reconcile on-disk blobs with live DB rows on startup.

        - Files inside ``root`` with no matching live ``blobs`` row are
          deleted (they are leftovers from a crash between ``os.replace``
          and the commit).
        - Rows whose on-disk path is missing get ``evicted_at`` set so
          the client sees a ``410 Gone`` rather than a 500.

        Non-terminal executions / results keep their blob pinned —
        eviction there would drop the bytes a still-running worker
        expects. Returns the total number of rows / files reconciled.
        """
        if not self.root.exists():
            return 0
        async with self.store.tx() as tx:
            rows = await tx.fetch_all(
                "SELECT id, path, evicted_at FROM blobs",
            )

        live_paths: set[str] = set()
        missing_ids: list[int] = []
        for row in rows:
            path = str(row["path"])
            if row["evicted_at"] is not None:
                continue
            live_paths.add(path)
            if not Path(path).exists():
                missing_ids.append(int(row["id"]))

        removed = 0
        if missing_ids:
            async with self.store.tx() as tx:
                now_ts = time.time()
                for blob_id in missing_ids:
                    await tx.execute(
                        "UPDATE blobs SET evicted_at = ? WHERE id = ?",
                        (now_ts, blob_id),
                    )
            removed += len(missing_ids)

        for file_path in self.root.glob("**/*.bin"):
            if str(file_path) not in live_paths:
                file_path.unlink(missing_ok=True)
                removed += 1
        return removed


__all__ = ["BlobEvicted", "Blobs"]
