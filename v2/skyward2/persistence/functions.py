from __future__ import annotations

from skyward2.application.errors import HashMismatchError, NotFoundError
from skyward2.persistence.store import after, digest, now
from skyward2.persistence.tables import BlobRow, FunctionRow
from skyward2.protocol.schemas import Function, Page

CODEC = "cloudpickle+lz4"


class BlobStore:
    """Content addressed by its hash.

    Writing is idempotent for free: the same bytes have the same name, so a client
    that retries an upload is not uploading a second time, it is discovering the
    first one is already there. That is also why nothing here needs a revision or
    a lock — content that could change would not be content-addressed.
    """

    async def exists(self, sha256: str) -> bool:
        return await BlobRow.exists().where(BlobRow.sha256 == sha256)

    async def put(self, sha256: str, blob: bytes) -> bool:
        if (actual := digest(blob)) != sha256:
            raise HashMismatchError(f"blob hashes to {actual}, not {sha256}", expected=sha256, actual=actual)

        if await self.exists(sha256):
            return False

        await BlobRow(sha256=sha256, data=blob, created_at=now()).save().run()
        return True

    async def get(self, sha256: str) -> bytes:
        row = await BlobRow.objects().where(BlobRow.sha256 == sha256).first()
        if row is None:
            raise NotFoundError(f"no such blob: {sha256}")
        return row.data

    async def store(self, blob: bytes) -> str:
        """Put content whose hash the caller has not bothered to compute."""
        sha256 = digest(blob)
        await self.put(sha256, blob)
        return sha256


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
        if pivot := await after(FunctionRow, cursor, column="sha256"):
            query = query.where(FunctionRow.created_at > pivot)

        rows = await query.order_by(FunctionRow.created_at).limit(limit)
        items = tuple(_to_function(row) for row in rows)
        return Page(items=items, next_cursor=items[-1].sha256 if len(items) == limit else None)

    async def register(self, sha256: str, blob: bytes, name: str | None) -> tuple[Function, bool]:
        written = await self._blobs.put(sha256, blob)
        if not written and await self.exists(sha256):
            return await self.get(sha256), False

        await FunctionRow(
            sha256=sha256,
            size_bytes=len(blob),
            codec=CODEC,
            name=name,
            created_at=now(),
        ).save().run()

        return await self.get(sha256), True


def _to_function(row: FunctionRow) -> Function:
    return Function(
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        codec=row.codec,
        created_at=row.created_at,
        name=row.name,
    )
