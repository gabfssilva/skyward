"""The three things every store does the same way."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import msgspec
from msgspec import Struct
from piccolo.table import Table

from skyward2.application.errors import IdempotencyConflictError, NotFoundError
from skyward2.persistence.tables import IdempotencyRow


def now() -> datetime:
    return datetime.now(UTC)


def ident(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def packed(value: object) -> str:
    """A column's worth of JSON.

    Text, not a parsed object: piccolo's ``JSONB`` accepts either, and letting it
    parse on the way out only to hand msgspec a dict it has to walk again is a
    conversion nobody asked for.
    """
    return msgspec.json.encode(value).decode()


def unpacked[T](raw: str, kind: type[T]) -> T:
    return msgspec.json.decode(raw.encode(), type=kind)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


async def once(scope: str, key: str, body: Struct | None, produce: Callable[[], Awaitable[str]]) -> tuple[str, bool]:
    """Do it, or say what doing it produced the first time.

    Parameters
    ----------
    scope : str
        What the key is a key *to*. The same ``Idempotency-Key`` may legitimately
        be reused across endpoints by a client that generates one per request; it
        is only a replay if it is the same key on the same operation.
    body : Struct | None
        The request. Its fingerprint is what separates a retry from a collision:
        the same key with a different request is not a caller being careful, it is
        a caller with a bug, and it gets a 409 instead of a second resource.
    produce : Callable[[], Awaitable[str]]
        Called only when the key is new. Returns the id of what it created.

    Returns
    -------
    tuple[str, bool]
        The resource id, and whether this call was the one that created it.
    """
    scoped = f"{scope}:{key}"
    fingerprint = digest(msgspec.json.encode(body) if body is not None else b"")

    if seen := await IdempotencyRow.objects().where(IdempotencyRow.key == scoped).first():
        if seen.fingerprint != fingerprint:
            raise IdempotencyConflictError(f"key already used with a different request: {key}")
        return seen.resource_id, False

    resource_id = await produce()
    await IdempotencyRow(
        key=scoped,
        fingerprint=fingerprint,
        resource_id=resource_id,
        created_at=now(),
    ).save().run()

    return resource_id, True


async def after(table: type[Table], cursor: str | None, column: str = "id") -> datetime | None:
    """Where a page picks up: the moment the last item of the previous one was created."""
    if cursor is None:
        return None

    created = table._meta.get_column_by_name("created_at")
    rows = await table.select(created).where(table._meta.get_column_by_name(column) == cursor)
    if not rows:
        raise NotFoundError(f"no such cursor: {cursor}")

    return rows[0]["created_at"]
