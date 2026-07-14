from __future__ import annotations

from datetime import timedelta
from typing import Any

import msgspec

from skyward2.application.errors import LeaseHeldError, NotFoundError, RevisionConflictError
from skyward2.application.provider import Binding
from skyward2.persistence.store import after, digest, ident, now, once, packed, unpacked
from skyward2.persistence.tables import ComputeRow, GenerationRow
from skyward2.protocol.schemas import (
    Compute,
    ComputeCreate,
    ComputeSpec,
    ComputeSpecPatch,
    ComputeState,
    ComputeStatus,
    Error,
    Generation,
    GenerationCreate,
    Lease,
    LeaseClaim,
    Page,
)

LIVE: tuple[ComputeState, ...] = ("requested", "provisioning", "recovering", "ready", "degraded", "deleting")
"""The states in which a compute still has something owed to it."""


class ComputeStore:
    async def create(self, body: ComputeCreate, idempotency_key: str) -> tuple[Compute, bool]:
        async def insert() -> str:
            compute = ident("cmp")
            await ComputeRow(
                id=compute,
                name=body.name,
                spec=packed(body.spec),
                status_state="requested",
                created_at=now(),
            ).save().run()
            await self.freeze(compute, 1, body.spec)
            return compute

        compute_id, created = await once("compute.create", idempotency_key, body, insert)
        return await self.get(compute_id), created

    async def get(self, ref: str) -> Compute:
        return _to_compute(await self._row(ref))

    async def list(self, cursor: str | None, limit: int, state: ComputeState | None, owned: bool | None) -> Page[Compute]:
        query = ComputeRow.objects()

        if pivot := await after(ComputeRow, cursor):
            query = query.where(ComputeRow.created_at > pivot)
        if state:
            query = query.where(ComputeRow.status_state == state)
        if owned:
            query = query.where(ComputeRow.lease_owner.is_not_null() & (ComputeRow.lease_expires_at > now()))
        elif owned is False:
            query = query.where(ComputeRow.lease_owner.is_null() | (ComputeRow.lease_expires_at <= now()))

        rows = await query.order_by(ComputeRow.created_at).limit(limit)
        items = tuple(_to_compute(row) for row in rows)
        return Page(items=items, next_cursor=items[-1].id if len(items) == limit else None)

    async def patch(self, ref: str, body: ComputeSpecPatch, expected_revision: int) -> Compute:
        """Resize in place.

        A new size is a new definition, so it is a new generation — but the
        infrastructure is not replaced, it is grown or drained. That is the whole
        reason ``nodes`` is the only field the API lets through here: everything
        else would mean throwing the machines away, and that has to be asked for.
        """
        row = await self.checked(ref, expected_revision)
        spec = msgspec.structs.replace(unpacked(row.spec, ComputeSpec), nodes=body.nodes)

        row.spec = packed(spec)
        row.revision += 1
        row.generation += 1
        await row.save().run()

        await self.freeze(row.id, row.generation, spec)
        return await self.get(row.id)

    async def delete(self, ref: str, expected_revision: int, idempotency_key: str) -> Compute:
        """Write the intent, and nothing else.

        Nothing is torn down here. The reconciler destroys the infrastructure and
        only calls the compute ``deleted`` once the provider confirms it is gone —
        a store that marked it deleted on the way out would be a store that lies
        about the bill.
        """

        async def mark() -> str:
            row = await self.checked(ref, expected_revision)
            spec = msgspec.structs.replace(unpacked(row.spec, ComputeSpec), desired="deleted")

            row.spec = packed(spec)
            row.revision += 1
            row.status_state = "deleting"
            await row.save().run()
            return row.id

        compute_id, _ = await once("compute.delete", idempotency_key, None, mark)
        return await self.get(compute_id)

    async def claim_lease(self, ref: str, claim: LeaseClaim) -> Lease:
        """Take ownership, if it is free or already ours.

        Zero owners is a legitimate state, and a temporary one — a daemon
        restarting, a script killed. What must never happen is two, because a
        compute's live SSH connections and casty client exist in exactly one
        process.
        """
        row = await self._row(ref)
        expires = now() + timedelta(seconds=claim.ttl_seconds)

        held = row.lease_owner is not None and row.lease_expires_at is not None and row.lease_expires_at > now()
        if held and row.lease_owner != claim.owner:
            raise LeaseHeldError(f"compute {row.id} is owned by {row.lease_owner}", owner=row.lease_owner)

        row.lease_owner = claim.owner
        row.lease_expires_at = expires
        row.revision += 1
        await row.save().run()

        return Lease(owner=claim.owner, expires_at=expires)

    async def release_lease(self, ref: str) -> None:
        row = await self._row(ref)
        row.lease_owner = None
        row.lease_expires_at = None
        row.revision += 1
        await row.save().run()

    async def binding(self, compute_id: str) -> Binding:
        return unpacked((await self._row(compute_id)).binding, dict[str, Any])

    async def bind(self, compute_id: str, binding: Binding) -> None:
        """Record what the provider built, before anything is built on top of it."""
        row = await self._row(compute_id)
        row.binding = packed(binding)
        await row.save().run()

    async def observe(self, compute_id: str, status: ComputeStatus) -> None:
        """Write what was seen. Only the reconciler calls this."""
        await ComputeRow.update({
            ComputeRow.status_state: status.state,
            ComputeRow.status_observed_generation: status.observed_generation,
            ComputeRow.status_nodes_ready: status.nodes_ready,
            ComputeRow.status_nodes_total: status.nodes_total,
            ComputeRow.status_drift: packed(status.drift),
            ComputeRow.status_error: packed(status.last_error) if status.last_error else None,
            ComputeRow.revision: ComputeRow.revision + 1,
        }).where(ComputeRow.id == compute_id).run()

    async def live(self) -> tuple[str, ...]:
        """Every compute that still owes something — the sweep's worklist."""
        rows = await ComputeRow.select(ComputeRow.id).where(ComputeRow.status_state.is_in(list(LIVE)))
        return tuple(row["id"] for row in rows)

    async def _row(self, ref: str) -> ComputeRow:
        row = await ComputeRow.objects().where((ComputeRow.id == ref) | (ComputeRow.name == ref)).first()
        if row is None:
            raise NotFoundError(f"no such compute: {ref}")
        return row

    async def checked(self, ref: str, expected_revision: int) -> ComputeRow:
        row = await self._row(ref)
        if row.revision != expected_revision:
            raise RevisionConflictError(
                f"compute {row.id} is at revision {row.revision}, not {expected_revision}",
                current=row.revision,
            )
        return row

    async def freeze(self, compute_id: str, number: int, spec: ComputeSpec) -> None:
        await GenerationRow(
            id=f"{compute_id}:{number}",
            compute_id=compute_id,
            number=number,
            spec=packed(spec),
            hash=digest(msgspec.json.encode(spec)),
            applied=False,
            created_at=now(),
        ).save().run()


class GenerationStore:
    def __init__(self, computes: ComputeStore) -> None:
        self._computes = computes

    async def list(self, compute: str) -> Page[Generation]:
        rows = await GenerationRow.objects().where(GenerationRow.compute_id == compute).order_by(GenerationRow.number)
        return Page(items=tuple(_to_generation(row) for row in rows))

    async def get(self, compute: str, number: int) -> Generation:
        row = await GenerationRow.objects().where(
            (GenerationRow.compute_id == compute) & (GenerationRow.number == number),
        ).first()
        if row is None:
            raise NotFoundError(f"no such generation: {compute}:{number}")
        return _to_generation(row)

    async def create(self, compute: str, body: GenerationCreate, expected_revision: int, idempotency_key: str) -> Generation:
        """Replace the infrastructure: same compute, new definition.

        Without a ``source`` this applies whatever drift is pending; with one, it
        goes back to that generation's definition. Either way the current machines
        are condemned — which is why this is a separate verb from ``PATCH`` and
        not an inference from it.
        """
        async def branch() -> str:
            row = await self._computes.checked(compute, expected_revision)
            spec = (await self.get(compute, body.source)).spec if body.source else unpacked(row.spec, ComputeSpec)

            row.spec = packed(spec)
            row.revision += 1
            row.generation += 1
            await row.save().run()

            await self._computes.freeze(row.id, row.generation, spec)
            return f"{row.id}:{row.generation}"

        generation_id, _ = await once("generation.create", idempotency_key, body, branch)
        compute_id, number = generation_id.rsplit(":", 1)
        return await self.get(compute_id, int(number))

    async def apply(self, compute: str, number: int) -> None:
        """Mark a generation as the one the machines actually reflect."""
        await GenerationRow.update({GenerationRow.applied: True}).where(
            (GenerationRow.compute_id == compute) & (GenerationRow.number == number),
        ).run()


def _to_compute(row: ComputeRow) -> Compute:
    return Compute(
        id=row.id,
        name=row.name,
        revision=row.revision,
        generation=row.generation,
        spec=unpacked(row.spec, ComputeSpec),
        status=ComputeStatus(
            state=msgspec.convert(row.status_state, ComputeState),
            observed_generation=row.status_observed_generation,
            nodes_ready=row.status_nodes_ready,
            nodes_total=row.status_nodes_total,
            drift=unpacked(row.status_drift, tuple[str, ...]),
            last_error=unpacked(row.status_error, Error) if row.status_error else None,
        ),
        lease=Lease(owner=row.lease_owner, expires_at=row.lease_expires_at),
        created_at=row.created_at,
    )


def _to_generation(row: GenerationRow) -> Generation:
    return Generation(
        number=row.number,
        spec=unpacked(row.spec, ComputeSpec),
        hash=row.hash,
        created_at=row.created_at,
        applied=row.applied,
    )
