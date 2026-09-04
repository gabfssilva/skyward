from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import msgspec
from msgspec import Struct, field
from piccolo.columns import Column

from skyward.server.persistence.db import transaction
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.store import after, digest, ident, now, once, packed, unpacked
from skyward.server.persistence.tables import ComputeRow, GenerationRow
from skyward.shared import lifecycle
from skyward.shared.errors import ComputeNotResizableError, LeaseHeldError, NameTakenError, NotFoundError, RevisionConflictError
from skyward.shared.events import (
    ComputeBound,
    ComputeCreated,
    ComputeDegraded,
    ComputeDeleting,
    ComputeDeletionFailed,
    Event,
    GenerationApplied,
    GenerationCreated,
    LeaseClaimed,
    LeaseReleased,
)
from skyward.shared.provider import Binding
from skyward.shared.schemas import (
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
    Market,
    Offer,
    Page,
)
from skyward.shared.tls import Authority
from skyward.worker import plugins

LIVE: tuple[ComputeState, ...] = ("requested", "provisioning", "ready", "degraded", "deleting")
"""The states in which a compute still has something owed to it."""

ATTEMPTS = 3
"""How many times :meth:`ComputeStore.apply` re-reads a row that moved under it before giving up.

The window is the one between reading the row and the conditional write, and
what lands in it is a lease renewal or a binding — writes about other columns.
Three is generous for a loop that never contends with itself for long.
"""

STATUS: Mapping[str, Column] = {
    "nodes_ready": ComputeRow.status_nodes_ready,
    "nodes_total": ComputeRow.status_nodes_total,
    "generation": ComputeRow.status_observed_generation,
}
"""An event's field, and the status column it is the observation of."""


class Infrastructure(Struct, frozen=True):
    """Everything about a compute that is real, costs money, and is not in the spec.

    None of it is in the API's ``Compute``, and all of it has to survive the
    daemon: the provider account the machines were bought from, the offer they
    were bought as, whatever the adapter needs to find them again, and the key
    without which they are unreachable metal that keeps billing.
    """

    provider_id: str | None = None
    offer_id: str | None = None
    offer: Offer | None = None
    """The offer as it read when the compute was bound to it.

    A snapshot, not a reference: the offer cache is replaced wholesale on refresh,
    so by the time a machine is launched the row it was picked from may be gone —
    but the price it was picked at is what the machines are billed against.
    """
    binding: Binding = field(default_factory=dict)
    private_key: str | None = None
    authority: Authority | None = None
    """The certificate authority of this compute's cluster.

    Minted with the key, and null on a compute bound before there was one: the
    material a running worker was given cannot be changed under it, so a fleet
    that came up without an authority stays without one until it is replaced.
    """
    markets: tuple[Market, ...] = ()
    volumes: tuple[str, ...] = ()
    """The bootstrap phases that mount the compute's buckets, as rendered at bind time.

    Here and not on the spec for the same reason ``private_key`` is: they hold the
    credentials the provider minted, and the spec is served back by the compute API.
    """


class ComputeStore:
    """The computes, and the one way their state moves.

    Every write of ``status_state`` goes through :meth:`apply`, which consults the
    lifecycle table, writes the state the event leads to, and records the event in
    the same transaction. There is no way to move a compute without saying so, which
    is the property a watcher of the stream is promised.
    """

    def __init__(self, events: EventStore) -> None:
        self._events = events

    async def apply(self, event: Event) -> bool:
        """Apply one event to the compute it names, and say whether its state moved.

        The row is read fresh and written conditionally on the revision that read
        saw — an optimistic lock around exactly the read-decide-write, and no wider,
        because the revision is also bumped by writes that have nothing to do with
        the state (a lease renewal, a binding) and a caller holding an older one
        would conflict with those for no reason. A row that moved between the read
        and the write is read again; a transition is always computed against the
        state the row is actually in.

        Three outcomes, by what the table says and what the event carries. A move
        writes the state and the event together. A repeat records nothing: an event
        leading where the row already is (the reconciler says ``ready`` to a ready
        compute on every pass), or a fact whose every column the row already shows
        (the provider refusing to release with the same words as last tick). What a
        repeat carries that is new — a count — is still written. A fact with
        something to say is recorded, and touches no state.
        """
        projection = await _projected(event)
        attempts = ATTEMPTS

        while True:
            row = await self._status(event.compute)
            current = msgspec.convert(row["status_state"], ComputeState)
            state = lifecycle.compute(current, event)
            moved = state is not None and state != current
            columns = {column: value for column, value in projection.items() if row[column._meta.name] != value}
            repeat = not moved and (state is not None or (bool(projection) and not columns))
            live = None

            async with transaction():
                if moved or columns:
                    landed = await ComputeRow.update({
                        **columns,
                        ComputeRow.status_state: state or current,
                        ComputeRow.revision: row["revision"] + 1,
                    }).where((ComputeRow.id == row["id"]) & (ComputeRow.revision == row["revision"])).returning(ComputeRow.id).run()
                    if not landed:
                        attempts -= 1
                        if attempts:
                            continue
                        raise RevisionConflictError(f"compute {event.compute} kept moving while {type(event).__name__} was applied", current=row["revision"])
                if not repeat:
                    live = await self._events.append(event)

            if live is not None:
                self._events.deliver(live)
            return moved

    async def create(self, body: ComputeCreate, idempotency_key: str) -> tuple[Compute, bool]:
        """Write the definition down, and say plainly when the name is not free.

        The name is unique across every compute the store has ever held, deleted
        ones included, and the caller chose it. Left to the database that is an
        integrity error on the way out — a 500 for something the caller could have
        been told, and could fix by picking another name.
        """
        async def insert() -> str:
            if body.name and (taken := await ComputeRow.objects().where(ComputeRow.name == body.name).first()):
                raise NameTakenError(
                    f"the name {body.name!r} is compute {taken.id}'s, which is {taken.status_state}",
                    name=body.name,
                    compute=taken.id,
                )

            compute = ident("cmp")
            await ComputeRow(
                id=compute,
                name=body.name,
                spec=await packed(body.spec),
                status_state="requested",
                created_at=now(),
            ).save().run()
            await self.apply(ComputeCreated(compute=compute))
            await self._freeze(compute, 1, body.spec)
            return compute

        compute_id, created = await once("compute.create", idempotency_key, body, insert)
        return await self.get(compute_id), created

    async def get(self, ref: str) -> Compute:
        return await _to_compute(await self._row(ref))

    async def list(self, cursor: str | None, limit: int, state: ComputeState | None, owned: bool | None, live: bool | None) -> Page[Compute]:
        """Newest first.

        A compute's row outlives its machines, so what a daemon has the most of is
        history: paged from the oldest end, a page answers what a laptop ran months
        ago before it answers what is running now.
        """
        query = ComputeRow.objects()

        if pivot := await after(ComputeRow, cursor):
            query = query.where(ComputeRow.created_at < pivot)
        if state:
            query = query.where(ComputeRow.status_state == state)
        if live:
            query = query.where(ComputeRow.status_state.is_in(list(LIVE)))
        elif live is False:
            query = query.where(ComputeRow.status_state.not_in(list(LIVE)))
        if owned:
            query = query.where(ComputeRow.lease_owner.is_not_null() & (ComputeRow.lease_expires_at > now()))
        elif owned is False:
            query = query.where(ComputeRow.lease_owner.is_null() | (ComputeRow.lease_expires_at <= now()))

        rows = await query.order_by(ComputeRow.created_at, ascending=False).limit(limit)
        items = tuple([await _to_compute(row) for row in rows])
        return Page(items=items, next_cursor=items[-1].id if len(items) == limit else None)

    async def patch(self, ref: str, body: ComputeSpecPatch, expected_revision: int) -> Compute:
        """Resize in place.

        A new size is a new definition, so it is a new generation — but the
        infrastructure is not replaced, it is grown or drained. That is the whole
        reason ``nodes`` is the only field the API lets through here: everything
        else would mean throwing the machines away, and that has to be asked for.

        A compute running a collective is refused, because growing it is not
        something the machines can be told: the process group is formed on the first
        task and never formed again, so a rank arriving later blocks in a rendezvous
        the others have already left. Writing the same size back is not a resize and
        is left alone, so a retry of a request that landed still answers.
        """
        row = await self.checked(ref, expected_revision)
        current = await unpacked(row.spec, ComputeSpec)

        if body.nodes == current.nodes:
            return await _to_compute(row)
        if freezes := plugins.collective(current.plugins):
            raise ComputeNotResizableError(
                f"compute {row.id} runs {freezes}, a collective: its process group was formed with the ranks it "
                f"started with, and a machine added now would block in it",
                plugin=freezes,
            )

        await self.regenerate(row, msgspec.structs.replace(current, nodes=body.nodes))
        return await self.get(row.id)

    async def delete(self, ref: str, expected_revision: int, idempotency_key: str) -> Compute:
        """Write the intent, and say that it was written.

        Nothing is torn down here. The reconciler destroys the infrastructure and
        only calls the compute ``deleted`` once the provider confirms it is gone —
        a store that marked it deleted on the way out would be a store that lies
        about the bill. What is written is ``deleting``, through :meth:`apply`, so
        the answer to this request already says so and so does the stream.
        """

        async def mark() -> str:
            row = await self.checked(ref, expected_revision)
            spec = msgspec.structs.replace(await unpacked(row.spec, ComputeSpec), desired="deleted")

            row.spec = await packed(spec)
            row.revision += 1
            await row.save().run()
            await self.apply(ComputeDeleting(compute=row.id, nodes_ready=row.status_nodes_ready, nodes_total=row.status_nodes_total))
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

        renewal = held and row.lease_owner == claim.owner
        row.lease_owner = claim.owner
        row.lease_expires_at = expires
        row.revision += 1
        await row.save().run()
        if not renewal:
            await self.apply(LeaseClaimed(compute=row.id, owner=claim.owner))

        return Lease(owner=claim.owner, expires_at=expires)

    async def release_lease(self, ref: str) -> None:
        row = await self._row(ref)
        row.lease_owner = None
        row.lease_expires_at = None
        row.revision += 1
        await row.save().run()
        await self.apply(LeaseReleased(compute=row.id))

    async def infrastructure(self, compute_id: str) -> Infrastructure:
        row = await self._row(compute_id)
        return Infrastructure(
            provider_id=row.provider_id,
            offer_id=row.offer_id,
            offer=await unpacked(row.offer, Offer) if row.offer else None,
            binding=await unpacked(row.binding, dict[str, Any]),
            private_key=row.private_key,
            authority=await unpacked(row.authority, Authority) if row.authority else None,
            markets=await unpacked(row.markets, tuple[Market, ...]),
            volumes=await unpacked(row.volumes, tuple[str, ...]),
        )

    async def bind(self, compute_id: str, infrastructure: Infrastructure) -> bool:
        """Record what the provider built, before anything is built on top of it.

        Written before the machines are launched, and not after: a binding that is
        only in memory when the daemon dies is a network, a keypair and a security
        group that nobody will ever find again — and a key that is lost is a fleet
        that keeps billing and can never be logged into.

        The write is conditional on the key: a row that already carries a different
        private key was bound by somebody else — another daemon on the same file,
        racing — and overwriting it would strand every machine launched under the
        first key behind a lock the store no longer opens. The loser's write is a
        no-op, and the answer says which it was; only a binding that landed is said
        on the stream, with the offer it replaced when the compute moved.
        """
        if infrastructure.offer is None:
            raise ValueError("a binding names the offer the compute was bought as")

        guard = ComputeRow.private_key.is_null()
        if infrastructure.private_key is not None:
            guard = guard | (ComputeRow.private_key == infrastructure.private_key)

        before = await self._status(compute_id)
        landed = await ComputeRow.update({
            ComputeRow.provider_id: infrastructure.provider_id,
            ComputeRow.offer_id: infrastructure.offer_id,
            ComputeRow.offer: await packed(infrastructure.offer),
            ComputeRow.binding: await packed(infrastructure.binding),
            ComputeRow.private_key: infrastructure.private_key,
            ComputeRow.authority: await packed(infrastructure.authority) if infrastructure.authority else None,
            ComputeRow.markets: await packed(infrastructure.markets),
            ComputeRow.volumes: await packed(infrastructure.volumes),
            ComputeRow.revision: ComputeRow.revision + 1,
        }).where((ComputeRow.id == compute_id) & guard).returning(ComputeRow.id).run()
        if not landed:
            return False

        await self.apply(ComputeBound(
            compute=compute_id,
            offer=infrastructure.offer.id,
            instance_type=infrastructure.offer.instance_type,
            region=infrastructure.offer.region,
            markets=infrastructure.markets,
            previous=before["offer_id"],
        ))
        return True

    async def regenerate(self, row: ComputeRow, spec: ComputeSpec) -> int:
        """A new definition for the same compute: the row moves on, and the generation is frozen and said."""
        row.spec = await packed(spec)
        row.revision += 1
        row.generation += 1
        await row.save().run()
        await self._freeze(row.id, row.generation, spec)
        return row.generation

    async def live(self) -> tuple[str, ...]:
        """Every compute that still owes something — the sweep's worklist."""
        rows = await ComputeRow.select(ComputeRow.id).where(ComputeRow.status_state.is_in(list(LIVE)))
        return tuple(row["id"] for row in rows)

    async def checked(self, ref: str, expected_revision: int) -> ComputeRow:
        row = await self._row(ref)
        if row.revision != expected_revision:
            raise RevisionConflictError(
                f"compute {row.id} is at revision {row.revision}, not {expected_revision}",
                current=row.revision,
            )
        return row

    async def _row(self, ref: str) -> ComputeRow:
        row = await ComputeRow.objects().where((ComputeRow.id == ref) | (ComputeRow.name == ref)).first()
        if row is None:
            raise NotFoundError(f"no such compute: {ref}")
        return row

    async def _status(self, compute_id: str) -> dict[str, Any]:
        """The columns :meth:`apply` decides by — not the row, which carries the key and three JSON blobs."""
        decided_by = (ComputeRow.id, ComputeRow.status_state, ComputeRow.revision, ComputeRow.offer_id, ComputeRow.status_error, *STATUS.values())
        row = await ComputeRow.select(*decided_by).where(ComputeRow.id == compute_id).first()
        if row is None:
            raise NotFoundError(f"no such compute: {compute_id}")
        return row

    async def _freeze(self, compute_id: str, number: int, spec: ComputeSpec) -> None:
        await GenerationRow(
            id=f"{compute_id}:{number}",
            compute_id=compute_id,
            number=number,
            spec=await packed(spec),
            hash=await digest(msgspec.json.encode(spec)),
            applied=False,
            created_at=now(),
        ).save().run()
        await self.apply(GenerationCreated(compute=compute_id, number=number))


class GenerationStore:
    def __init__(self, computes: ComputeStore) -> None:
        self._computes = computes

    async def list(self, compute: str) -> Page[Generation]:
        rows = await GenerationRow.objects().where(GenerationRow.compute_id == compute).order_by(GenerationRow.number)
        return Page(items=tuple([await _to_generation(row) for row in rows]))

    async def get(self, compute: str, number: int) -> Generation:
        row = await GenerationRow.objects().where(
            (GenerationRow.compute_id == compute) & (GenerationRow.number == number),
        ).first()
        if row is None:
            raise NotFoundError(f"no such generation: {compute}:{number}")
        return await _to_generation(row)

    async def create(self, compute: str, body: GenerationCreate, expected_revision: int, idempotency_key: str) -> Generation:
        """Replace the infrastructure: same compute, new definition.

        Without a ``source`` this applies whatever drift is pending; with one, it
        goes back to that generation's definition. Either way the current machines
        are condemned — which is why this is a separate verb from ``PATCH`` and
        not an inference from it.
        """
        async def branch() -> str:
            row = await self._computes.checked(compute, expected_revision)
            spec = (await self.get(compute, body.source)).spec if body.source else await unpacked(row.spec, ComputeSpec)

            number = await self._computes.regenerate(row, spec)
            return f"{row.id}:{number}"

        generation_id, _ = await once("generation.create", idempotency_key, body, branch)
        compute_id, number = generation_id.rsplit(":", 1)
        return await self.get(compute_id, int(number))

    async def apply(self, compute: str, number: int) -> None:
        """Mark a generation as the one the machines actually reflect, and say so."""
        await GenerationRow.update({GenerationRow.applied: True}).where(
            (GenerationRow.compute_id == compute) & (GenerationRow.number == number),
        ).run()
        await self._computes.apply(GenerationApplied(compute=compute, number=number))


async def _projected(event: Event) -> dict[Column, Any]:
    """The status columns an event carries with it.

    The event is the observation and the columns are a projection of it: a count
    lands in the column of the same name, and a failure lands in ``status_error``.
    A transition that is not a failure clears the error, because the compute has
    demonstrably moved past whatever it was.
    """
    fields = msgspec.structs.asdict(event)
    columns: dict[Column, Any] = {column: fields[field] for field, column in STATUS.items() if field in fields}
    match event:
        case ComputeDegraded(error=error, code=code) | ComputeDeletionFailed(error=error, code=code):
            columns[ComputeRow.status_error] = await packed(Error(code=code, message=error, retryable=True))
        case _ if type(event) in lifecycle.COMPUTE:
            columns[ComputeRow.status_error] = None
    return columns


async def _to_compute(row: ComputeRow) -> Compute:
    return Compute(
        id=row.id,
        name=row.name,
        revision=row.revision,
        generation=row.generation,
        spec=await unpacked(row.spec, ComputeSpec),
        status=ComputeStatus(
            state=msgspec.convert(row.status_state, ComputeState),
            observed_generation=row.status_observed_generation,
            nodes_ready=row.status_nodes_ready,
            nodes_total=row.status_nodes_total,
            drift=await unpacked(row.status_drift, tuple[str, ...]),
            last_error=await unpacked(row.status_error, Error) if row.status_error else None,
        ),
        lease=Lease(owner=row.lease_owner, expires_at=row.lease_expires_at),
        created_at=row.created_at,
    )


async def _to_generation(row: GenerationRow) -> Generation:
    return Generation(
        number=row.number,
        spec=await unpacked(row.spec, ComputeSpec),
        hash=row.hash,
        created_at=row.created_at,
        applied=row.applied,
    )
