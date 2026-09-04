"""Every state change is an event, and the table says which events change which state.

The table is pure and needs no store. The store half needs one: that applying an
event writes the state and the event together, that saying again what is already
true writes nothing, and that a row moving under a write is read again rather than
overwritten.
"""

from pathlib import Path

import pytest

from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.tables import ComputeRow, EventRow
from skyward.shared import lifecycle
from skyward.shared.errors import IllegalTransitionError
from skyward.shared.events import (
    ComputeBound,
    ComputeDegraded,
    ComputeDeleted,
    ComputeDeleting,
    ComputeProvisioning,
    ComputeReady,
    ComputeReleaseFailed,
    CostEvent,
    Event,
)
from skyward.shared.schemas import ComputeCreate, ComputeSpec, ComputeState, Image, NodeBounds, ProviderRef, Spec, Worker

pytestmark = pytest.mark.local

SPEC = ComputeSpec(
    specs=(Spec(provider=ProviderRef(kind="fake"), accelerator="a100", accelerator_count=1),),
    nodes=NodeBounds(initial=2),
    image=Image(python="3.13"),
    worker=Worker(concurrency=1, executor="thread"),
)


def provisioning(ready: int = 0, total: int = 2) -> ComputeProvisioning:
    return ComputeProvisioning(compute="cmp_1", nodes_ready=ready, nodes_total=total, generation=1)


def ready(count: int = 2) -> ComputeReady:
    return ComputeReady(compute="cmp_1", nodes_ready=count, nodes_total=count, generation=1)


def describe_the_compute_table() -> None:
    @pytest.mark.parametrize(
        ("state", "event", "expected"),
        [
            ("requested", provisioning(), "provisioning"),
            ("requested", ready(), "ready"),
            ("provisioning", ready(), "ready"),
            ("ready", provisioning(1), "provisioning"),
            ("provisioning", ComputeDegraded(compute="cmp_1", error="boom"), "degraded"),
            ("ready", ComputeDegraded(compute="cmp_1", error="boom"), "degraded"),
            ("degraded", provisioning(), "provisioning"),
            ("degraded", ready(), "ready"),
            ("ready", ComputeDeleting(compute="cmp_1", nodes_ready=2, nodes_total=2), "deleting"),
            ("degraded", ComputeDeleting(compute="cmp_1", nodes_ready=0, nodes_total=2), "deleting"),
            ("deleting", ComputeDeleted(compute="cmp_1"), "deleted"),
        ],
    )
    def every_arrow_of_the_diagram_leads_where_it_says(state: ComputeState, event: Event, expected: ComputeState) -> None:
        assert lifecycle.compute(state, event) == expected

    @pytest.mark.parametrize(
        ("state", "event"),
        [
            ("deleted", provisioning()),
            ("deleted", ready()),
            ("deleted", ComputeDeleting(compute="cmp_1", nodes_ready=0, nodes_total=0)),
            ("deleting", ready()),
            ("deleting", provisioning()),
            ("ready", ComputeDeleted(compute="cmp_1")),
        ],
    )
    def an_arrow_the_diagram_does_not_have_is_refused(state: ComputeState, event: Event) -> None:
        with pytest.raises(IllegalTransitionError):
            lifecycle.compute(state, event)

    def saying_where_it_already_is_answers_the_same_state() -> None:
        assert lifecycle.compute("ready", ready(3)) == "ready"
        assert lifecycle.compute("deleted", ComputeDeleted(compute="cmp_1")) == "deleted"

    def a_fact_moves_nothing() -> None:
        bound = ComputeBound(compute="cmp_1", offer="off_1", instance_type="a100.x", region=None, markets=("spot",))

        assert lifecycle.compute("ready", bound) is None
        assert lifecycle.compute("deleting", ComputeReleaseFailed(compute="cmp_1", error="later")) is None
        assert lifecycle.leads(bound) is None

    def the_client_reads_the_destination_without_the_origin() -> None:
        assert lifecycle.leads(ready()) == "ready"
        assert lifecycle.leads(ComputeDeleted(compute="cmp_1")) == "deleted"


def describe_applying_an_event_to_the_store() -> None:
    async def a_transition_writes_the_state_and_the_event_together(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)

        assert await computes.apply(compute, provisioning()) is True
        assert await computes.apply(compute, ready()) is True

        assert (await computes.get(compute)).status.state == "ready"
        assert await _recorded(compute) == ["compute.created", "compute.provisioning", "compute.ready"]

    async def the_same_pass_again_writes_the_counts_and_records_nothing(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)
        await computes.apply(compute, ready(2))

        assert await computes.apply(compute, ready(3)) is False

        assert (await computes.get(compute)).status.nodes_ready == 3
        assert await _recorded(compute) == ["compute.created", "compute.ready"]

    async def degrading_twice_is_said_once(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)

        assert await computes.apply(compute, ComputeDegraded(compute=compute, error="boom")) is True
        assert await computes.apply(compute, ComputeDegraded(compute=compute, error="boom")) is False

        assert (await _recorded(compute)).count("compute.degraded") == 1

    async def a_fact_is_recorded_and_the_state_stays(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)
        before = await computes.get(compute)

        await computes.apply(compute, ComputeBound(compute=compute, offer="off_1", instance_type="a100.x", region=None, markets=("spot",)))

        after = await computes.get(compute)
        assert after.status.state == before.status.state
        assert after.revision == before.revision
        assert (await _recorded(compute))[-1] == "compute.bound"

    async def an_illegal_event_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)

        with pytest.raises(IllegalTransitionError):
            await computes.apply(compute, ComputeDeleted(compute=compute))

        assert (await computes.get(compute)).status.state == "requested"
        assert await _recorded(compute) == ["compute.created"]

    async def deleting_is_said_once_however_many_times_it_is_asked(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)
        revision = (await computes.get(compute)).revision

        await computes.delete(compute, revision, "once")
        await computes.delete(compute, revision, "once")

        assert (await computes.get(compute)).status.state == "deleting"
        assert (await _recorded(compute)).count("compute.deleting") == 1

    async def a_row_that_moved_under_the_write_is_read_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        computes, compute = await _given(tmp_path)
        reads = 0
        original = computes._row

        async def renewed_underneath(ref: str) -> ComputeRow:
            """The row as apply reads it — and, the first time, a lease renewal landing right after."""
            nonlocal reads
            row = await original(ref)
            reads += 1
            if reads == 1:
                await ComputeRow.update({ComputeRow.revision: ComputeRow.revision + 1}).where(ComputeRow.id == ref).run()
            return row

        monkeypatch.setattr(computes, "_row", renewed_underneath)

        assert await computes.apply(compute, ready()) is True

        assert reads == 2, "the first write lost to the renewal and the row was read again"
        assert (await computes.get(compute)).status.state == "ready"
        assert (await _recorded(compute)).count("compute.ready") == 1

    async def the_stream_sees_the_transition_after_it_is_committed(tmp_path: Path) -> None:
        events = EventStore()
        computes = ComputeStore(events)
        await connect(tmp_path / "skyward.sqlite")
        compute, _ = await computes.create(ComputeCreate(spec=SPEC), idempotency_key="given")
        await computes.apply(compute.id, ready())

        names = []
        async for _, name, _ in events.stream(None, compute.id, None, None):
            names.append(name)
            if name == "compute.ready":
                break

        assert names == ["compute.created", "compute.ready"]


def describe_the_client_fold() -> None:
    async def a_pool_waits_on_the_state_the_events_lead_to() -> None:
        from skyward.core.view import ComputeView, observe

        view = ComputeView(id="cmp_1")
        for event in (provisioning(), ready(), ComputeDeleting(compute="cmp_1", nodes_ready=2, nodes_total=2), ComputeDeleted(compute="cmp_1")):
            view = observe(view, event)

        assert view.state == "deleted"

    async def a_gauge_does_not_move_the_state() -> None:
        from datetime import UTC, datetime

        from skyward.core.view import ComputeView, observe

        view = observe(ComputeView(id="cmp_1", state="ready"), CostEvent(compute="cmp_1", cost=1.0, nodes=2, at=datetime.now(UTC)))

        assert view.state == "ready"

    async def a_replay_folded_over_a_hydrated_view_does_not_raise() -> None:
        from skyward.core.view import ComputeView, observe

        view = observe(ComputeView(id="cmp_1", state="ready"), provisioning())

        assert view.state == "provisioning"


async def _given(tmp_path: Path) -> tuple[ComputeStore, str]:
    await connect(tmp_path / "skyward.sqlite")
    computes = ComputeStore(EventStore())
    compute, _ = await computes.create(ComputeCreate(spec=SPEC), idempotency_key="given")
    return computes, compute.id


async def _recorded(compute: str) -> list[str]:
    rows = await EventRow.select(EventRow.type).where(EventRow.compute_id == compute).order_by(EventRow.sequence)
    return [row["type"] for row in rows]
