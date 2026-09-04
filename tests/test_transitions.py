"""Every state change is an event, and the table says which events change which state.

The table is pure and needs no store. The store half needs one: that applying an
event writes the state and the event together, that saying again what is already
true writes nothing, and that a row moving under a write is read again rather than
overwritten.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from skyward.core.view import ComputeView, observe
from skyward.server.application.machines import Machines
from skyward.server.application.mock import OFFER
from skyward.server.application.reconciler import Reconciler, Wakeup
from skyward.server.persistence.computes import ComputeStore, GenerationStore, Infrastructure
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.tables import ComputeRow, EventRow
from skyward.server.persistence.tasks import TaskStore
from skyward.shared import lifecycle
from skyward.shared.errors import IllegalTransitionError
from skyward.shared.events import (
    ComputeBound,
    ComputeDegraded,
    ComputeDeleted,
    ComputeDeleting,
    ComputeDeletionFailed,
    ComputeProvisioning,
    ComputeReady,
    CostEvent,
    Event,
)
from skyward.shared.schemas import ComputeState
from tests.conftest import given

pytestmark = pytest.mark.local


def provisioning(ready: int = 0, total: int = 2, compute: str = "cmp_1") -> ComputeProvisioning:
    return ComputeProvisioning(compute=compute, nodes_ready=ready, nodes_total=total, generation=1)


def ready(count: int = 2, compute: str = "cmp_1") -> ComputeReady:
    return ComputeReady(compute=compute, nodes_ready=count, nodes_total=count, generation=1)


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
            ("deleting", ComputeDegraded(compute="cmp_1", error="boom")),
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
        assert lifecycle.compute("deleting", ComputeDeletionFailed(compute="cmp_1", error="later")) is None
        assert lifecycle.leads(bound) is None

    def the_client_reads_the_destination_without_the_origin() -> None:
        assert lifecycle.leads(ready()) == "ready"
        assert lifecycle.leads(ComputeDeleted(compute="cmp_1")) == "deleted"


def describe_applying_an_event_to_the_store() -> None:
    async def a_transition_writes_the_state_and_the_event_together(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)

        assert await computes.apply(provisioning(compute=compute)) is True
        assert await computes.apply(ready(compute=compute)) is True

        assert (await computes.get(compute)).status.state == "ready"
        assert await _recorded(compute) == ["compute.created", "compute.generation.created", "compute.provisioning", "compute.ready"]

    async def the_same_pass_again_writes_the_counts_and_records_nothing(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)
        await computes.apply(ready(2, compute=compute))

        assert await computes.apply(ready(3, compute=compute)) is False

        assert (await computes.get(compute)).status.nodes_ready == 3
        assert (await _recorded(compute))[-1] == "compute.ready"
        assert (await _recorded(compute)).count("compute.ready") == 1

    async def degrading_twice_is_said_once(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)

        assert await computes.apply(ComputeDegraded(compute=compute, error="boom")) is True
        assert await computes.apply(ComputeDegraded(compute=compute, error="boom")) is False

        assert (await _recorded(compute)).count("compute.degraded") == 1

    async def a_failure_carries_its_own_code_into_the_status(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)

        await computes.apply(ComputeDegraded(compute=compute, error="quota", code="capability_mismatch"))

        error = (await computes.get(compute)).status.last_error
        assert error is not None
        assert (error.code, error.message) == ("capability_mismatch", "quota")

    async def a_transition_that_is_not_a_failure_clears_the_error(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)
        await computes.apply(ComputeDegraded(compute=compute, error="boom"))

        await computes.apply(ready(compute=compute))

        assert (await computes.get(compute)).status.last_error is None

    async def a_fact_is_recorded_and_the_state_stays(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)
        before = await computes.get(compute)

        await computes.apply(ComputeBound(compute=compute, offer="off_1", instance_type="a100.x", region=None, markets=("spot",)))

        after = await computes.get(compute)
        assert after.status.state == before.status.state
        assert after.revision == before.revision
        assert (await _recorded(compute))[-1] == "compute.bound"

    async def a_fact_saying_what_the_row_already_shows_is_a_repeat(tmp_path: Path) -> None:
        """The provider refusing to release, tick after tick, is one event until it refuses differently."""
        computes, compute = await _given(tmp_path)
        await computes.delete(compute, (await computes.get(compute)).revision, "once")

        await computes.apply(ComputeDeletionFailed(compute=compute, error="still busy", code="release_pending"))
        await computes.apply(ComputeDeletionFailed(compute=compute, error="still busy", code="release_pending"))
        await computes.apply(ComputeDeletionFailed(compute=compute, error="quota", code="release_pending"))

        assert (await _recorded(compute)).count("compute.deletion_failed") == 2
        error = (await computes.get(compute)).status.last_error
        assert error is not None
        assert error.code == "release_pending"

    async def an_illegal_event_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)

        with pytest.raises(IllegalTransitionError):
            await computes.apply(ComputeDeleted(compute=compute))

        assert (await computes.get(compute)).status.state == "requested"
        assert await _recorded(compute) == ["compute.created", "compute.generation.created"]

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
        original = computes._status

        async def renewed_underneath(compute_id: str) -> dict[str, Any]:
            """The row as apply reads it — and, the first time, a lease renewal landing right after."""
            nonlocal reads
            row = await original(compute_id)
            reads += 1
            if reads == 1:
                await ComputeRow.update({ComputeRow.revision: ComputeRow.revision + 1}).where(ComputeRow.id == compute_id).run()
            return row

        monkeypatch.setattr(computes, "_status", renewed_underneath)

        assert await computes.apply(ready(compute=compute)) is True

        assert reads == 2, "the first write lost to the renewal and the row was read again"
        assert (await computes.get(compute)).status.state == "ready"
        assert (await _recorded(compute)).count("compute.ready") == 1

    async def the_stream_sees_the_transition_after_it_is_committed(tmp_path: Path) -> None:
        events = EventStore()
        computes, compute = await _given(tmp_path, events)
        await computes.apply(ready(compute=compute))

        names = []
        async for _, name, _ in events.stream(None, compute, None, None):
            names.append(name)
            if name == "compute.ready":
                break

        assert names == ["compute.created", "compute.generation.created", "compute.ready"]


def describe_what_the_store_says_on_its_own() -> None:
    async def a_binding_that_landed_is_said_and_one_that_lost_is_not(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)

        assert await computes.bind(compute, Infrastructure(offer=OFFER, offer_id=OFFER.id, private_key="winner")) is True
        assert await computes.bind(compute, Infrastructure(offer=OFFER, offer_id=OFFER.id, private_key="loser")) is False

        assert (await _recorded(compute)).count("compute.bound") == 1

    async def a_generation_applied_is_said_by_the_generations(tmp_path: Path) -> None:
        computes, compute = await _given(tmp_path)

        await GenerationStore(computes).apply(compute, 1)

        assert (await _recorded(compute))[-1] == "compute.generation.applied"


def describe_the_client_fold() -> None:
    def a_pool_waits_on_the_state_the_events_lead_to() -> None:
        view = ComputeView(id="cmp_1")
        for event in (provisioning(), ready(), ComputeDeleting(compute="cmp_1", nodes_ready=2, nodes_total=2), ComputeDeleted(compute="cmp_1")):
            view = observe(view, event)

        assert view.state == "deleted"

    def a_fact_leaves_the_view_as_it_was() -> None:
        view = ComputeView(id="cmp_1", state="ready")
        bound = ComputeBound(compute="cmp_1", offer="off_1", instance_type="a100.x", region=None, markets=("spot",))

        assert observe(view, bound) is view
        assert observe(view, CostEvent(compute="cmp_1", cost=1.0, nodes=2, at=datetime.now(UTC))).state == "ready"

    def a_replay_folded_over_a_hydrated_view_does_not_raise() -> None:
        view = observe(ComputeView(id="cmp_1", state="ready"), provisioning())

        assert view.state == "provisioning"


async def _given(tmp_path: Path, events: EventStore | None = None) -> tuple[ComputeStore, str]:
    computes, compute = await given(tmp_path / "skyward.sqlite", events=events)
    return computes, compute.id


async def _recorded(compute: str) -> list[str]:
    rows = await EventRow.select(EventRow.type).where(EventRow.compute_id == compute).order_by(EventRow.sequence)
    return [row["type"] for row in rows]


def describe_a_pass_that_breaks() -> None:
    async def on_the_way_up_it_degrades_the_compute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        computes, compute, reconciler = await _reconciler(tmp_path, monkeypatch)

        await reconciler.compute(compute)
        await reconciler.compute(compute)

        assert (await computes.get(compute)).status.state == "degraded"
        assert (await _recorded(compute)).count("compute.degraded") == 1

    async def on_the_way_out_it_is_a_fact_and_the_teardown_goes_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        computes, compute, reconciler = await _reconciler(tmp_path, monkeypatch)
        await computes.delete(compute, (await computes.get(compute)).revision, "once")

        await reconciler.compute(compute)
        await reconciler.compute(compute)

        assert (await computes.get(compute)).status.state == "deleting"
        assert (await _recorded(compute)).count("compute.deletion_failed") == 1
        error = (await computes.get(compute)).status.last_error
        assert error is not None
        assert (error.code, error.message) == ("reconcile_failed", "the cloud is down")


async def _reconciler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ComputeStore, str, Reconciler]:
    """A reconciler whose every pass breaks on the provider, over a compute of its own."""
    events = EventStore()
    computes, compute = await _given(tmp_path, events)
    nodes, blobs = NodeStore(), BlobStore()
    machines = Machines(computes, nodes, ProviderStore(), OfferCache(ProviderStore()), blobs, events)

    async def down(*_: object) -> None:
        raise RuntimeError("the cloud is down")

    monkeypatch.setattr(machines, "resolve", down)
    reconciler = Reconciler(computes, GenerationStore(computes), nodes, TaskStore(computes, nodes, blobs), machines, events, Wakeup())
    return computes, compute, reconciler
