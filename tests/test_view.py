"""What a callback is handed: the compute folded into one value.

The fold is pure — an event in, the view after it out — so most of this file
needs no daemon. The observer at the end is the loop around the fold: one
stream, every subscriber, and a callback that raises is reported, not raised.
"""

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from datetime import UTC, datetime

import httpx
import pytest

from skyward.core.client import Client
from skyward.core.console import Observer
from skyward.core.view import (
    ERRORS,
    HISTORY,
    TAIL,
    ComputeView,
    NodeView,
    observe,
)
from skyward.shared.events import (
    ComputeDegraded,
    ComputeReady,
    ConsoleEvent,
    CostEvent,
    Event,
    MetricEvent,
    NodeEvent,
    PhaseEvent,
    ProgressEvent,
    TaskEvent,
)

pytestmark = pytest.mark.local


def describe_folding_the_lifecycle() -> None:
    def a_node_event_creates_the_row_it_speaks_of() -> None:
        view = observe(ComputeView(id="cmp_1"), NodeEvent(compute="cmp_1", node="nod_1", state="provisioning"))

        assert [node.state for node in view.nodes] == ["provisioning"]

    def a_compute_event_moves_the_state() -> None:
        view = observe(ComputeView(id="cmp_1"), ComputeReady(compute="cmp_1", nodes_ready=1, nodes_total=1, generation=1))

        assert view.state == "ready"

    def a_cost_event_moves_the_accrual() -> None:
        view = observe(ComputeView(id="cmp_1"), CostEvent(compute="cmp_1", cost=1.25, nodes=2, at=datetime.now(UTC)))

        assert view.cost == 1.25

    def ready_nodes_are_counted_from_the_rows() -> None:
        view = ComputeView(id="cmp_1", nodes=(NodeView(id="a", state="ready"), NodeView(id="b", state="bootstrapping")))

        assert view.nodes_ready == 1


def describe_the_errors_window() -> None:
    def what_goes_wrong_is_kept_wherever_it_happened() -> None:
        view = ComputeView(id="cmp_1")
        view = observe(view, NodeEvent(compute="cmp_1", node="nod_1", state="lost", error="preempted"))
        view = observe(view, ComputeDegraded(compute="cmp_1", error="below the floor"))
        view = observe(view, PhaseEvent(compute="cmp_1", node="nod_1", event="failed", phase="uv sync", at=datetime.now(UTC), error="no wheel"))

        assert view.errors == ("preempted", "below the floor", "no wheel")

    def the_same_message_twice_in_a_row_is_one_entry() -> None:
        view = ComputeView(id="cmp_1")
        for _ in range(3):
            view = observe(view, NodeEvent(compute="cmp_1", node="nod_1", state="lost", error="preempted"))

        assert view.errors == ("preempted",)

    def the_window_drops_the_oldest_first() -> None:
        view = ComputeView(id="cmp_1")
        for index in range(ERRORS + 5):
            view = observe(view, NodeEvent(compute="cmp_1", node="nod_1", state="lost", error=f"error {index}"))

        assert len(view.errors) == ERRORS
        assert view.errors[0] == "error 5"
        assert view.errors[-1] == f"error {ERRORS + 4}"


def describe_the_bootstrap_checklist() -> None:
    def a_phase_opens_then_closes() -> None:
        view = ComputeView(id="cmp_1")
        view = observe(view, PhaseEvent(compute="cmp_1", node="nod_1", event="started", phase="uv sync", at=datetime.now(UTC)))
        view = observe(view, PhaseEvent(compute="cmp_1", node="nod_1", event="completed", phase="uv sync", at=datetime.now(UTC)))

        (phase,) = view.nodes[0].phases
        assert phase.finished and phase.error is None

    def a_broken_phase_says_why() -> None:
        view = observe(
            ComputeView(id="cmp_1"),
            PhaseEvent(compute="cmp_1", node="nod_1", event="failed", phase="apt", at=datetime.now(UTC), error="mirror down"),
        )

        assert view.nodes[0].phases[0].error == "mirror down"


def describe_the_windows_a_long_run_never_outgrows() -> None:
    def the_tail_keeps_the_last_lines_only() -> None:
        view = ComputeView(id="cmp_1")
        for index in range(TAIL + 10):
            view = observe(view, ConsoleEvent(compute="cmp_1", node="nod_1", content=f"line {index}"))

        assert len(view.nodes[0].tail) == TAIL
        assert view.nodes[0].tail[-1] == f"line {TAIL + 9}"

    def a_gauge_keeps_a_short_history() -> None:
        view = ComputeView(id="cmp_1")
        for index in range(HISTORY + 3):
            view = observe(view, MetricEvent(compute="cmp_1", node="nod_1", name="cpu", value=float(index)))

        assert len(view.nodes[0].metrics["cpu"]) == HISTORY

    def a_gauge_nobody_draws_is_not_kept() -> None:
        view = observe(ComputeView(id="cmp_1"), MetricEvent(compute="cmp_1", node="nod_1", name="custom_thing", value=1.0))

        assert view.nodes == ()


def describe_what_a_machine_says_before_it_has_an_address() -> None:
    def progress_lands_on_the_node_the_api_already_named() -> None:
        view = ComputeView(id="cmp_1", nodes=(NodeView(id="nod_1"),))
        view = observe(view, ProgressEvent(compute="cmp_1", node="nod_1", progress="downloading", completion=0.4))

        assert view.nodes[0].progress == "downloading"
        assert view.nodes[0].completion == 0.4

    def progress_for_a_node_nobody_named_is_dropped() -> None:
        view = observe(ComputeView(id="cmp_1"), ProgressEvent(compute="cmp_1", node="nod_9", progress="downloading"))

        assert view.nodes == ()

    def a_machine_that_arrives_stops_saying_it_is_on_its_way() -> None:
        view = ComputeView(id="cmp_1", nodes=(NodeView(id="nod_1", progress="downloading", completion=0.9),))
        view = observe(view, NodeEvent(compute="cmp_1", node="nod_1", state="connecting"))

        assert view.nodes[0].progress is None


def describe_folding_tasks() -> None:
    def a_started_task_is_a_running_row() -> None:
        view = observe(ComputeView(id="cmp_1"), TaskEvent(compute="cmp_1", task="tsk_1", state="started"))

        assert [(task.id, task.state) for task in view.tasks] == [("tsk_1", "running")]

    def an_outcome_lands_on_the_row_it_belongs_to() -> None:
        view = observe(ComputeView(id="cmp_1"), TaskEvent(compute="cmp_1", task="tsk_1", state="started"))
        view = observe(view, TaskEvent(compute="cmp_1", task="tsk_1", state="succeeded"))

        assert view.tasks[0].state == "succeeded"


def describe_the_observer() -> None:
    async def every_callback_sees_every_event_with_the_view_after_it() -> None:
        events = (
            NodeEvent(compute="cmp_1", node="nod_1", state="provisioning"),
            NodeEvent(compute="cmp_1", node="nod_1", state="ready"),
            ComputeReady(compute="cmp_1", nodes_ready=1, nodes_total=1, generation=1),
        )
        seen: list[tuple[Event, ComputeView]] = []

        def spy(event: Event, view: ComputeView) -> None:
            seen.append((event, view))

        await Observer(_Stream(events), "cmp_1", callbacks=(spy,)).follow()

        assert [type(event) for event, _ in seen] == [NodeEvent, NodeEvent, ComputeReady]
        assert seen[-1][1].state == "ready"
        assert seen[-1][1].nodes_ready == 1

    async def a_callback_that_raises_does_not_take_the_stream_with_it(capsys: pytest.CaptureFixture[str]) -> None:
        events = (ComputeReady(compute="cmp_1", nodes_ready=1, nodes_total=1, generation=1),)
        landed: list[Event] = []

        def broken(event: Event, view: ComputeView) -> None:
            raise RuntimeError("boom")

        def steady(event: Event, view: ComputeView) -> None:
            landed.append(event)

        await Observer(_Stream(events), "cmp_1", callbacks=(broken, steady)).follow()

        assert len(landed) == 1, "the callback after the broken one still ran"
        assert "a callback raised" in capsys.readouterr().err


class _Stream(Client):
    """A client whose event stream is a script and whose API is down.

    The observer must survive both: fold the events it is given, and shrug at a
    ``refresh`` it cannot make.
    """

    def __init__(self, events: tuple[Event, ...]) -> None:
        super().__init__(httpx.AsyncClient(), AsyncExitStack())
        self._script = events

    async def events(self, compute: str) -> AsyncGenerator[tuple[str, bytes]]:
        import msgspec

        for event in self._script:
            yield "event", msgspec.json.encode(event)

    async def call[T](
        self,
        method: str,
        path: str,
        kind: type[T],
        /,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        **query: object,
    ) -> T:
        raise RuntimeError("the control plane is not answering")
