"""The live console: the panel at the foot of the terminal, and the lines it leaves above it."""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from io import StringIO
from types import MappingProxyType

import pytest
from rich.text import Text

from skyward.api.v1 import TaskCounts
from skyward.core.live import LISTED, RichConsole, Seen, advance, heard, panel, said
from skyward.core.view import ComputeView, NodeView, PhaseView
from skyward.shared.events import ComputeDeleted, ConsoleEvent, NodeEvent
from skyward.shared.schemas import NodeState

pytestmark = pytest.mark.local

T0 = 1_800_000_000.0
BORN = datetime.fromtimestamp(T0, UTC)
COMPUTE = ComputeView(
    id="cmp_1",
    name="train",
    state="provisioning",
    provider="aws",
    region="us-east-1",
    accelerator="A100",
    allocation="spot",
    initial=4,
    created_at=BORN,
    cost=0.44,
    rate=5.16,
)


def describe_the_panel() -> None:
    def describe_when_machines_are_still_being_bought() -> None:
        def it_leaves_the_time_to_the_header() -> None:
            view = with_nodes(*(node(rank, "requested") for rank in range(4)))

            drawn = draw(view, advance(Seen(), view, T0 + 5), T0 + 20)

            assert drawn[0].endswith("0:20 · $0.44")
            assert drawn[1] == "⠋ provision  0/4 machines"
            assert len(drawn) == 2, "nothing runs before a machine is up"

    def describe_when_nodes_bootstrap() -> None:
        def it_closes_the_provision_and_lists_each_node_still_setting_up() -> None:
            seen = advance(Seen(), with_nodes(*(node(rank, "requested") for rank in range(4))), T0)
            seen = advance(seen, with_nodes(*(node(rank, "bootstrapping") for rank in range(4))), T0 + 38)
            view = with_nodes(
                node(0, "ready"),
                node(1, "bootstrapping", step="deps", said="Resolved 84 packages in 1.2s"),
                node(2, "ready"),
                node(3, "bootstrapping", step="venv", said="Using CPython 3.12.7"),
            )

            drawn = draw(view, advance(seen, view, T0 + 100), T0 + 118)

            assert drawn[1].startswith("✓ provision  4/4 machines") and drawn[1].endswith("38s")
            assert drawn[2].startswith("⠋ bootstrap  2/4 ready") and drawn[2].endswith("1:20"), "timed from the first node to begin it"
            assert drawn[3:] == ["    1  install deps  Resolved 84 packages in 1.2s", "    3  create venv   Using CPython 3.12.7"]

        def it_names_the_step_a_node_last_began_while_it_moves_to_the_next() -> None:
            seen = advance(Seen(), with_nodes(node(0, "bootstrapping"), node(1, "bootstrapping")), T0)
            between = replace(node(1, "bootstrapping", said="+ numpy==2.5.3"), phases=(PhaseView("deps", started=True, finished=True),))
            view = with_nodes(node(0, "ready"), between)

            drawn = draw(view, advance(seen, view, T0 + 30), T0 + 30)

            assert drawn[2:] == ["    1  install deps  + numpy==2.5.3"]

    def describe_when_the_last_node_is_ready_before_the_compute_is() -> None:
        def it_draws_the_run_with_nothing_on_it_yet() -> None:
            seen = advance(Seen(), with_nodes(node(0, "bootstrapping")), T0)
            view = with_nodes(node(0, "ready"))

            drawn = draw(view, advance(seen, view, T0 + 60), T0 + 61)

            assert drawn[1:] == ["· idle"], "rather than a header with nothing under it"

    def describe_when_the_compute_runs() -> None:
        def it_shows_the_tasks_and_the_gauges_across_the_nodes() -> None:
            view = running()

            drawn = draw(view, advance(Seen(), view, T0), T0 + 304)

            assert drawn[1] == "⠋ 12 running · 140 done · 2 failed"
            assert drawn[2] == "  gpu 86% 80–92   vram 48/80 GB 46–49   cpu 33%   mem 41%"
            assert drawn[0].startswith("skyward · train · 4× A100 80GB spot · aws us-east-1 · $5.16/h")

    def describe_when_a_node_goes_away_while_it_runs() -> None:
        def it_draws_the_replacement_beside_the_run() -> None:
            _, view, seen = lost_and_asked_again()

            drawn = draw(view, seen, T0 + 461)

            assert drawn[1].startswith("⠋ provision  3/4 machines · replacing node 2") and drawn[1].endswith("0:11")
            assert drawn[2].startswith("⠋ 12 running")

        def it_adds_the_bootstrap_once_the_replacement_has_a_machine() -> None:
            ran, view, seen = lost_and_asked_again()
            view = replace(view, nodes=(*ran.nodes[:2], ran.nodes[3], node(2, "bootstrapping", id="nod_2b", step="deps", said="Resolved 84 packages")))

            drawn = draw(view, advance(seen, view, T0 + 490), T0 + 526)

            assert drawn[1].startswith("✓ provision  4/4 machines · node 2 replaced") and drawn[1].endswith("40s")
            assert drawn[2].startswith("⠋ bootstrap  3/4 ready") and drawn[2].endswith("0:36")
            assert drawn[3] == "    2  install deps  Resolved 84 packages"
            assert drawn[4].startswith("⠋ 12 running")

        def it_leaves_only_the_run_once_the_replacement_is_ready() -> None:
            ran, view, seen = lost_and_asked_again()
            bootstrapping = replace(view, nodes=(*ran.nodes[:2], ran.nodes[3], node(2, "bootstrapping", id="nod_2b")))
            back = replace(view, nodes=(*ran.nodes[:2], ran.nodes[3], node(2, "ready", id="nod_2b")))
            seen = advance(advance(seen, bootstrapping, T0 + 490), back, T0 + 562)

            drawn = draw(back, seen, T0 + 570)

            assert drawn[1].startswith("⠋ 12 running")
            assert len(drawn) == 3, "the header, the tasks and the gauges"

    def describe_when_more_nodes_bootstrap_than_it_lists() -> None:
        def it_groups_them_by_step_and_names_the_slowest() -> None:
            steps = ("venv", "deps", "deps", "deps", "deps", "worker")
            setting = tuple(node(10 + index, "bootstrapping", id=f"nod_s{index}", step=step, said=f"from {10 + index}") for index, step in enumerate(steps))
            view = replace(COMPUTE, state="ready", initial=8, nodes=(node(0, "ready"), node(1, "ready"), *setting))
            seen = advance(Seen(), replace(view, nodes=view.nodes[:2]), T0)
            seen = advance(seen, replace(view, nodes=(*view.nodes[:2], *(replace(each, state="requested") for each in setting))), T0 + 1)
            seen = advance(seen, replace(view, nodes=(*view.nodes[:2], *setting[1:])), T0 + 5)
            seen = advance(seen, view, T0 + 9)

            drawn = draw(view, seen, T0 + 20)

            assert len(setting) > LISTED
            assert drawn[3:6] == [
                "    create venv     node 10 · from 10",
                "    install deps    4 nodes · slowest 11 · from 11",
                "    start worker    node 15 · from 15",
            ]


def describe_the_scrollback() -> None:
    def it_puts_what_a_node_printed_behind_its_rank() -> None:
        view = running()

        line = said(ConsoleEvent(compute="cmp_1", node="nod_3", content="epoch 2/10"), view, advance(Seen(), view, T0))

        assert plain(line) == "3 │ epoch 2/10"

    def it_names_the_only_node_too() -> None:
        view = with_nodes(node(0, "ready"))

        line = said(ConsoleEvent(compute="cmp_1", node="nod_0", content="epoch 2/10"), view, advance(Seen(), view, T0))

        assert plain(line) == "0 │ epoch 2/10", "or it reads like a print of the local process"

    def it_keeps_what_a_node_prints_while_it_bootstraps_in_the_panel() -> None:
        view = with_nodes(node(0, "ready"), node(1, "bootstrapping"))

        assert said(ConsoleEvent(compute="cmp_1", node="nod_1", content="Resolved"), view, advance(Seen(), view, T0)) is None

    def it_says_a_node_went_away_and_why() -> None:
        view = with_nodes(node(0, "ready"), node(1, "lost"))

        line = said(NodeEvent(compute="cmp_1", node="nod_1", state="lost", error="spot interruption"), view, advance(Seen(), view, T0))

        assert plain(line) == "1 │ ✗ lost · spot interruption"

    def it_says_how_long_a_replacement_took_once_it_is_ready() -> None:
        ran, view, seen = lost_and_asked_again()
        seen = advance(seen, replace(view, nodes=(*ran.nodes[:2], ran.nodes[3], node(2, "bootstrapping", id="nod_2b"))), T0 + 490)
        back = replace(view, nodes=(*ran.nodes[:2], ran.nodes[3], node(2, "ready", id="nod_2b")))

        line = said(NodeEvent(compute="cmp_1", node="nod_2b", state="ready"), back, advance(seen, back, T0 + 562))

        assert plain(line) == "2 │ ✓ replaced · provision 40s · bootstrap 1m12s"

    def it_says_nothing_of_the_nodes_the_compute_opened_with() -> None:
        seen = advance(Seen(), with_nodes(node(0, "bootstrapping"), node(1, "bootstrapping")), T0)
        view = with_nodes(node(0, "ready"), node(1, "bootstrapping"))

        assert said(NodeEvent(compute="cmp_1", node="nod_0", state="ready"), view, advance(seen, view, T0 + 60)) is None


def describe_the_end() -> None:
    def describe_when_the_compute_is_deleted() -> None:
        def it_leaves_a_summary_where_the_panel_was() -> None:
            out = StringIO()
            console = RichConsole(out)
            view = running()
            console.opened(view)

            console.event(ComputeDeleted(compute="cmp_1"), replace(view, state="deleted"))
            console.closed(replace(view, state="deleted"))

            printed = out.getvalue()
            assert "✓ skyward · train · 4× A100 80GB spot · aws us-east-1" in printed
            assert "tasks      12 running · 140 done · 2 failed" in printed
            assert "detached" not in printed

    def describe_when_the_console_stops_watching_a_compute_still_up() -> None:
        def it_says_the_compute_keeps_running() -> None:
            out = StringIO()
            console = RichConsole(out)
            console.opened(running())

            console.closed(running())

            assert "↩ detached · train keeps running · up " in out.getvalue()


def lost_and_asked_again() -> tuple[ComputeView, ComputeView, Seen]:
    """A running compute whose node 2 went away at 7:20, and the node asked for in its place at 7:29."""
    ran = running()
    seen = advance(Seen(), ran, T0)
    lost = replace(ran, nodes=(*ran.nodes[:2], replace(ran.nodes[2], state="lost"), ran.nodes[3]))
    seen = heard(advance(seen, lost, T0 + 440), NodeEvent(compute="cmp_1", node="nod_2", state="lost"), lost)
    view = replace(lost, nodes=(*lost.nodes, node(2, "requested", id="nod_2b", created_at=datetime.fromtimestamp(T0 + 450, UTC))))
    return ran, view, advance(seen, view, T0 + 451)


def running() -> ComputeView:
    counts = TaskCounts(queued=0, running=12, succeeded=140, failed=2, cancelled=0, timed_out=0, indeterminate=0)
    return replace(
        COMPUTE,
        state="ready",
        task_counts=counts,
        nodes=tuple(node(rank, "ready", metrics=gauges(80 + rank * 4, 46 + rank, 33, 41)) for rank in range(4)),
    )


def with_nodes(*nodes: NodeView) -> ComputeView:
    return replace(COMPUTE, nodes=nodes)


def node(
    rank: int,
    state: NodeState,
    *,
    id: str | None = None,
    step: str | None = None,
    said: str | None = None,
    metrics: Mapping[str, tuple[float, ...]] | None = None,
    created_at: datetime = BORN,
) -> NodeView:
    return NodeView(
        id=id or f"nod_{rank}",
        rank=rank,
        state=state,
        market="spot",
        price_per_hour=1.29,
        created_at=created_at,
        phases=(PhaseView("env", started=True, finished=True), PhaseView(step, started=True)) if step else (),
        tail=(said,) if said else (),
        metrics=MappingProxyType(dict(metrics or {})),
    )


def gauges(gpu: float, vram_gb: float, cpu: float, mem: float) -> Mapping[str, tuple[float, ...]]:
    return {"gpu_util": (gpu,), "gpu_mem_mb": (vram_gb * 1024,), "gpu_mem_total_mb": (81920.0,), "cpu": (cpu,), "mem": (mem,)}


def draw(view: ComputeView, seen: Seen, now: float) -> list[str]:
    return plains(panel(view, seen, now))


def plains(lines: Iterable[Text]) -> list[str]:
    return [line.plain.rstrip() for line in lines]


def plain(line: Text | None) -> str | None:
    return None if line is None else line.plain
