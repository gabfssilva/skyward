"""The live panel: fold events, merge the poll, apply the keyboard, draw the wall."""

from __future__ import annotations

import io
import logging
from dataclasses import replace
from datetime import UTC, datetime
from types import MappingProxyType

import pytest
from rich.console import Console as RichConsole
from rich.console import RenderableType

from skyward.protocol.schemas import (
    Compute,
    ComputeSpec,
    ComputeStatus,
    Error,
    Execution,
    Image,
    Lease,
    Node,
    NodeBounds,
    Page,
    PluginRef,
    ProviderRef,
    RetryPolicy,
    Spec,
    Task,
)
from skyward.sdk.live import (
    Flats,
    Hexes,
    NodeRow,
    View,
    _decode,
    _gb,
    _quiet,
    command,
    observe,
    plan_wall,
    refresh,
    render,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 15, tzinfo=UTC)

_FULL = {
    "gpu_util": 87.0,
    "gpu_mem_mb": 61 * 1024.0,
    "gpu_mem_total_mb": 80 * 1024.0,
    "cpu": 34.0,
    "mem_used_mb": 21 * 1024.0,
    "mem_total_mb": 64 * 1024.0,
}


def _compute(
    nodes_ready: int = 0,
    nodes_total: int = 0,
    state: str = "provisioning",
    pip: tuple[str, ...] = (),
    plugins: tuple[PluginRef, ...] = (),
) -> Compute:
    return Compute(
        id="cmp_1",
        name="embed-train",
        revision=1,
        generation=1,
        spec=ComputeSpec(
            specs=(Spec(provider=ProviderRef(kind="aws"), accelerator="A100", region="us-east-1"),),
            nodes=NodeBounds(desired=nodes_total or 1),
            allocation="spot",
            image=Image(pip=pip),
            plugins=plugins,
        ),
        status=ComputeStatus(state=state, observed_generation=1, nodes_ready=nodes_ready, nodes_total=nodes_total),
        lease=Lease(),
        created_at=NOW,
    )


def _node(node_id: str, rank: int, state: str = "ready", price: float = 1.19) -> Node:
    return Node(
        id=node_id,
        compute_id="cmp_1",
        generation=1,
        rank=rank,
        revision=1,
        desired="present",
        state=state,
        provider_binding={"region": "us-east-1a"},
        created_at=NOW,
        machine=f"i-{rank:08x}",
        address=f"10.0.0.{rank}",
        accelerator="A100",
        price_per_hour=price,
    )


def _row(node_id: str, rank: int, state: str = "ready", metrics: dict[str, float] | None = None) -> NodeRow:
    history = {name: (value,) for name, value in (metrics or {}).items()}
    return NodeRow(id=node_id, state=state, rank=rank, metrics=MappingProxyType(history))


def _task(task_id: str, state: str = "running", function: str = "fnsha0000", node: str = "nod_0", n_exec: int = 1, error: str | None = None) -> Task:
    executions = tuple(
        Execution(
            id=f"exe_{i}",
            rank=0,
            ordinal=i,
            state="failed" if (error and i == n_exec - 1) else "started",
            node_id=node,
            started_at=NOW,
            error=Error(code="task_failed", message=error, retryable=False) if (error and i == n_exec - 1) else None,
        )
        for i in range(n_exec)
    )
    return Task(
        id=task_id,
        compute_id="cmp_1",
        generation=1,
        function=function,
        args_sha256="a",
        dispatch="one",
        state=state,
        retry=RetryPolicy(),
        executions=executions,
        submitted_at=NOW,
    )


def _draw(renderable: RenderableType) -> str:
    console = RichConsole(file=io.StringIO(), width=220, color_system=None)
    console.print(renderable)
    return console.file.getvalue()


def _ready_nodes(n: int, metrics: dict[str, float] | None = None) -> tuple[NodeRow, ...]:
    return tuple(_row(f"nod_{i:02d}", i, "ready", metrics) for i in range(n))


# --------------------------------------------------------------------------- #
# observe / refresh                                                             #
# --------------------------------------------------------------------------- #


def test_a_metric_event_moves_that_nodes_gauge():
    view = observe(View(), "node.metrics", b'{"compute":"cmp_1","node":"nod_1","name":"gpu_util","value":87.5}')
    (row,) = view.nodes
    assert row.id == "nod_1"
    assert row.metrics["gpu_util"] == (87.5,)


def test_a_node_state_event_moves_that_nodes_state():
    view = observe(View(), "node.ready", b'{"compute":"cmp_1","node":"nod_1"}')
    (row,) = view.nodes
    assert row.state == "ready"


def test_a_state_transition_is_stamped_for_the_journey():
    view = observe(View(), "node.provisioning", b'{"compute":"cmp_1","node":"nod_1"}')
    view = observe(view, "node.connecting", b'{"compute":"cmp_1","node":"nod_1"}')
    (row,) = view.nodes
    assert set(row.state_at) == {"provisioning", "connecting"}
    assert row.state_at["provisioning"] <= row.state_at["connecting"]


def test_a_repeated_state_keeps_its_first_stamp():
    view = observe(View(), "node.connecting", b'{"compute":"cmp_1","node":"nod_1"}')
    first = view.nodes[0].state_at["connecting"]
    again = observe(view, "node.connecting", b'{"compute":"cmp_1","node":"nod_1"}')
    assert again.nodes[0].state_at["connecting"] == first


def test_a_compute_state_event_moves_the_pool():
    view = observe(View(), "compute.ready", b'{"compute":"cmp_1"}')
    assert view.pool.state == "ready"


def test_a_phase_event_moves_that_nodes_checklist():
    view = observe(View(), "node.phase", b'{"node":"nod_1","event":"started","phase":"deps","at":"2026-07-15T00:00:00+00:00"}')
    view = observe(view, "node.phase", b'{"node":"nod_1","event":"completed","phase":"deps","at":"2026-07-15T00:00:41+00:00"}')
    (row,) = view.nodes
    (mark,) = row.phases
    assert mark.name == "deps"
    assert mark.started is not None and mark.finished is not None
    assert (mark.finished - mark.started).total_seconds() == 41


def test_a_failed_phase_carries_its_error():
    view = observe(View(), "node.phase", b'{"node":"nod_1","event":"failed","phase":"python","error":"exit code 1"}')
    (row,) = view.nodes
    assert row.phases[0].error == "exit code 1"


def test_the_umbrella_bootstrap_phase_never_reaches_the_checklist():
    """The script wraps the real phases in a `bootstrap` of its own; the checklist shows the parts, not the wrapper."""
    view = observe(View(), "node.phase", b'{"node":"nod_1","event":"started","phase":"bootstrap"}')
    view = observe(view, "node.phase", b'{"node":"nod_1","event":"started","phase":"apt"}')
    view = observe(view, "node.phase", b'{"node":"nod_1","event":"completed","phase":"apt"}')
    view = observe(view, "node.phase", b'{"node":"nod_1","event":"completed","phase":"bootstrap"}')
    (row,) = view.nodes
    assert [mark.name for mark in row.phases] == ["apt"]


def test_an_installed_package_line_marks_the_package():
    view = observe(View(), "node.console", b'{"node":"nod_1","content":" + numpy==2.3.1"}')
    (row,) = view.nodes
    assert row.installed["numpy"] == "2.3.1"


def test_a_downloading_line_marks_the_package_as_fetching():
    view = observe(View(), "node.console", b'{"node":"nod_1","content":"Downloading torch (846.7MiB)"}')
    view = observe(view, "node.console", b'{"node":"nod_1","content":" Downloaded torch"}')
    (row,) = view.nodes
    assert row.fetching["torch"] == "846.7MiB", "the size survives the Downloaded confirmation"


def test_a_downloaded_line_alone_still_marks_the_package():
    """uv on a real machine says `Downloaded name` with no size and no Downloading before it."""
    view = observe(View(), "node.console", b'{"node":"nod_1","content":" Downloaded nvidia-cufile-cu12"}')
    (row,) = view.nodes
    assert "nvidia-cufile-cu12" in row.fetching


def test_the_poll_carries_the_package_plan():
    merged = refresh(View(), _compute(pip=("torch>=2.0", "numpy")), Page(items=()))
    assert merged.plan == ("torch>=2.0", "numpy")


def test_the_plan_includes_what_the_plugins_add():
    """The torch guide has an empty image.pip — torch arrives via the plugin's transform,
    the same one the connector applies before bootstrapping."""
    merged = refresh(View(), _compute(plugins=(PluginRef(kind="torch"),)), Page(items=()))
    assert merged.plan == ("torch",)


def test_the_poll_derives_the_bootstrap_steps():
    merged = refresh(View(), _compute(pip=("numpy",)), Page(items=()))
    assert merged.steps == ("apt", "uv", "venv", "skyward", "deps")
    bare = refresh(View(), _compute(), Page(items=()))
    assert bare.steps == ("apt", "uv", "venv", "skyward"), "no packages, no deps phase"


def test_a_console_line_goes_to_the_nodes_tail():
    view = observe(View(), "node.console", b'{"node":"nod_1","content":"installing torch"}')
    (row,) = view.nodes
    assert row.tail == ("installing torch",)


def test_a_console_line_that_says_whose_it_is_goes_to_the_tasks_output_too():
    view = observe(View(), "node.console", b'{"node":"nod_1","content":"epoch 3","task":"tsk_1"}')
    assert view.outputs["tsk_1"] == ("epoch 3",)
    (row,) = view.nodes
    assert row.tail == ("epoch 3",)


def test_an_event_the_panel_does_not_draw_changes_nothing():
    before = observe(View(), "node.ready", b'{"compute":"cmp_1","node":"nod_1"}')
    after = observe(before, "task.succeeded", b'{"compute":"cmp_1","task":"tsk_1"}')
    assert after == before


def test_the_poll_fills_structure_and_keeps_what_the_stream_gave():
    view = observe(View(), "node.metrics", b'{"compute":"cmp_1","node":"nod_1","name":"gpu_util","value":87.5}')
    view = observe(view, "node.phase", b'{"node":"nod_1","event":"started","phase":"deps","at":"2026-07-15T00:00:00+00:00"}')
    view = observe(view, "node.console", b'{"node":"nod_1","content":"hello"}')
    view = observe(view, "node.console", b'{"node":"nod_1","content":" + numpy==2.3.1"}')
    merged = refresh(view, _compute(nodes_ready=1, nodes_total=1, state="ready"), Page(items=(_node("nod_1", 0),)))

    assert merged.pool.name == "embed-train"
    assert merged.pool.provider == "AWS"
    assert merged.pool.ready == 1
    (row,) = merged.nodes
    assert row.address == "10.0.0.0"
    assert row.machine == "i-00000000"
    assert row.metrics["gpu_util"] == (87.5,), "the poll does not carry utilisation, so it must not erase it"
    assert row.phases[0].name == "deps", "nor the phases"
    assert row.tail == ("hello", " + numpy==2.3.1"), "nor the tail"
    assert row.installed == {"numpy": "2.3.1"}, "nor what uv said it installed"


def test_the_poll_stamps_a_state_the_stream_never_said():
    """A late attach sees the poll first; the journey clock starts there rather than never."""
    merged = refresh(View(), _compute(nodes_total=1), Page(items=(_node("nod_1", 0, state="connecting"),)))
    (row,) = merged.nodes
    assert "connecting" in row.state_at


def test_the_poll_keeps_the_streams_stamp():
    view = observe(View(), "node.connecting", b'{"compute":"cmp_1","node":"nod_1"}')
    first = view.nodes[0].state_at["connecting"]
    merged = refresh(view, _compute(nodes_total=1), Page(items=(_node("nod_1", 0, state="connecting"),)))
    assert merged.nodes[0].state_at["connecting"] == first


def test_a_deleted_node_is_not_drawn():
    merged = refresh(View(), _compute(), Page(items=(_node("nod_1", 0, state="deleted"), _node("nod_2", 1))))
    assert [row.id for row in merged.nodes] == ["nod_2"]


def test_the_poll_leaves_the_keyboard_state_alone():
    """A repaint from the poll must not throw away where the viewer has navigated to."""
    view = replace(View(nodes=_ready_nodes(3)), ui=replace(View().ui, cursor=2))
    merged = refresh(view, _compute(nodes_ready=3, nodes_total=3, state="ready"), Page(items=(_node("nod_0", 0),)))
    assert merged.ui.cursor == 2


# --------------------------------------------------------------------------- #
# the wall plan: cell size as a function of how many must fit                   #
# --------------------------------------------------------------------------- #


def test_two_nodes_get_the_biggest_hexes():
    assert plan_wall(2, 76, 28) == Hexes(cols=2, scale=3)


def test_the_hex_size_answers_to_the_pool_not_the_terminal():
    """Ten nodes on a huge screen get medium hexes — a big terminal never inflates them."""
    assert plan_wall(10, 200, 60) == Hexes(cols=10, scale=2)


def test_more_nodes_step_down_the_ladder():
    small = plan_wall(40, 76, 28)
    big = plan_wall(6, 76, 28)
    assert isinstance(small, Hexes) and isinstance(big, Hexes)
    assert small.scale < big.scale


def test_a_flood_of_nodes_falls_back_to_flat_blocks():
    plan = plan_wall(200, 76, 28)
    assert isinstance(plan, Flats)
    assert -(-200 // plan.cols) <= 28


def test_a_tiny_terminal_falls_to_the_last_rung_rather_than_failing():
    plan = plan_wall(500, 20, 5)
    assert isinstance(plan, Flats)
    assert plan.block_w == 2
    assert not plan.numbered, "two cells cannot hold a three-digit rank"


# --------------------------------------------------------------------------- #
# render — the strip, the wall, the counts                                      #
# --------------------------------------------------------------------------- #


def test_empty_pool_says_it_is_waiting():
    assert "waiting" in _draw(render(View()))


def test_the_strip_totals_the_price():
    view = refresh(View(), _compute(nodes_ready=2, nodes_total=2, state="ready"), Page(items=(_node("nod_0", 0), _node("nod_1", 1))))
    assert "$2.38/hr" in _draw(render(view))


def test_the_meters_gauge_moves_the_strips_accrued_cost():
    view = refresh(View(), _compute(nodes_ready=1, nodes_total=1, state="ready"), Page(items=(_node("nod_0", 0),)))
    view = observe(view, "compute.cost", b'{"compute":"cmp_1","cost":0.13,"nodes":1,"at":"2026-07-15T00:00:00+00:00"}')
    assert "$0.13" in _draw(render(view))


def test_the_poll_does_not_erase_the_accrued_cost():
    view = observe(View(), "compute.cost", b'{"compute":"cmp_1","cost":0.13,"nodes":1,"at":"2026-07-15T00:00:00+00:00"}')
    merged = refresh(view, _compute(nodes_ready=1, nodes_total=1, state="ready"), Page(items=(_node("nod_0", 0),)))
    assert merged.pool.cost == 0.13, "the poll does not carry the meter's reading, so it must not erase it"


def test_the_strip_leads_with_the_spec_not_the_hash():
    view = refresh(View(), _compute(nodes_ready=2, nodes_total=2, state="ready"), Page(items=(_node("nod_0", 0), _node("nod_1", 1))))
    out = _draw(render(view))
    assert "2× A100" in out
    assert "embed-train" in out, "the name is still there, beside the spec"


def test_the_strip_speaks_for_the_compute_not_the_nodes():
    """A pool that has been ready for days stays ready while a replacement boots below."""
    nodes = Page(items=(_node("nod_0", 0, "ready"), _node("nod_1", 1, "bootstrapping")))
    view = refresh(View(), _compute(nodes_ready=1, nodes_total=2, state="ready"), nodes)
    out = _draw(render(view))
    assert "1/2 ready" in out
    assert "bootstrap 1" in out, "the booting node speaks through the counts under the wall"


def test_every_node_is_a_numbered_block_on_the_wall():
    view = View(nodes=_ready_nodes(30, {"gpu_util": 80.0}))
    out = _draw(render(view))
    assert " 13 " in out, "every rank is on the wall — no pagination"
    assert " 29 " in out


def test_the_counts_total_the_wall():
    nodes = (*_ready_nodes(3), _row("nod_boot", 3, "bootstrapping"), _row("nod_bad", 4, "lost"))
    out = _draw(render(View(nodes=nodes)))
    assert "ready 3" in out
    assert "bootstrap 1" in out
    assert "lost 1" in out


def test_an_idle_node_is_called_out_in_the_counts():
    nodes = (_row("nod_0", 0, "ready", {"gpu_util": 2.0}), _row("nod_1", 1, "ready", {"gpu_util": 90.0}))
    assert "idle 1" in _draw(render(View(nodes=nodes)))


# --------------------------------------------------------------------------- #
# render — the inspector                                                        #
# --------------------------------------------------------------------------- #


def test_the_inspector_shows_the_node_under_the_cursor():
    view = View(nodes=_ready_nodes(3, _FULL))
    view = command(view, "right")
    out = _draw(render(view))
    assert "INSPECTOR · #1" in out


def test_a_ready_nodes_inspector_is_metrics_work_and_machine():
    row = replace(_row("nod_0", 0, "ready", _FULL), machine="i-0f3a21", address="10.0.0.7", price=1.19, region="us-east-1a")
    view = View(nodes=(row,), tasks=(_task_row("tsk_0", "running", "embed_batch", "nod_0"),))
    out = _draw(render(view))
    assert "METRICS" in out
    assert "87%" in out
    assert "61" in out and "/80G" in out
    assert "embed_batch" in out, "the running task lives in the inspector's work section"
    assert "10.0.0.7" in out
    assert "$1.19/hr" in out


def test_a_booting_nodes_inspector_is_the_journey_and_the_tail():
    view = observe(View(), "node.provisioning", b'{"compute":"cmp_1","node":"nod_1"}')
    view = observe(view, "node.connecting", b'{"compute":"cmp_1","node":"nod_1"}')
    view = observe(view, "node.console", b'{"node":"nod_1","content":"waiting for ssh"}')
    out = _draw(render(view))
    assert "JOURNEY" in out
    assert "provision" in out and "connect" in out
    assert "○ bootstrap" in out, "the road ahead is hollow dots"
    assert "waiting for ssh" in out, "the tail is the output section"


def test_a_bootstrapping_inspector_names_the_running_phase():
    view = observe(View(), "node.bootstrapping", b'{"compute":"cmp_1","node":"nod_1"}')
    view = observe(view, "node.phase", b'{"node":"nod_1","event":"started","phase":"deps"}')
    out = _draw(render(view))
    assert "bootstrap · deps" in out


def test_a_failed_nodes_inspector_is_a_cross_and_a_callout():
    view = observe(View(), "node.bootstrapping", b'{"compute":"cmp_1","node":"nod_1"}')
    view = observe(view, "node.failed", b'{"compute":"cmp_1","node":"nod_1","error":"apt: no such package"}')
    out = _draw(render(view))
    assert "✗ failed" in out
    assert "apt: no such package" in out


def test_a_failed_phase_shows_in_the_callout():
    view = observe(View(), "node.bootstrapping", b'{"compute":"cmp_1","node":"nod_1"}')
    view = observe(view, "node.phase", b'{"node":"nod_1","event":"failed","phase":"deps","error":"exit code 1"}')
    view = observe(view, "node.failed", b'{"compute":"cmp_1","node":"nod_1"}')
    assert "exit code 1" in _draw(render(view))


def test_an_idle_nodes_inspector_prices_the_idleness():
    row = replace(_row("nod_0", 0, "ready", {"gpu_util": 2.0}), price=1.19)
    out = _draw(render(View(nodes=(row,))))
    assert "idle" in out


def test_a_replacements_inspector_shows_the_retry():
    view = observe(View(), "node.bootstrapping", b'{"compute":"cmp_1","node":"nod_1"}')
    view = observe(view, "node.failed", b'{"compute":"cmp_1","node":"nod_1","error":"apt: no such package"}')
    view = observe(view, "node.provisioning", b'{"compute":"cmp_1","node":"nod_1"}')
    out = _draw(render(view))
    assert "RETRY" in out
    assert "apt: no such package" in out, "the reason it fell survives into the retry"


# --------------------------------------------------------------------------- #
# command — the keyboard, as a pure function of the view                        #
# --------------------------------------------------------------------------- #


def test_left_and_right_walk_the_wall_and_stop_at_the_ends():
    view = View(nodes=_ready_nodes(6))
    assert command(view, "right").ui.cursor == 1
    assert command(view, "left").ui.cursor == 0, "there is nothing before the first block"
    end = view
    for _ in range(20):
        end = command(end, "right")
    assert end.ui.cursor == 5, "the cursor cannot run off the wall"


def test_up_and_down_move_by_one_band_of_the_honeycomb():
    view = View(nodes=_ready_nodes(30))
    cols = plan_wall(30, 76, 28).cols
    assert command(view, "down").ui.cursor == cols
    assert command(command(view, "down"), "up").ui.cursor == 0
    assert command(view, "up") == view, "there is nothing above the first row"


def test_attention_jumps_to_the_broken_block():
    nodes = (*_ready_nodes(5, {"gpu_util": 80.0}), _row("nod_bad", 5, "lost"))
    view = View(nodes=nodes)
    assert command(view, ("char", "a")).ui.cursor == 5


def test_attention_cycles_when_more_than_one_needs_it():
    nodes = (
        _row("nod_bad", 0, "failed"),
        *_ready_nodes(3, {"gpu_util": 80.0})[0:0],
        _row("nod_1", 1, "ready", {"gpu_util": 80.0}),
        _row("nod_idle", 2, "ready", {"gpu_util": 1.0}),
    )
    view = View(nodes=nodes)
    first = command(view, ("char", "a"))
    assert first.ui.cursor == 2, "the idle block is attention too"
    assert command(first, ("char", "a")).ui.cursor == 0, "and it wraps back around to the failure"


def test_go_to_rank_builds_a_query_and_jumps():
    view = View(nodes=_ready_nodes(30))
    view = command(view, ("char", "/"))
    assert view.ui.find == ""
    for char in "13":
        view = command(view, ("char", char))
    view = command(view, "enter")
    assert view.ui.find is None
    assert view.ui.cursor == 13


def test_go_to_rank_ignores_letters():
    view = command(View(nodes=_ready_nodes(3)), ("char", "/"))
    assert command(view, ("char", "x")).ui.find == ""


def test_go_to_rank_escape_cancels_without_moving():
    view = View(nodes=_ready_nodes(30))
    view = command(command(view, ("char", "/")), ("char", "9"))
    cancelled = command(view, "escape")
    assert cancelled.ui.find is None
    assert cancelled.ui.cursor == 0


def test_quit_raises_the_flag():
    assert command(View(), ("char", "q")).ui.quitting is True


# --------------------------------------------------------------------------- #
# decode                                                                        #
# --------------------------------------------------------------------------- #


def test_decode_splits_a_cbreak_read_into_keys():
    assert _decode("\x1b[A\x1b[B\x1b[C\x1b[Dx\r\x7f\x1b") == [
        "up",
        "down",
        "right",
        "left",
        ("char", "x"),
        "enter",
        "backspace",
        "escape",
    ]


# --------------------------------------------------------------------------- #
# helpers under test                                                            #
# --------------------------------------------------------------------------- #


def test_a_small_machine_keeps_a_decimal_and_a_big_one_drops_it():
    assert _gb(400) == "0.4"
    assert _gb(61000) == "60"


def test_the_panel_hands_the_root_logger_back_as_it_found_it():
    """A pinned panel owns the root logger while it is up, and not a moment longer."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    with _quiet(RichConsole(file=io.StringIO())):
        assert root.level == logging.WARNING
        assert len(root.handlers) == 1
    assert root.handlers == handlers
    assert root.level == level


# --------------------------------------------------------------------------- #
# small local helpers                                                           #
# --------------------------------------------------------------------------- #


def _task_row(task_id: str, state: str, function: str, node: str):
    from skyward.sdk.live import TaskRow

    return TaskRow(id=task_id, state=state, function=function, node=node, started_at=NOW, submitted_at=NOW)
