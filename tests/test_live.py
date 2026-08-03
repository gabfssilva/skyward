"""The Rich console folds v2 events into the v1 layout."""

from __future__ import annotations

import io
import logging
import sys
from datetime import UTC, datetime

import pytest
from rich.console import Console

from skyward.core.live import (
    NodeRow,
    Pool,
    RichConsole,
    View,
    _event_line,
    _quiet,
    observe,
    refresh,
    refresh_tasks,
    render,
)
from skyward.shared.schemas import (
    Compute,
    ComputeSpec,
    ComputeStatus,
    Image,
    Lease,
    Node,
    NodeBounds,
    Page,
    ProviderRef,
    RetryPolicy,
    Spec,
    Task,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 15, tzinfo=UTC)


def _compute(
    nodes_ready: int = 0,
    nodes_total: int = 0,
    state: str = "provisioning",
    cpus: int | None = None,
    memory_gb: int | None = None,
) -> Compute:
    return Compute(
        id="cmp_1",
        name="embed-train",
        revision=1,
        generation=1,
        spec=ComputeSpec(
            specs=(
                Spec(
                    provider=ProviderRef(kind="aws"),
                    accelerator="A100",
                    cpus=cpus,
                    memory_gb=memory_gb,
                    region="us-east-1",
                ),
            ),
            nodes=NodeBounds(desired=nodes_total or 1),
            allocation="spot",
            image=Image(),
        ),
        status=ComputeStatus(
            state=state,
            observed_generation=1,
            nodes_ready=nodes_ready,
            nodes_total=nodes_total,
        ),
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


def _task(state: str) -> Task:
    return Task(
        id=f"tsk_{state}",
        compute_id="cmp_1",
        generation=1,
        function="f" * 64,
        args_sha256="a" * 64,
        dispatch="one",
        state=state,
        retry=RetryPolicy(),
        executions=(),
        submitted_at=NOW,
    )


def _draw(view: View) -> str:
    output = io.StringIO()
    Console(file=output, width=160, color_system=None).print(render(view))
    return output.getvalue()


def test_a_metric_event_moves_the_nodes_gauge():
    view = observe(
        View(),
        "node.metrics",
        b'{"compute":"cmp_1","node":"nod_1","name":"gpu_util","value":87.5}',
    )

    assert view.nodes[0].metrics["gpu_util"] == (87.5,)


def test_state_phase_and_console_events_move_the_footer():
    view = observe(View(), "node.bootstrapping", b'{"compute":"cmp_1","node":"nod_1"}')
    view = observe(view, "node.phase", b'{"node":"nod_1","event":"started","phase":"deps"}')
    view = observe(view, "node.console", b'{"node":"nod_1","content":"installing torch"}')

    row = view.nodes[0]
    assert row.state == "bootstrapping"
    assert row.phases[0].name == "deps"
    assert row.tail == ("installing torch",)


def test_polling_preserves_data_that_only_exists_in_the_stream():
    view = observe(
        View(),
        "node.metrics",
        b'{"compute":"cmp_1","node":"nod_1","name":"gpu_util","value":87.5}',
    )
    view = observe(view, "node.console", b'{"node":"nod_1","content":"hello"}')

    merged = refresh(view, _compute(1, 1, "ready"), Page(items=(_node("nod_1", 0),)))

    assert merged.nodes[0].metrics["gpu_util"] == (87.5,)
    assert merged.nodes[0].tail == ("hello",)


def test_deleted_nodes_are_not_rendered():
    view = refresh(
        View(),
        _compute(1, 1, "ready"),
        Page(items=(_node("nod_1", 0, "deleted"),)),
    )

    assert view.nodes == ()


def test_rich_mode_uses_the_v1_progress_badges():
    nodes = Page(items=(_node("nod_0", 0, "ready"), _node("nod_1", 1, "connecting")))
    view = refresh(View(), _compute(1, 2), nodes)

    output = _draw(view)

    assert "ssh 1/2" in output
    assert "bootstrap 1/2" in output
    assert "ready 1/2" in output
    assert "2×" in output
    assert "A100" in output
    assert "AWS" in output


def test_rich_footer_preserves_the_complete_v1_infrastructure_line():
    view = refresh(
        View(),
        _compute(1, 1, "ready", cpus=8, memory_gb=32),
        Page(items=(_node("nod_0", 0),)),
    )

    output = _draw(view)

    for label in ("skyward", "1×", "spot", "A100", "us-east-1", "☁️ AWS", "8 vCPU", "32 GB", "⚡ 1× A100"):
        assert label in output


def test_bootstrap_output_is_pinned_above_the_badges():
    view = observe(View(), "node.bootstrapping", b'{"compute":"cmp_1","node":"nod_1"}')
    view = observe(view, "node.phase", b'{"node":"nod_1","event":"started","phase":"deps"}')
    view = observe(view, "node.console", b'{"node":"nod_1","content":"installing torch"}')

    output = _draw(view)

    assert "nod_1" in output
    assert "deps" in output
    assert "installing torch" in output


def test_metrics_and_tasks_are_badges():
    view = observe(
        View(nodes=(NodeRow("nod_1", state="ready"),)),
        "node.metrics",
        b'{"compute":"cmp_1","node":"nod_1","name":"gpu_util","value":75}',
    )
    view = refresh_tasks(view, Page(items=(_task("running"), _task("succeeded"))))

    output = _draw(view)

    assert "gpu 75%" in output
    assert "● 1 running" in output
    assert "✓ 1 done" in output


def test_tasks_use_the_registered_function_name():
    task = _task("running")

    view = refresh_tasks(View(), Page(items=(task,)), {task.function: "cuda_available"})

    assert view.tasks[0].function == "cuda_available"


def test_memory_usage_is_derived_from_used_and_total_megabytes():
    view = View(nodes=(NodeRow("nod_1", state="ready"),))
    view = observe(
        view,
        "node.metrics",
        b'{"compute":"cmp_1","node":"nod_1","name":"mem_used_mb","value":8192}',
    )
    view = observe(
        view,
        "node.metrics",
        b'{"compute":"cmp_1","node":"nod_1","name":"mem_total_mb","value":16384}',
    )

    output = _draw(view)

    assert "mem 50%" in output
    assert "mem 12288%" not in output


def test_bootstrap_is_complete_only_when_the_node_is_ready():
    connecting = _draw(View(pool=Pool(total=1, desired=1), nodes=(NodeRow("nod_1", state="connecting"),)))
    bootstrapping = _draw(View(pool=Pool(total=1, desired=1), nodes=(NodeRow("nod_1", state="bootstrapping"),)))
    ready = _draw(View(pool=Pool(total=1, desired=1), nodes=(NodeRow("nod_1", state="ready"),)))

    assert "ssh 0/1" in connecting
    assert "bootstrap 0/1" in connecting
    assert "ssh 1/1" in bootstrapping
    assert "bootstrap 0/1" in bootstrapping
    assert "ssh 1/1" in ready
    assert "bootstrap 1/1" in ready


def test_ready_event_is_a_v1_badged_line():
    line = _event_line(
        View(nodes=(NodeRow("nod_1", state="ready", machine="i-abc"),)),
        "node.ready",
        {"node": "nod_1"},
    )

    assert line.plain == "  i-abc     ✓ Joined"
    assert not line.style
    assert line.spans[0].end == 10


def test_the_console_hands_the_root_logger_back_as_it_found_it():
    output = io.StringIO()
    console = Console(file=output)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level

    with _quiet(console):
        assert root.level == logging.WARNING

    assert root.handlers == handlers
    assert root.level == level


def test_rich_output_defaults_to_stderr():
    assert RichConsole(object(), "cmp_1")._console.file is sys.stderr


async def test_rich_follower_prints_before_the_compute_finishes():
    class Client:
        async def call(self, method, path, kind, **params):
            if path.endswith("/nodes"):
                return Page(items=(_node("nod_1", 0, "connecting"),))
            if path == "/v1/tasks":
                return Page(items=())
            return _compute(0, 1)

        async def events(self, compute):
            yield "node.connecting", b'{"compute":"cmp_1","node":"nod_1"}'

    output = io.StringIO()

    await RichConsole(Client(), "cmp_1", output).follow()

    assert "skyward" in output.getvalue()
    assert "waiting" in output.getvalue()


async def test_rich_follower_labels_local_stdout():
    class Client:
        async def call(self, method, path, kind, **params):
            if path.endswith("/nodes"):
                return Page(items=(_node("nod_1", 0),))
            if path == "/v1/tasks":
                return Page(items=())
            return _compute(1, 1, "ready")

        async def events(self, compute):
            print({"is_cuda_available": True, "devices": 1})
            yield "compute.deleted", b'{"compute":"cmp_1"}'

    output = io.StringIO()
    stdout = sys.stdout

    await RichConsole(Client(), "cmp_1", output).follow()

    assert "local" in output.getvalue()
    assert "{'is_cuda_available': True, 'devices': 1}" in output.getvalue()
    assert sys.stdout is stdout


async def test_rich_follower_prints_the_v1_session_summary_when_the_compute_stops():
    class Client:
        async def call(self, method, path, kind, **params):
            if path.endswith("/nodes"):
                return Page(items=(_node("nod_1", 0),))
            if path == "/v1/tasks":
                return Page(items=(_task("succeeded"),))
            return _compute(1, 1, "ready")

        async def events(self, compute):
            yield "compute.deleted", b'{"compute":"cmp_1"}'

    output = io.StringIO()

    await RichConsole(Client(), "cmp_1", output).follow()

    assert "Session Summary" in output.getvalue()
    assert "1 completed" in output.getvalue()
