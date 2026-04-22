"""Tests for ``minimal_console_actor`` renderable and helpers."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest
from rich.console import Console, Group

from skyward.accelerators import Accelerator
from skyward.actors.console.minimal import (
    _avg,
    _dominant_bootstrap_phase,
    _header,
    _node_tails,
    _phase_label,
    _status,
    _summary,
    _View,
)
from skyward.api.model import Instance, InstanceType, Offer
from skyward.api.views import (
    BootstrapView,
    NodeStatus,
    NodeView,
    PoolPhase,
    PoolView,
    ScalingView,
    SessionView,
    TasksView,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


# ── Fixtures ─────────────────────────────────────────────────────


def _instance(
    *, region: str = "us-east-1", itype_name: str = "p4d.24xlarge",
    accel_name: str | None = "A100", accel_count: int = 8,
    vcpus: float = 96, memory_gb: float = 1152,
    spot_price: float | None = 1.0, on_demand_price: float | None = 3.0,
    spot: bool = False,
) -> Instance:
    accel = (
        Accelerator(name=accel_name, memory="40GB", count=accel_count)
        if accel_name else None
    )
    itype = InstanceType(
        name=itype_name, accelerator=accel, vcpus=vcpus,
        memory_gb=memory_gb, architecture="x86_64", specific=None,
    )
    offer = Offer(
        id="offer-1", instance_type=itype, spot_price=spot_price,
        on_demand_price=on_demand_price, billing_unit="hour", specific=None,
    )
    return Instance(
        id=f"i-{itype_name}-{region}", status="ready", offer=offer,
        region=region, spot=spot,
    )


def _pool(
    *, phase: PoolPhase = PoolPhase.READY,
    nodes: MappingProxyType[int, NodeView] | None = None,
    instances: tuple[Instance, ...] = (),
    tasks: TasksView | None = None,
    scaling: ScalingView | None = None,
    total_nodes: int | None = None,
) -> PoolView:
    resolved_nodes = nodes if nodes is not None else MappingProxyType({})
    return PoolView(
        name="train", phase=phase,
        tasks=tasks or TasksView(), scaling=scaling or ScalingView(),
        total_nodes=total_nodes if total_nodes is not None else len(resolved_nodes),
        nodes=resolved_nodes, instances=instances,
    )


def _text(renderable: object) -> str:
    """Render a renderable down to its plain text (strips markup)."""
    console = Console(width=200, color_system=None, record=True)
    console.print(renderable)
    return console.export_text()


# ── Helpers ──────────────────────────────────────────────────────


class TestPhaseLabel:
    @pytest.mark.parametrize(
        ("phase", "expected"),
        [
            (PoolPhase.PROVISIONING, "provisioning"),
            (PoolPhase.SSH, "ssh"),
            (PoolPhase.BOOTSTRAP, "bootstrap"),
            (PoolPhase.READY, "ready"),
            (PoolPhase.STOPPED, "stopped"),
        ],
    )
    def test_basic_phases(self, phase: PoolPhase, expected: str) -> None:
        assert _phase_label(_pool(phase=phase)) == expected

    def test_bootstrap_includes_active_phase(self) -> None:
        nodes = MappingProxyType({
            0: NodeView(
                node_id=0, status=NodeStatus.SSH,
                bootstrap=BootstrapView(
                    phases=("apt", "deps"), completed=frozenset(),
                    active="deps", output="",
                ),
            ),
        })
        pool = _pool(phase=PoolPhase.BOOTSTRAP, nodes=nodes, total_nodes=1)
        assert _phase_label(pool) == "bootstrap · deps"


class TestDominantBootstrapPhase:
    def test_picks_most_common(self) -> None:
        def n(nid: int, active: str) -> NodeView:
            return NodeView(
                node_id=nid, status=NodeStatus.SSH,
                bootstrap=BootstrapView(
                    phases=(), completed=frozenset(),
                    active=active, output="",
                ),
            )

        nodes = MappingProxyType({
            0: n(0, "apt"), 1: n(1, "deps"), 2: n(2, "deps"),
        })
        pool = _pool(phase=PoolPhase.BOOTSTRAP, nodes=nodes, total_nodes=3)
        assert _dominant_bootstrap_phase(pool) == "deps"

    def test_none_when_no_bootstrap(self) -> None:
        assert _dominant_bootstrap_phase(_pool()) is None


class TestAvg:
    def test_averages_metric_across_nodes(self) -> None:
        nodes = MappingProxyType({
            0: NodeView(
                node_id=0, status=NodeStatus.READY,
                metrics=MappingProxyType({"gpu.util": 80.0}),
            ),
            1: NodeView(
                node_id=1, status=NodeStatus.READY,
                metrics=MappingProxyType({"gpu.util": 60.0}),
            ),
        })
        pool = _pool(nodes=nodes, total_nodes=2)
        assert _avg(pool, "gpu.util") == 70.0

    def test_returns_none_when_metric_missing(self) -> None:
        assert _avg(_pool(), "gpu.util") is None


# ── Header ───────────────────────────────────────────────────────


class TestHeader:
    def test_empty_instances_shows_provisioning(self) -> None:
        txt = _text(_header(_pool(phase=PoolPhase.PROVISIONING)))
        assert "provisioning" in txt

    def test_single_group_header(self) -> None:
        pool = _pool(instances=(_instance(), _instance(region="us-east-1")))
        txt = _text(_header(pool))
        assert "2× p4d.24xlarge" in txt
        assert "A100×8" in txt
        assert "96 vCPU" in txt
        assert "1152 GB" in txt

    def test_heterogeneous_header_two_lines(self) -> None:
        pool = _pool(instances=(
            _instance(itype_name="p4d.24xlarge"),
            _instance(itype_name="RTX4090", vcpus=24, memory_gb=128, accel_count=1),
        ))
        txt = _text(_header(pool))
        assert "p4d.24xlarge" in txt
        assert "RTX4090" in txt

    def test_same_itype_multiple_regions_collapse(self) -> None:
        pool = _pool(instances=(
            _instance(region="CA-MTL-1"),
            _instance(region="EUR-IS-1"),
            _instance(region="EU-CZ-1"),
        ))
        txt = _text(_header(pool))
        assert txt.count("p4d.24xlarge") == 1
        assert "3× p4d.24xlarge" in txt
        assert "CA-MTL-1" in txt
        assert "EUR-IS-1" in txt
        assert "EU-CZ-1" in txt

    def test_region_dedup(self) -> None:
        pool = _pool(instances=(
            _instance(region="CA-MTL-1"),
            _instance(region="CA-MTL-1"),
        ))
        txt = _text(_header(pool))
        assert txt.count("CA-MTL-1") == 1
        assert "2× p4d.24xlarge" in txt

    def test_accelerator_count_formatted_as_int(self) -> None:
        pool = _pool(instances=(_instance(accel_count=1),))
        txt = _text(_header(pool))
        assert "A100×1" in txt
        assert "×1.0" not in txt

    def test_fractional_accelerator_count_preserved(self) -> None:
        from skyward.actors.console.minimal import _fmt_count
        assert _fmt_count(0.5) == "0.5"
        assert _fmt_count(1) == "1"
        assert _fmt_count(1.0) == "1"
        assert _fmt_count(8) == "8"

    def test_on_demand_rate_in_header(self) -> None:
        pool = _pool(instances=(
            _instance(on_demand_price=3.0, spot=False),
            _instance(region="us-east-2", on_demand_price=3.0, spot=False),
        ))
        txt = _text(_header(pool))
        assert "$6.00/hr" in txt

    def test_spot_rate_in_header(self) -> None:
        pool = _pool(instances=(
            _instance(spot_price=1.0, spot=True),
        ))
        txt = _text(_header(pool))
        assert "$1.00/hr" in txt

    def test_zero_price_hides_rate(self) -> None:
        pool = _pool(instances=(
            _instance(spot_price=None, on_demand_price=None),
        ))
        txt = _text(_header(pool))
        assert "/hr" not in txt

    def test_elastic_shows_range(self) -> None:
        scaling = ScalingView(is_elastic=True, min_nodes=2, max_nodes=8, desired=4)
        pool = _pool(instances=(_instance(),), scaling=scaling)
        txt = _text(_header(pool))
        assert "autoscale 2–8" in txt

    def test_non_accelerator_instance(self) -> None:
        pool = _pool(instances=(_instance(accel_name=None),))
        txt = _text(_header(pool))
        assert "p4d.24xlarge" in txt
        assert "A100" not in txt


# ── Status line ──────────────────────────────────────────────────


class TestStatus:
    def test_ready_shows_counts_and_tasks(self) -> None:
        nodes = MappingProxyType({0: NodeView(node_id=0, status=NodeStatus.READY)})
        tasks = TasksView(done=5, failed=1, running=2, queued=1, first_task_at=0.0)
        pool = _pool(phase=PoolPhase.READY, nodes=nodes, tasks=tasks, total_nodes=1)
        txt = _text(_status(pool))
        assert "ready" in txt
        assert "1/1 ready" in txt
        assert "tasks 3/5 ✓" in txt
        assert "1 ✗" in txt

    def test_in_flight_tasks_show_with_zero_done(self) -> None:
        tasks = TasksView(running=1, first_task_at=0.0)
        pool = _pool(phase=PoolPhase.READY, tasks=tasks)
        txt = _text(_status(pool))
        assert "tasks 1/0 ✓" in txt

    def test_scaling_up_suffix(self) -> None:
        scaling = ScalingView(desired=4, pending=2, reconciler_state="scaling_up")
        pool = _pool(phase=PoolPhase.READY, scaling=scaling, total_nodes=2)
        txt = _text(_status(pool))
        assert "+2 pending" in txt

    def test_draining_suffix(self) -> None:
        scaling = ScalingView(desired=2, draining=1, reconciler_state="draining")
        pool = _pool(phase=PoolPhase.READY, scaling=scaling, total_nodes=3)
        txt = _text(_status(pool))
        assert "-1 draining" in txt

    def test_watching_omits_reconciler_suffix(self) -> None:
        txt = _text(_status(_pool(phase=PoolPhase.READY)))
        assert "pending" not in txt
        assert "draining" not in txt

    def test_zero_tasks_hides_counter(self) -> None:
        txt = _text(_status(_pool(phase=PoolPhase.READY)))
        assert "tasks" not in txt
        assert "0✓" not in txt

    def test_running_tasks_shows_counter(self) -> None:
        pool = _pool(phase=PoolPhase.READY, tasks=TasksView(running=3))
        txt = _text(_status(pool))
        assert "tasks 3/0 ✓" in txt

    def test_gpu_metric_shown_when_present(self) -> None:
        nodes = MappingProxyType({
            0: NodeView(
                node_id=0, status=NodeStatus.READY,
                metrics=MappingProxyType({"gpu_util": 87.0}),
            ),
        })
        pool = _pool(phase=PoolPhase.READY, nodes=nodes, total_nodes=1)
        txt = _text(_status(pool))
        assert "gpu 87%" in txt

    def test_multi_gpu_metrics_averaged(self) -> None:
        nodes = MappingProxyType({
            0: NodeView(
                node_id=0, status=NodeStatus.READY,
                metrics=MappingProxyType({
                    "gpu_util_0": 80.0, "gpu_util_1": 60.0,
                    "gpu_mem_mb_0": 20480.0, "gpu_mem_mb_1": 10240.0,
                }),
            ),
        })
        pool = _pool(phase=PoolPhase.READY, nodes=nodes, total_nodes=1)
        txt = _text(_status(pool))
        assert "gpu 70%" in txt
        assert "vram 15.0GB" in txt

    def test_total_cost_shown_when_started(self) -> None:
        import time as _time

        pool = _pool(
            phase=PoolPhase.READY,
            instances=(_instance(on_demand_price=10.0, spot=False),),
            total_nodes=1,
        )
        pool = replace(pool, started_at=_time.monotonic() - 1800)
        txt = _text(_status(pool))
        assert "Σ $" in txt

    def test_total_cost_hidden_without_pricing(self) -> None:
        import time as _time

        pool = _pool(
            phase=PoolPhase.READY,
            instances=(_instance(spot_price=None, on_demand_price=None),),
            total_nodes=1,
        )
        pool = replace(pool, started_at=_time.monotonic() - 100)
        txt = _text(_status(pool))
        assert "Σ" not in txt

    def test_cpu_and_mem_metrics_shown(self) -> None:
        nodes = MappingProxyType({
            0: NodeView(
                node_id=0, status=NodeStatus.READY,
                metrics=MappingProxyType({"cpu": 45.0, "mem": 72.0}),
            ),
        })
        pool = _pool(phase=PoolPhase.READY, nodes=nodes, total_nodes=1)
        txt = _text(_status(pool))
        assert "cpu 45%" in txt
        assert "mem 72%" in txt


# ── Node tails ───────────────────────────────────────────────────


class TestNodeTails:
    def test_no_tails_when_ready(self) -> None:
        nodes = MappingProxyType({
            0: NodeView(
                node_id=0, status=NodeStatus.READY,
                bootstrap=BootstrapView(
                    phases=(), completed=frozenset(),
                    active="deps", output="installing",
                ),
            ),
        })
        pool = _pool(phase=PoolPhase.READY, nodes=nodes, total_nodes=1)
        assert _node_tails(pool) == []

    def test_tails_during_bootstrap(self) -> None:
        nodes = MappingProxyType({
            0: NodeView(
                node_id=0, status=NodeStatus.SSH,
                bootstrap=BootstrapView(
                    phases=("deps",), completed=frozenset(),
                    active="deps", output="Installing torch...",
                ),
            ),
            1: NodeView(
                node_id=1, status=NodeStatus.SSH,
                bootstrap=BootstrapView(
                    phases=("deps",), completed=frozenset(),
                    active="deps", output="Collecting numpy",
                ),
            ),
        })
        pool = _pool(phase=PoolPhase.BOOTSTRAP, nodes=nodes, total_nodes=2)
        tails = _node_tails(pool)
        assert len(tails) == 2
        assert "node-0" in _text(tails[0])
        assert "Installing torch" in _text(tails[0])
        assert "Collecting numpy" in _text(tails[1])

    def test_tail_uses_instance_id_when_available(self) -> None:
        from skyward.actors.console.minimal import _INSTANCE_ID_WIDTH

        nodes = MappingProxyType({
            0: NodeView(
                node_id=0, status=NodeStatus.SSH,
                instance=_instance(),
                bootstrap=BootstrapView(
                    phases=("deps",), completed=frozenset(),
                    active="deps", output="Installing torch...",
                ),
            ),
        })
        pool = _pool(phase=PoolPhase.BOOTSTRAP, nodes=nodes, total_nodes=1)
        tails = _node_tails(pool)
        assert len(tails) == 1
        txt = _text(tails[0])
        inst_id = nodes[0].instance.id  # type: ignore[union-attr]
        assert f"{inst_id[:_INSTANCE_ID_WIDTH]}/0" in txt

    def test_ready_nodes_excluded_from_tails(self) -> None:
        nodes = MappingProxyType({
            0: NodeView(node_id=0, status=NodeStatus.READY, bootstrap=None),
            1: NodeView(
                node_id=1, status=NodeStatus.SSH,
                bootstrap=BootstrapView(
                    phases=(), completed=frozenset(),
                    active="deps", output="still going",
                ),
            ),
        })
        pool = _pool(phase=PoolPhase.BOOTSTRAP, nodes=nodes, total_nodes=2)
        tails = _node_tails(pool)
        assert len(tails) == 1
        assert "node-1" in _text(tails[0])

    def test_tails_visible_during_ready_when_node_joins_late(self) -> None:
        """Late joiners (replacement / scale-up) surface bootstrap output
        even though the pool itself is already READY."""
        nodes = MappingProxyType({
            0: NodeView(node_id=0, status=NodeStatus.READY, bootstrap=None),
            1: NodeView(node_id=1, status=NodeStatus.READY, bootstrap=None),
            2: NodeView(
                node_id=2, status=NodeStatus.BOOTSTRAPPING,
                bootstrap=BootstrapView(
                    phases=("apt", "deps"), completed=frozenset({"apt"}),
                    active="deps", output="Installing torch==2.4.0",
                ),
            ),
        })
        pool = _pool(phase=PoolPhase.READY, nodes=nodes, total_nodes=3)
        tails = _node_tails(pool)
        assert len(tails) == 1
        txt = _text(tails[0])
        assert "node-2" in txt
        assert "Installing torch==2.4.0" in txt


# ── Renderable ───────────────────────────────────────────────────


class TestViewRenderable:
    def test_empty_view_renders_waiting(self) -> None:
        view = _View()
        txt = _text(view)
        assert "waiting" in txt

    def test_rich_returns_group(self) -> None:
        view = _View(view=SessionView(pools=MappingProxyType({"train": _pool()})))
        assert isinstance(view.__rich__(), Group)


class TestSummary:
    def test_summary_contains_task_counts(self) -> None:
        nodes = MappingProxyType({
            0: NodeView(node_id=0, status=NodeStatus.READY),
            1: NodeView(node_id=1, status=NodeStatus.READY),
        })
        pool = _pool(
            phase=PoolPhase.STOPPED, nodes=nodes, total_nodes=2,
            tasks=TasksView(done=42, failed=1, first_task_at=0.0),
        )
        txt = _text(_summary(pool, started_at=0.0))
        assert "done" in txt
        assert "42 tasks" in txt
        assert "1 failed" in txt


# ── resolve_console ──────────────────────────────────────────────


class TestResolveConsole:
    def test_true_maps_to_rich(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from skyward.actors.console import console_actor, resolve_console

        monkeypatch.setenv("SKYWARD_CONSOLE_FORCE_TTY", "1")
        assert resolve_console(True) is console_actor

    def test_rich_literal_maps_to_rich(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from skyward.actors.console import console_actor, resolve_console

        monkeypatch.setenv("SKYWARD_CONSOLE_FORCE_TTY", "1")
        assert resolve_console("rich") is console_actor

    def test_minimal_literal_maps_to_minimal(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from skyward.actors.console import minimal_console_actor, resolve_console

        monkeypatch.setenv("SKYWARD_CONSOLE_FORCE_TTY", "1")
        assert resolve_console("minimal") is minimal_console_actor

    def test_false_maps_to_none(self) -> None:
        from skyward.actors.console import resolve_console

        assert resolve_console(False) is None

    def test_silent_literal_maps_to_none(self) -> None:
        from skyward.actors.console import resolve_console

        assert resolve_console("silent") is None
