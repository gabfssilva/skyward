"""Tests for the console actor — State, View, Controller."""

from __future__ import annotations

from io import StringIO
from types import MappingProxyType

import pytest
from rich.console import Console

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]

# --- State tests ---


class TestPhase:
    def test_phases_are_ordered(self) -> None:
        from skyward.actors.console.state import _Phase

        phases = list(_Phase)
        names = [p.name for p in phases]
        assert names == [
            "PROVISIONING", "SSH", "BOOTSTRAP", "WORKERS", "READY", "STOPPED",
        ]


class TestState:
    def test_default_state(self) -> None:
        from skyward.actors.console.state import _Phase, _State

        state = _State(total_nodes=4)
        assert state.phase == _Phase.PROVISIONING
        assert state.total_nodes == 4
        assert state.tasks_running == 0
        assert state.tasks_done == 0
        assert state.tasks_failed == 0
        assert state.cluster is None
        assert state.instances == ()
        assert state.task_latencies == ()

    def test_state_is_frozen(self) -> None:
        from skyward.actors.console.state import _State

        state = _State(total_nodes=2)
        with pytest.raises(AttributeError):
            state.phase = "bad"  # type: ignore[misc]


class TestNodeStatus:
    def test_node_status_values(self) -> None:
        from skyward.actors.console.state import _NodeStatus

        statuses = list(_NodeStatus)
        names = [s.name for s in statuses]
        assert names == ["WAITING", "SSH", "BOOTSTRAPPING", "READY"]


class TestThroughput:
    def test_throughput(self) -> None:
        from skyward.actors.console.state import _State, _throughput

        state = _State(
            total_nodes=1, tasks_done=10,
            first_task_at=100.0, pool_started_at=0.0,
        )
        rate = _throughput(state, now=160.0)
        assert rate == 10.0


# --- View tests ---


def _capture_console() -> tuple[Console, StringIO]:
    buf = StringIO()
    console = Console(file=buf, color_system=None, width=120)
    return console, buf


class TestBadge:
    def test_fixed_badge_has_style(self) -> None:
        from skyward.actors.console.view import _badge_style

        style = _badge_style("skyward")
        assert style.bgcolor is not None

    def test_dynamic_badge_uses_golden_angle(self) -> None:
        from skyward.actors.console.view import _badge_style

        s1 = _badge_style("i-abc123")
        s2 = _badge_style("i-def456")
        assert s1.bgcolor != s2.bgcolor

    def test_same_label_returns_same_style(self) -> None:
        from skyward.actors.console.view import _badge_style

        s1 = _badge_style("i-abc123")
        s2 = _badge_style("i-abc123")
        assert s1 == s2


class TestEmit:
    def test_emit_prints_badge_and_text(self) -> None:
        from skyward.actors.console.view import _emit

        console, buf = _capture_console()
        _emit(console, "skyward", "hello world")
        output = buf.getvalue()
        assert "skyward" in output
        assert "hello world" in output

    def test_emit_with_symbol(self) -> None:
        from skyward.actors.console.view import _emit

        console, buf = _capture_console()
        _emit(console, "tasks", "\u2713 train completed (2.1s)")
        output = buf.getvalue()
        assert "tasks" in output
        assert "\u2713 train completed" in output


class TestFooter:
    def test_provisioning_footer_single_line(self) -> None:
        from skyward.actors.console.state import _Phase, _State
        from skyward.actors.console.view import _LiveFooter

        console, buf = _capture_console()
        footer = _LiveFooter()
        footer.state = _State(total_nodes=4, phase=_Phase.PROVISIONING)
        console.print(footer)
        output = buf.getvalue()
        assert "provisioning" in output.lower()

    def test_bootstrap_footer_shows_progress(self) -> None:
        from skyward.actors.console.state import _NodeStatus, _Phase, _State
        from skyward.actors.console.view import _LiveFooter

        console, buf = _capture_console()
        footer = _LiveFooter()
        footer.state = _State(
            total_nodes=3, phase=_Phase.BOOTSTRAP,
            nodes=MappingProxyType({
                0: _NodeStatus.BOOTSTRAPPING,
                1: _NodeStatus.SSH,
            }),
        )
        console.print(footer)
        output = buf.getvalue()
        assert "bootstrap" in output.lower()
        assert "1/3" in output

    def test_ready_footer_shows_task_counts(self) -> None:
        from skyward.actors.console.state import _Phase, _State
        from skyward.actors.console.view import _LiveFooter

        console, buf = _capture_console()
        footer = _LiveFooter()
        footer.state = _State(
            total_nodes=2, phase=_Phase.READY,
            tasks_running=3, tasks_done=12,
            first_task_at=100.0, pool_started_at=0.0,
        )
        console.print(footer)
        output = buf.getvalue()
        assert "ready" in output.lower()
        assert "3 running" in output
        assert "12 done" in output

    def test_stopping_footer(self) -> None:
        from skyward.actors.console.state import _Phase, _State
        from skyward.actors.console.view import _LiveFooter

        console, buf = _capture_console()
        footer = _LiveFooter()
        footer.state = _State(total_nodes=2, phase=_Phase.STOPPED)
        console.print(footer)
        assert "shutting down" in buf.getvalue().lower()


class TestSummary:
    def test_summary_includes_duration(self) -> None:
        from skyward.actors.console.state import _State
        from skyward.actors.console.view import _render_summary

        console, buf = _capture_console()
        state = _State(
            total_nodes=2, tasks_done=10, tasks_failed=1,
            pool_started_at=0.0, first_task_at=10.0,
            task_latencies=(1.0, 2.0, 3.0),
        )
        result = _render_summary(state, now=600.0)
        console.print(result)
        output = buf.getvalue()
        assert "Session Summary" in output
        assert "10 completed" in output
        assert "1 failed" in output

    def test_summary_includes_throughput(self) -> None:
        from skyward.actors.console.state import _State
        from skyward.actors.console.view import _render_summary

        console, buf = _capture_console()
        state = _State(
            total_nodes=1, tasks_done=60, first_task_at=0.0,
            pool_started_at=0.0, task_latencies=tuple(1.0 for _ in range(60)),
        )
        result = _render_summary(state, now=60.0)
        console.print(result)
        output = buf.getvalue()
        assert "tasks/min" in output

    def test_summary_shows_node_distribution(self) -> None:
        from skyward.actors.console.state import _State
        from skyward.actors.console.view import _render_summary

        console, buf = _capture_console()
        state = _State(
            total_nodes=3, tasks_done=30, pool_started_at=0.0,
            tasks_per_node=MappingProxyType({
                0: 12, 1: 10, 2: 8,
            }),
        )
        result = _render_summary(state, now=60.0)
        console.print(result)
        output = buf.getvalue()
        assert "Distribution" in output
        assert "avg 10" in output
        assert "min 8" in output
        assert "max 12" in output

    def test_summary_with_no_tasks(self) -> None:
        from skyward.actors.console.state import _State
        from skyward.actors.console.view import _render_summary

        console, buf = _capture_console()
        state = _State(total_nodes=1, pool_started_at=0.0)
        result = _render_summary(state, now=60.0)
        console.print(result)
        output = buf.getvalue()
        assert "Session Summary" in output
        assert "0 completed" in output


# --- Controller tests ---


class TestConsoleActor:
    @staticmethod
    def _make_spec():
        from skyward.core.spec import Image, Nodes, PoolSpec

        return PoolSpec(nodes=Nodes(min=2), accelerator=None, region="us-east-1", image=Image())

    def test_actor_can_be_spawned(self) -> None:
        import asyncio

        from skyward.actors.console import LocalOutput, console_actor

        async def run() -> None:
            from casty import ActorSystem

            async with ActorSystem("test") as system:
                ref = system.spawn(console_actor(), "console")
                ref.tell(LocalOutput(line="hello"))
                await asyncio.sleep(0.1)

        asyncio.run(run())

    def test_actor_handles_start_pool(self) -> None:
        import asyncio

        from skyward.actors.console import console_actor

        async def run() -> None:
            from casty import ActorSystem, SpyEvent

            from skyward.actors.pool.messages import StartPool

            spec = self._make_spec()
            async with ActorSystem("test") as system:
                ref = system.spawn(console_actor(), "console")
                ref.tell(SpyEvent(
                    actor_path="/test/pool",
                    event=StartPool(
                        spec=spec,
                        provider_config=None,  # type: ignore[arg-type]
                        provider=None,
                        offers=(),
                        reply_to=None,  # type: ignore[arg-type]
                    ),
                    timestamp=0.0,
                ))
                await asyncio.sleep(0.1)

        asyncio.run(run())


class TestFormatTask:
    def test_simple_function(self) -> None:
        from skyward.actors.console.view import _format_task

        def train() -> None:
            pass

        result = _format_task(train, (), {})
        assert result == "train"

    def test_with_args(self) -> None:
        from skyward.actors.console.view import _format_task

        def train() -> None:
            pass

        result = _format_task(train, (10,), {})
        assert result == "train(10)"

    def test_with_kwargs(self) -> None:
        from skyward.actors.console.view import _format_task

        def train() -> None:
            pass

        result = _format_task(train, (), {"lr": 0.01})
        assert result == "train(lr=0.01)"

    def test_long_signature_truncated(self) -> None:
        from skyward.actors.console.view import _format_task

        def train() -> None:
            pass

        result = _format_task(train, ("a" * 50,), {"b": "c" * 50})
        assert "\u2026" in result


class TestNodeLabel:
    def test_resolves_from_instances(self) -> None:
        from unittest.mock import MagicMock

        from skyward.actors.console.state import _State
        from skyward.actors.console.view import _node_label

        inst0 = MagicMock()
        inst0.id = "i-abc"
        inst1 = MagicMock()
        inst1.id = "i-def"
        state = _State(total_nodes=2, instances=(inst0, inst1))
        assert _node_label(state, 0) == "i-abc"
        assert _node_label(state, 1) == "i-def"

    def test_fallback_for_missing_instance(self) -> None:
        from skyward.actors.console.state import _State
        from skyward.actors.console.view import _node_label

        state = _State(total_nodes=2)
        assert _node_label(state, 0) == "node-0"
        assert _node_label(state, 5) == "node-5"


class TestInstanceIdResolution:
    def test_resolve_from_nodes_by_index(self) -> None:
        from skyward.actors.console.state import _State
        from skyward.actors.console.view import _resolve_instance_id

        state = _State(total_nodes=2)
        assert _resolve_instance_id(state, node_id=0) is None

    def test_resolve_with_instances(self) -> None:
        from unittest.mock import MagicMock

        from skyward.actors.console.state import _State
        from skyward.actors.console.view import _resolve_instance_id

        inst0 = MagicMock()
        inst0.id = "i-abc"
        inst1 = MagicMock()
        inst1.id = "i-def"
        state = _State(total_nodes=2, instances=(inst0, inst1))
        assert _resolve_instance_id(state, node_id=0) == "i-abc"
        assert _resolve_instance_id(state, node_id=1) == "i-def"
        assert _resolve_instance_id(state, node_id=5) is None

    def test_resolve_with_none_node_id(self) -> None:
        from skyward.actors.console.state import _State
        from skyward.actors.console.view import _resolve_instance_id

        state = _State(total_nodes=2)
        assert _resolve_instance_id(state, node_id=None) is None


class TestSshUrl:
    def test_ssh_url_with_custom_port(self) -> None:
        from unittest.mock import MagicMock

        from skyward.actors.console.state import _State
        from skyward.actors.console.view import _ssh_url

        inst = MagicMock()
        inst.id = "i-abc"
        inst.ip = "10.0.0.1"
        inst.ssh_port = 2222

        state = _State(total_nodes=1, instances=(inst,), ssh_user="root")
        assert _ssh_url(state, 0) == "ssh://root@10.0.0.1:2222"


class TestNodeIdFromPath:
    def test_extracts_node_id(self) -> None:
        from skyward.actors.console.view import _node_id_from_path

        assert _node_id_from_path("/system/user/pool/node-0") == 0
        assert _node_id_from_path("/system/user/pool/node-15") == 15
        assert _node_id_from_path("node-3/monitor-xxx") == 3

    def test_returns_none_for_no_node(self) -> None:
        from skyward.actors.console.view import _node_id_from_path

        assert _node_id_from_path("/system/user/pool") is None
        assert _node_id_from_path("reconciler") is None


class TestCollectBadges:
    @staticmethod
    def _badges_plain(state: object) -> str:
        from skyward.actors.console.view import _collect_badges
        infra, status, tasks = _collect_badges(state)  # type: ignore[arg-type]
        return "".join(b.plain for b in [*infra, *status, *tasks])

    def test_always_has_skyward(self) -> None:
        from skyward.actors.console.state import _State

        assert "skyward" in self._badges_plain(_State(total_nodes=2))

    def test_infra_badges(self) -> None:
        from unittest.mock import MagicMock

        from skyward.actors.console.state import _State

        inst = MagicMock()
        inst.spot = True
        inst.region = "us-east-1"
        inst.offer.instance_type.name = "t4g.small"
        inst.offer.instance_type.vcpus = 2
        inst.offer.instance_type.memory_gb = 8
        inst.offer.instance_type.accelerator = None
        inst.offer.spot_price = 0.0
        inst.offer.on_demand_price = 0.0

        cluster = MagicMock()
        cluster.spec.provider = "aws"

        state = _State(total_nodes=2, instances=(inst, inst), cluster=cluster)
        plain = self._badges_plain(state)
        assert "2\u00d7" in plain
        assert "spot" in plain
        assert "t4g.small" in plain
        assert "us-east-1" in plain
        assert "AWS" in plain
        assert "4 vCPU" in plain
        assert "16 GB" in plain

    def test_ready_static(self) -> None:
        from skyward.actors.console.state import _NodeStatus, _Phase, _State

        state = _State(
            total_nodes=4, phase=_Phase.READY,
            nodes=MappingProxyType(dict.fromkeys(range(4), _NodeStatus.READY)),
        )
        plain = self._badges_plain(state)
        assert "ready" in plain
        assert "workers 4/4" in plain

    def test_ready_static_shows_reconciler(self) -> None:
        from skyward.actors.console.state import _NodeStatus, _Phase, _State

        state = _State(
            total_nodes=4, phase=_Phase.READY,
            nodes=MappingProxyType(dict.fromkeys(range(4), _NodeStatus.READY)),
            reconciler_state="watching",
        )
        plain = self._badges_plain(state)
        assert "in sync" in plain
        assert "min" not in plain
        assert "max" not in plain

    def test_ready_static_scaling(self) -> None:
        from skyward.actors.console.state import _NodeStatus, _Phase, _State

        state = _State(
            total_nodes=4, phase=_Phase.READY,
            nodes=MappingProxyType(dict.fromkeys(range(3), _NodeStatus.READY)),
            reconciler_state="scaling_up", desired_nodes=4, pending_nodes=1,
        )
        plain = self._badges_plain(state)
        assert "scaling" in plain
        assert "pending 1" in plain

    def test_ready_elastic_in_sync(self) -> None:
        from skyward.actors.console.state import _NodeStatus, _Phase, _State

        state = _State(
            total_nodes=4, phase=_Phase.READY,
            nodes=MappingProxyType(dict.fromkeys(range(4), _NodeStatus.READY)),
            is_elastic=True, desired_nodes=4, min_nodes=2, max_nodes=10,
            reconciler_state="watching",
        )
        plain = self._badges_plain(state)
        assert "in sync" in plain
        assert "min 2" in plain
        assert "max 10" in plain

    def test_ready_elastic_scaling(self) -> None:
        from skyward.actors.console.state import _NodeStatus, _Phase, _State

        state = _State(
            total_nodes=4, phase=_Phase.READY,
            nodes=MappingProxyType(dict.fromkeys(range(4), _NodeStatus.READY)),
            is_elastic=True, desired_nodes=8, pending_nodes=4,
            min_nodes=2, max_nodes=10, reconciler_state="scaling_up",
        )
        plain = self._badges_plain(state)
        assert "scaling" in plain
        assert "8" in plain
        assert "pending 4" in plain

    def test_tasks_badges(self) -> None:
        from skyward.actors.console.state import _NodeStatus, _Phase, _State

        state = _State(
            total_nodes=2, phase=_Phase.READY,
            nodes=MappingProxyType(dict.fromkeys(range(2), _NodeStatus.READY)),
            tasks_queued=5, tasks_running=3, tasks_done=12, tasks_failed=1,
        )
        plain = self._badges_plain(state)
        assert "5 queued" in plain
        assert "3 running" in plain
        assert "12 done" in plain
        assert "1 failed" in plain

    def test_metric_gauges(self) -> None:
        from skyward.actors.console.state import _NodeStatus, _Phase, _State

        state = _State(
            total_nodes=1, phase=_Phase.READY,
            nodes=MappingProxyType({0: _NodeStatus.READY}),
            metrics=MappingProxyType({
                0: MappingProxyType({
                    "cpu": 77.0, "mem": 65.0,
                    "gpu_util": 82.0,
                    "gpu_mem_mb": 7200.0,
                    "gpu_mem_total_mb": 10000.0,
                }),
            }),
        )
        plain = self._badges_plain(state)
        assert "cpu 77%" in plain
        assert "mem 65%" in plain
        assert "gpu 82%" in plain
        assert "vram 72%" in plain

    def test_vram_from_spec(self) -> None:
        from unittest.mock import MagicMock

        from skyward.actors.console.state import _State

        accel = MagicMock()
        accel.name = "A100"
        accel.count = 1
        accel.memory = ""

        inst = MagicMock()
        inst.spot = True
        inst.region = "us-east-1"
        inst.offer.instance_type.name = "p4d.24xlarge"
        inst.offer.instance_type.vcpus = 96
        inst.offer.instance_type.memory_gb = 192
        inst.offer.instance_type.accelerator = accel
        inst.offer.spot_price = 0.0
        inst.offer.on_demand_price = 0.0

        cluster = MagicMock()
        cluster.spec.provider = "aws"

        state = _State(
            total_nodes=1, instances=(inst,), cluster=cluster,
            spec_accelerator_memory="80GB",
        )
        assert "80GB" in self._badges_plain(state)

    def test_footer_returns_group(self) -> None:
        from rich.console import Group

        from skyward.actors.console.state import _Phase, _State
        from skyward.actors.console.view import _LiveFooter

        footer = _LiveFooter()
        footer.state = _State(total_nodes=2, phase=_Phase.READY)
        console, _ = _capture_console()
        results = list(footer.__rich_console__(console, console.options))
        assert any(isinstance(r, Group) for r in results)
