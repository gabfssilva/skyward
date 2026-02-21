"""Tests for the console actor — Model, View, Controller."""

from __future__ import annotations

from io import StringIO
from types import MappingProxyType

import pytest
from rich.console import Console

# --- Model tests ---


class TestPhase:
    def test_phases_are_ordered(self) -> None:
        from skyward.actors.console import _Phase

        phases = list(_Phase)
        names = [p.name for p in phases]
        assert names == [
            "PROVISIONING", "SSH", "BOOTSTRAP", "WORKERS", "READY", "STOPPING",
        ]


class TestState:
    def test_default_state(self) -> None:
        from skyward.actors.console import _Phase, _State

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
        from skyward.actors.console import _State

        state = _State(total_nodes=2)
        with pytest.raises(AttributeError):
            state.phase = "bad"  # type: ignore[misc]


class TestNodeStatus:
    def test_node_status_values(self) -> None:
        from skyward.actors.console import _NodeStatus

        statuses = list(_NodeStatus)
        names = [s.name for s in statuses]
        assert names == ["WAITING", "SSH", "BOOTSTRAPPING", "READY"]


class TestStateTransitions:
    def test_advance_to_ssh(self) -> None:
        from skyward.actors.console import _on_cluster_ready, _Phase, _State

        state = _State(total_nodes=4, phase=_Phase.PROVISIONING)
        new = _on_cluster_ready(state)
        assert new.phase == _Phase.SSH

    def test_ssh_connected_advances_node(self) -> None:
        from skyward.actors.console import (
            _NodeStatus,
            _on_ssh_connected,
            _Phase,
            _State,
        )

        state = _State(total_nodes=2, phase=_Phase.SSH)
        new = _on_ssh_connected(state, "i-abc")
        assert new.nodes["i-abc"] == _NodeStatus.SSH
        assert new.phase == _Phase.SSH

    def test_ssh_all_connected_advances_to_bootstrap(self) -> None:
        from skyward.actors.console import (
            _on_ssh_connected,
            _Phase,
            _State,
        )

        state = _State(total_nodes=1, phase=_Phase.SSH)
        new = _on_ssh_connected(state, "i-abc")
        assert new.phase == _Phase.BOOTSTRAP

    def test_bootstrap_done_advances_node(self) -> None:
        from skyward.actors.console import (
            _NodeStatus,
            _on_bootstrap_done,
            _Phase,
            _State,
        )

        state = _State(total_nodes=2, phase=_Phase.BOOTSTRAP)
        new = _on_bootstrap_done(state, "i-abc")
        assert new.nodes["i-abc"] == _NodeStatus.BOOTSTRAPPING

    def test_worker_started_advances_node(self) -> None:
        from skyward.actors.console import (
            _NodeStatus,
            _on_worker_started,
            _Phase,
            _State,
        )

        state = _State(total_nodes=2, phase=_Phase.WORKERS)
        new = _on_worker_started(state, "i-abc")
        assert new.nodes["i-abc"] == _NodeStatus.READY
        assert new.phase == _Phase.WORKERS

    def test_worker_all_started_advances_to_ready(self) -> None:
        from skyward.actors.console import (
            _on_worker_started,
            _Phase,
            _State,
        )

        state = _State(total_nodes=1, phase=_Phase.WORKERS)
        new = _on_worker_started(state, "i-abc")
        assert new.phase == _Phase.READY

    def test_record_task_submitted(self) -> None:
        from skyward.actors.console import _on_task_submitted, _State

        state = _State(total_nodes=1)
        new = _on_task_submitted(state, "t1", "train", "single")
        assert new.tasks_queued == 1
        assert new.tasks_running == 0
        assert "t1" in new.inflight

    def test_record_task_assigned(self) -> None:
        from skyward.actors.console import (
            _on_task_assigned,
            _on_task_submitted,
            _State,
        )

        state = _State(total_nodes=1)
        state = _on_task_submitted(state, "t1", "train", "single")
        new = _on_task_assigned(state, "t1", "i-abc")
        assert new.tasks_queued == 0
        assert new.tasks_running == 1

    def test_record_task_done(self) -> None:
        from skyward.actors.console import (
            _on_task_assigned,
            _on_task_done,
            _on_task_submitted,
            _State,
        )

        state = _State(total_nodes=1)
        state = _on_task_submitted(state, "t1", "train", "single")
        state = _on_task_assigned(state, "t1", "i-abc")
        new = _on_task_done(state, "t1", elapsed=2.5)
        assert new.tasks_running == 0
        assert new.tasks_done == 1
        assert new.task_latencies == (2.5,)

    def test_record_task_failed(self) -> None:
        from skyward.actors.console import _on_task_failed, _on_task_submitted, _State

        state = _State(total_nodes=1)
        state = _on_task_submitted(state, "t1", "train", "single")
        new = _on_task_failed(state, "t1")
        assert new.tasks_running == 0
        assert new.tasks_failed == 1

    def test_record_metric(self) -> None:
        from skyward.actors.console import _on_metric, _State

        state = _State(total_nodes=1)
        new = _on_metric(state, "i-abc", "gpu_util", 87.5)
        assert new.metrics["i-abc"]["gpu_util"] == 87.5

    def test_throughput(self) -> None:
        from skyward.actors.console import _State, _throughput

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
        from skyward.actors.console import _badge_style

        style = _badge_style("skyward")
        assert style.bgcolor is not None

    def test_dynamic_badge_uses_golden_angle(self) -> None:
        from skyward.actors.console import _badge_style

        s1 = _badge_style("i-abc123")
        s2 = _badge_style("i-def456")
        assert s1.bgcolor != s2.bgcolor

    def test_same_label_returns_same_style(self) -> None:
        from skyward.actors.console import _badge_style

        s1 = _badge_style("i-abc123")
        s2 = _badge_style("i-abc123")
        assert s1 == s2


class TestEmit:
    def test_emit_prints_badge_and_text(self) -> None:
        from skyward.actors.console import _emit

        console, buf = _capture_console()
        _emit(console, "skyward", "hello world")
        output = buf.getvalue()
        assert "skyward" in output
        assert "hello world" in output

    def test_emit_with_symbol(self) -> None:
        from skyward.actors.console import _emit

        console, buf = _capture_console()
        _emit(console, "tasks", "\u2713 train completed (2.1s)")
        output = buf.getvalue()
        assert "tasks" in output
        assert "\u2713 train completed" in output


class TestFooter:
    def test_provisioning_footer_single_line(self) -> None:
        from skyward.actors.console import _Phase, _render_footer, _State

        console, buf = _capture_console()
        state = _State(total_nodes=4, phase=_Phase.PROVISIONING)
        result = _render_footer(state)
        console.print(result)
        output = buf.getvalue()
        assert "provisioning" in output.lower()

    def test_bootstrap_footer_shows_nodes(self) -> None:
        from skyward.actors.console import (
            _NodeStatus,
            _Phase,
            _render_footer,
            _State,
        )

        console, buf = _capture_console()
        state = _State(
            total_nodes=3, phase=_Phase.BOOTSTRAP,
            nodes=MappingProxyType({
                "i-abc": _NodeStatus.BOOTSTRAPPING,
                "i-def": _NodeStatus.SSH,
            }),
        )
        result = _render_footer(state)
        console.print(result)
        output = buf.getvalue()
        assert "bootstrap" in output.lower()
        assert "i-abc" in output or "i-abc"[:8] in output

    def test_ready_footer_shows_task_counts(self) -> None:
        from skyward.actors.console import _Phase, _render_footer, _State

        console, buf = _capture_console()
        state = _State(
            total_nodes=2, phase=_Phase.READY,
            tasks_running=3, tasks_done=12,
            first_task_at=100.0, pool_started_at=0.0,
        )
        result = _render_footer(state)
        console.print(result)
        output = buf.getvalue()
        assert "ready" in output.lower()
        assert "3 running" in output
        assert "12 done" in output

    def test_stopping_footer(self) -> None:
        from skyward.actors.console import _Phase, _render_footer, _State

        console, buf = _capture_console()
        state = _State(total_nodes=2, phase=_Phase.STOPPING)
        result = _render_footer(state)
        console.print(result)
        assert "shutting down" in buf.getvalue().lower()


class TestSummary:
    def test_summary_includes_duration(self) -> None:
        from skyward.actors.console import _render_summary, _State

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
        from skyward.actors.console import _render_summary, _State

        console, buf = _capture_console()
        state = _State(
            total_nodes=1, tasks_done=60, first_task_at=0.0,
            pool_started_at=0.0, task_latencies=tuple(1.0 for _ in range(60)),
        )
        result = _render_summary(state, now=60.0)
        console.print(result)
        output = buf.getvalue()
        assert "tasks/min" in output

    def test_summary_with_no_tasks(self) -> None:
        from skyward.actors.console import _render_summary, _State

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
        from skyward.api.spec import Image, PoolSpec

        return PoolSpec(nodes=2, accelerator=None, region="us-east-1", image=Image())

    def test_actor_can_be_spawned(self) -> None:
        import asyncio

        from skyward.actors.console import LocalOutput, console_actor

        async def run() -> None:
            from casty import ActorSystem

            spec = self._make_spec()
            async with ActorSystem("test") as system:
                ref = system.spawn(console_actor(spec), "console")
                ref.tell(LocalOutput(line="hello"))
                await asyncio.sleep(0.1)

        asyncio.run(run())

    def test_actor_handles_start_pool(self) -> None:
        import asyncio

        from skyward.actors.console import console_actor

        async def run() -> None:
            from casty import ActorSystem, SpyEvent

            from skyward.actors.messages import StartPool

            spec = self._make_spec()
            async with ActorSystem("test") as system:
                ref = system.spawn(console_actor(spec), "console")
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
        from skyward.actors.console import _format_task

        def train() -> None:
            pass

        result = _format_task(train, (), {})
        assert result == "train"

    def test_with_args(self) -> None:
        from skyward.actors.console import _format_task

        def train() -> None:
            pass

        result = _format_task(train, (10,), {})
        assert result == "train(10)"

    def test_with_kwargs(self) -> None:
        from skyward.actors.console import _format_task

        def train() -> None:
            pass

        result = _format_task(train, (), {"lr": 0.01})
        assert result == "train(lr=0.01)"

    def test_long_signature_truncated(self) -> None:
        from skyward.actors.console import _format_task

        def train() -> None:
            pass

        result = _format_task(train, ("a" * 50,), {})
        assert "\u2026" in result


class TestInstanceIdResolution:
    def test_resolve_from_nodes_by_index(self) -> None:
        from skyward.actors.console import _resolve_instance_id, _State

        state = _State(total_nodes=2)
        assert _resolve_instance_id(state, node_id=0) is None

    def test_resolve_with_instances(self) -> None:
        from unittest.mock import MagicMock

        from skyward.actors.console import _resolve_instance_id, _State

        inst0 = MagicMock()
        inst0.id = "i-abc"
        inst1 = MagicMock()
        inst1.id = "i-def"
        state = _State(total_nodes=2, instances=(inst0, inst1))
        assert _resolve_instance_id(state, node_id=0) == "i-abc"
        assert _resolve_instance_id(state, node_id=1) == "i-def"
        assert _resolve_instance_id(state, node_id=5) is None

    def test_resolve_with_none_node_id(self) -> None:
        from skyward.actors.console import _resolve_instance_id, _State

        state = _State(total_nodes=2)
        assert _resolve_instance_id(state, node_id=None) is None
