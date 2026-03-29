"""Tests for the refactored console renderer — messages, view adapter, actor."""

from __future__ import annotations

from types import MappingProxyType

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestConsoleMessages:
    def test_view_updated(self) -> None:
        from skyward.actors.console.messages import ViewUpdated
        from skyward.api.views import SessionView

        msg = ViewUpdated(view=SessionView())
        assert msg.view is not None

    def test_event_received(self) -> None:
        from skyward.actors.console.messages import EventReceived
        from skyward.api.events import Pool

        ev = Pool.Provisioning(pool_name="p", total_nodes=1, started_at=0.0)
        msg = EventReceived(event=ev)
        assert msg.event is ev

    def test_log_received(self) -> None:
        from skyward.actors.console.messages import LogReceived
        from skyward.api.events import Log

        log = Log.Emitted(pool_name="p", node_id=0, message="hi", level="info")
        msg = LogReceived(log=log)
        assert msg.log.message == "hi"

    def test_local_output(self) -> None:
        from skyward.actors.console.messages import LocalOutput

        msg = LocalOutput(line="hello")
        assert msg.line == "hello"
        assert msg.stream == "stdout"

    def test_console_input_type(self) -> None:
        from skyward.actors.console.messages import (
            ConsoleInput,
            EventReceived,
            LocalOutput,
            LogReceived,
            ViewUpdated,
        )
        from skyward.api.events import Log, Pool
        from skyward.api.views import SessionView

        msgs: list[ConsoleInput] = [  # type: ignore[type-arg]
            ViewUpdated(view=SessionView()),
            EventReceived(event=Pool.Stopped(pool_name="p")),
            LogReceived(log=Log.Emitted(pool_name="p", node_id=0, message="x", level="info")),
            LocalOutput(line="y"),
        ]
        assert len(msgs) == 4


class TestStateFromPoolView:
    def test_converts_basic_pool(self) -> None:
        from skyward.actors.console.view import _state_from_pool_view
        from skyward.api.views import PoolPhase, PoolView, ScalingView, TasksView

        pool = PoolView(
            name="test",
            phase=PoolPhase.READY,
            tasks=TasksView(),
            scaling=ScalingView(),
            total_nodes=2,
            started_at=1.0,
            ready_at=2.0,
        )
        state = _state_from_pool_view(pool)
        assert state.total_nodes == 2
        assert state.ready_at == 2.0

    def test_converts_phase(self) -> None:
        from skyward.actors.console.state import _Phase
        from skyward.actors.console.view import _state_from_pool_view
        from skyward.api.views import PoolPhase, PoolView, ScalingView, TasksView

        for view_phase, expected_phase in [
            (PoolPhase.PROVISIONING, _Phase.PROVISIONING),
            (PoolPhase.SSH, _Phase.SSH),
            (PoolPhase.BOOTSTRAP, _Phase.BOOTSTRAP),
            (PoolPhase.WORKERS, _Phase.WORKERS),
            (PoolPhase.READY, _Phase.READY),
            (PoolPhase.STOPPING, _Phase.STOPPING),
        ]:
            pool = PoolView(
                name="test",
                phase=view_phase,
                tasks=TasksView(),
                scaling=ScalingView(),
                total_nodes=1,
                started_at=1.0,
            )
            state = _state_from_pool_view(pool)
            assert state.phase == expected_phase, f"Failed for {view_phase}"

    def test_converts_nodes(self) -> None:
        from skyward.actors.console.state import _NodeStatus
        from skyward.actors.console.view import _state_from_pool_view
        from skyward.api.views import (
            NodeStatus,
            NodeView,
            PoolPhase,
            PoolView,
            ScalingView,
            TasksView,
        )

        pool = PoolView(
            name="test",
            phase=PoolPhase.READY,
            tasks=TasksView(),
            scaling=ScalingView(),
            total_nodes=1,
            started_at=1.0,
            nodes=MappingProxyType(
                {0: NodeView(node_id=0, status=NodeStatus.READY)}
            ),
        )
        state = _state_from_pool_view(pool)
        assert state.nodes[0] == _NodeStatus.READY

    def test_converts_all_node_statuses(self) -> None:
        from skyward.actors.console.state import _NodeStatus
        from skyward.actors.console.view import _state_from_pool_view
        from skyward.api.views import (
            NodeStatus,
            NodeView,
            PoolPhase,
            PoolView,
            ScalingView,
            TasksView,
        )

        status_map = {
            NodeStatus.WAITING: _NodeStatus.WAITING,
            NodeStatus.SSH: _NodeStatus.SSH,
            NodeStatus.BOOTSTRAPPING: _NodeStatus.BOOTSTRAPPING,
            NodeStatus.READY: _NodeStatus.READY,
        }
        for view_status, expected_status in status_map.items():
            pool = PoolView(
                name="test",
                phase=PoolPhase.READY,
                tasks=TasksView(),
                scaling=ScalingView(),
                total_nodes=1,
                started_at=1.0,
                nodes=MappingProxyType(
                    {0: NodeView(node_id=0, status=view_status)}
                ),
            )
            state = _state_from_pool_view(pool)
            assert state.nodes[0] == expected_status, f"Failed for {view_status}"

    def test_converts_tasks(self) -> None:
        from skyward.actors.console.view import _state_from_pool_view
        from skyward.api.views import (
            PoolPhase,
            PoolView,
            ScalingView,
            TaskEntry,
            TasksView,
        )

        tasks = TasksView(
            queued=2,
            running=1,
            done=5,
            failed=1,
            inflight=MappingProxyType(
                {
                    "t1": TaskEntry(
                        task_id="t1",
                        name="train()",
                        kind="single",
                        started_at=1.0,
                        node_id=0,
                    ),
                }
            ),
        )
        pool = PoolView(
            name="test",
            phase=PoolPhase.READY,
            tasks=tasks,
            scaling=ScalingView(),
            total_nodes=1,
            started_at=1.0,
        )
        state = _state_from_pool_view(pool)
        assert state.tasks_queued == 2
        assert state.tasks_running == 1
        assert state.tasks_done == 5
        assert state.tasks_failed == 1
        assert "t1" in state.inflight

    def test_converts_scaling(self) -> None:
        from skyward.actors.console.view import _state_from_pool_view
        from skyward.api.views import PoolPhase, PoolView, ScalingView, TasksView

        scaling = ScalingView(
            desired=8,
            pending=2,
            draining=1,
            reconciler_state="scaling_up",
            is_elastic=True,
            min_nodes=2,
            max_nodes=16,
        )
        pool = PoolView(
            name="test",
            phase=PoolPhase.READY,
            tasks=TasksView(),
            scaling=scaling,
            total_nodes=4,
            started_at=1.0,
        )
        state = _state_from_pool_view(pool)
        assert state.desired_nodes == 8
        assert state.pending_nodes == 2
        assert state.draining_nodes == 1
        assert state.reconciler_state == "scaling_up"
        assert state.is_elastic is True
        assert state.min_nodes == 2
        assert state.max_nodes == 16

    def test_converts_bootstrap_spinners(self) -> None:
        from skyward.actors.console.view import _state_from_pool_view
        from skyward.api.views import (
            BootstrapView,
            NodeStatus,
            NodeView,
            PoolPhase,
            PoolView,
            ScalingView,
            TasksView,
        )

        bootstrap = BootstrapView(
            phases=("connecting", "apt", "uv"),
            completed=frozenset({"connecting"}),
            active="apt",
            output="installing...",
        )
        pool = PoolView(
            name="test",
            phase=PoolPhase.BOOTSTRAP,
            tasks=TasksView(),
            scaling=ScalingView(),
            total_nodes=1,
            started_at=1.0,
            nodes=MappingProxyType(
                {
                    0: NodeView(
                        node_id=0,
                        status=NodeStatus.BOOTSTRAPPING,
                        bootstrap=bootstrap,
                    )
                }
            ),
        )
        state = _state_from_pool_view(pool)
        assert 0 in state.bootstrap_spinners
        timeline = state.bootstrap_spinners[0]
        assert timeline.active == "apt"
        assert timeline.output == "installing..."
        assert "connecting" in timeline.completed

    def test_converts_metrics(self) -> None:
        from skyward.actors.console.view import _state_from_pool_view
        from skyward.api.views import (
            NodeStatus,
            NodeView,
            PoolPhase,
            PoolView,
            ScalingView,
            TasksView,
        )

        pool = PoolView(
            name="test",
            phase=PoolPhase.READY,
            tasks=TasksView(),
            scaling=ScalingView(),
            total_nodes=1,
            started_at=1.0,
            nodes=MappingProxyType(
                {
                    0: NodeView(
                        node_id=0,
                        status=NodeStatus.READY,
                        metrics=MappingProxyType({"gpu_util": 85.0, "cpu": 50.0}),
                    )
                }
            ),
        )
        state = _state_from_pool_view(pool)
        assert state.metrics[0]["gpu_util"] == 85.0
        assert state.metrics[0]["cpu"] == 50.0

    def test_converts_cluster_and_instances(self) -> None:
        from unittest.mock import MagicMock

        from skyward.actors.console.view import _state_from_pool_view
        from skyward.api.views import PoolPhase, PoolView, ScalingView, TasksView

        cluster = MagicMock()
        cluster.ssh_user = "ubuntu"
        cluster.ssh_key_path = "/tmp/key"
        inst = MagicMock()
        pool = PoolView(
            name="test",
            phase=PoolPhase.READY,
            tasks=TasksView(),
            scaling=ScalingView(),
            total_nodes=1,
            started_at=1.0,
            cluster=cluster,
            instances=(inst,),
        )
        state = _state_from_pool_view(pool)
        assert state.cluster is cluster
        assert state.instances == (inst,)
        assert state.ssh_user == "ubuntu"
        assert state.ssh_key_path == "/tmp/key"

    def test_task_fn_stats_and_latencies(self) -> None:
        from skyward.actors.console.view import _state_from_pool_view
        from skyward.api.views import PoolPhase, PoolView, ScalingView, TasksView

        tasks = TasksView(
            done=3,
            latencies=(1.0, 2.0, 3.0),
            fn_stats=MappingProxyType({"train": (1.0, 2.0, 3.0)}),
            fn_failed=MappingProxyType({"train": 1}),
            first_task_at=100.0,
            tasks_per_node=MappingProxyType({0: 2, 1: 1}),
        )
        pool = PoolView(
            name="test",
            phase=PoolPhase.READY,
            tasks=tasks,
            scaling=ScalingView(),
            total_nodes=2,
            started_at=1.0,
        )
        state = _state_from_pool_view(pool)
        assert state.task_latencies == (1.0, 2.0, 3.0)
        assert state.task_fn_stats["train"] == (1.0, 2.0, 3.0)
        assert state.task_fn_failed["train"] == 1
        assert state.first_task_at == 100.0
        assert state.tasks_per_node[0] == 2


class TestThroughputInState:
    """Verify _throughput was moved from model.py to state.py."""

    def test_throughput_in_state_module(self) -> None:
        from skyward.actors.console.state import _throughput

        assert callable(_throughput)

    def test_throughput_computes_correctly(self) -> None:
        from skyward.actors.console.state import _State, _throughput

        state = _State(
            total_nodes=1,
            tasks_done=10,
            first_task_at=100.0,
            pool_started_at=0.0,
        )
        rate = _throughput(state, now=160.0)
        assert rate == 10.0

    def test_throughput_zero_when_no_tasks(self) -> None:
        from skyward.actors.console.state import _State, _throughput

        state = _State(total_nodes=1)
        assert _throughput(state) == 0.0
