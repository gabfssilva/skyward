"""Tests for the view hierarchy — read-only session state snapshots."""

from __future__ import annotations

from types import MappingProxyType

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestBootstrapView:
    def test_construction(self) -> None:
        from skyward.api.views import BootstrapView

        bv = BootstrapView(
            phases=("apt", "pip", "worker"),
            completed=frozenset({"apt"}),
            active="pip",
            output="installing packages...",
        )
        assert bv.phases == ("apt", "pip", "worker")
        assert bv.completed == frozenset({"apt"})
        assert bv.active == "pip"
        assert bv.output == "installing packages..."

    def test_frozen(self) -> None:
        from skyward.api.views import BootstrapView

        bv = BootstrapView(
            phases=("apt",), completed=frozenset(), active="apt", output=""
        )
        with pytest.raises(AttributeError):
            bv.active = "pip"  # type: ignore[misc]


class TestNodeView:
    def test_defaults(self) -> None:
        from skyward.api.views import NodeStatus, NodeView

        nv = NodeView(node_id=0, status=NodeStatus.WAITING)
        assert nv.node_id == 0
        assert nv.status == NodeStatus.WAITING
        assert nv.instance is None
        assert nv.bootstrap is None
        assert isinstance(nv.metrics, MappingProxyType)
        assert len(nv.metrics) == 0

    def test_frozen(self) -> None:
        from skyward.api.views import NodeStatus, NodeView

        nv = NodeView(node_id=0, status=NodeStatus.WAITING)
        with pytest.raises(AttributeError):
            nv.node_id = 5  # type: ignore[misc]

    def test_with_bootstrap(self) -> None:
        from skyward.api.views import BootstrapView, NodeStatus, NodeView

        bv = BootstrapView(
            phases=("apt", "pip"), completed=frozenset(), active="apt", output=""
        )
        nv = NodeView(node_id=1, status=NodeStatus.BOOTSTRAPPING, bootstrap=bv)
        assert nv.bootstrap is not None
        assert nv.bootstrap.active == "apt"

    def test_with_metrics(self) -> None:
        from skyward.api.views import NodeStatus, NodeView

        metrics = MappingProxyType({"gpu_util": 85.0, "mem_used": 32.5})
        nv = NodeView(node_id=2, status=NodeStatus.READY, metrics=metrics)
        assert nv.metrics["gpu_util"] == 85.0
        assert nv.metrics["mem_used"] == 32.5

    def test_default_metrics_share_identity(self) -> None:
        from skyward.api.views import NodeStatus, NodeView

        nv1 = NodeView(node_id=0, status=NodeStatus.WAITING)
        nv2 = NodeView(node_id=1, status=NodeStatus.WAITING)
        assert nv1.metrics is nv2.metrics


class TestNodeStatus:
    def test_values(self) -> None:
        from skyward.api.views import NodeStatus

        names = [s.name for s in NodeStatus]
        assert names == ["WAITING", "SSH", "BOOTSTRAPPING", "READY"]


class TestPoolPhase:
    def test_values(self) -> None:
        from skyward.api.views import PoolPhase

        names = [p.name for p in PoolPhase]
        assert names == [
            "PROVISIONING", "SSH", "BOOTSTRAP", "WORKERS", "READY", "STOPPING",
        ]


class TestTaskEntry:
    def test_construction(self) -> None:
        from skyward.api.views import TaskEntry

        te = TaskEntry(task_id="abc", name="train", kind="single", started_at=1000.0)
        assert te.task_id == "abc"
        assert te.name == "train"
        assert te.kind == "single"
        assert te.started_at == 1000.0
        assert te.node_id == -1
        assert te.broadcast_total == 0
        assert te.broadcast_done == 0

    def test_broadcast_entry(self) -> None:
        from skyward.api.views import TaskEntry

        te = TaskEntry(
            task_id="bcast-1",
            name="eval",
            kind="broadcast",
            started_at=2000.0,
            broadcast_total=4,
            broadcast_done=2,
        )
        assert te.broadcast_total == 4
        assert te.broadcast_done == 2


class TestTasksView:
    def test_defaults(self) -> None:
        from skyward.api.views import TasksView

        tv = TasksView()
        assert tv.queued == 0
        assert tv.running == 0
        assert tv.done == 0
        assert tv.failed == 0
        assert isinstance(tv.inflight, MappingProxyType)
        assert len(tv.inflight) == 0
        assert tv.latencies == ()
        assert isinstance(tv.fn_stats, MappingProxyType)
        assert len(tv.fn_stats) == 0
        assert isinstance(tv.fn_failed, MappingProxyType)
        assert len(tv.fn_failed) == 0
        assert tv.first_task_at == 0.0
        assert tv.throughput == 0.0
        assert isinstance(tv.tasks_per_node, MappingProxyType)
        assert len(tv.tasks_per_node) == 0

    def test_frozen(self) -> None:
        from skyward.api.views import TasksView

        tv = TasksView()
        with pytest.raises(AttributeError):
            tv.queued = 5  # type: ignore[misc]

    def test_default_maps_share_identity(self) -> None:
        from skyward.api.views import TasksView

        tv1 = TasksView()
        tv2 = TasksView()
        assert tv1.inflight is tv2.inflight
        assert tv1.fn_stats is tv2.fn_stats
        assert tv1.fn_failed is tv2.fn_failed
        assert tv1.tasks_per_node is tv2.tasks_per_node


class TestScalingView:
    def test_defaults(self) -> None:
        from skyward.api.views import ScalingView

        sv = ScalingView()
        assert sv.desired == 0
        assert sv.pending == 0
        assert sv.draining == 0
        assert sv.reconciler_state == "watching"
        assert sv.is_elastic is False
        assert sv.min_nodes is None
        assert sv.max_nodes is None

    def test_elastic(self) -> None:
        from skyward.api.views import ScalingView

        sv = ScalingView(
            desired=4, is_elastic=True, min_nodes=2, max_nodes=8,
        )
        assert sv.is_elastic is True
        assert sv.min_nodes == 2
        assert sv.max_nodes == 8


class TestPoolView:
    def test_required_fields_only(self) -> None:
        from skyward.api.views import PoolPhase, PoolView, ScalingView, TasksView

        pv = PoolView(
            name="test-pool",
            phase=PoolPhase.PROVISIONING,
            tasks=TasksView(),
            scaling=ScalingView(),
        )
        assert pv.name == "test-pool"
        assert pv.phase == PoolPhase.PROVISIONING
        assert pv.total_nodes == 0
        assert isinstance(pv.nodes, MappingProxyType)
        assert len(pv.nodes) == 0
        assert pv.cluster is None
        assert pv.instances == ()
        assert pv.started_at == 0.0
        assert pv.ready_at == 0.0
        assert pv.spec is None

    def test_frozen(self) -> None:
        from skyward.api.views import PoolPhase, PoolView, ScalingView, TasksView

        pv = PoolView(
            name="p", phase=PoolPhase.READY, tasks=TasksView(), scaling=ScalingView(),
        )
        with pytest.raises(AttributeError):
            pv.phase = PoolPhase.STOPPING  # type: ignore[misc]

    def test_with_nodes(self) -> None:
        from skyward.api.views import (
            NodeStatus,
            NodeView,
            PoolPhase,
            PoolView,
            ScalingView,
            TasksView,
        )

        nodes = MappingProxyType({
            0: NodeView(node_id=0, status=NodeStatus.READY),
            1: NodeView(node_id=1, status=NodeStatus.SSH),
        })
        pv = PoolView(
            name="p",
            phase=PoolPhase.SSH,
            tasks=TasksView(),
            scaling=ScalingView(),
            total_nodes=2,
            nodes=nodes,
        )
        assert len(pv.nodes) == 2
        assert pv.nodes[0].status == NodeStatus.READY
        assert pv.nodes[1].status == NodeStatus.SSH


class TestSessionView:
    def test_empty(self) -> None:
        from skyward.api.views import SessionView

        sv = SessionView()
        assert isinstance(sv.pools, MappingProxyType)
        assert len(sv.pools) == 0

    def test_with_pool(self) -> None:
        from skyward.api.views import (
            PoolPhase,
            PoolView,
            ScalingView,
            SessionView,
            TasksView,
        )

        pv = PoolView(
            name="my-pool",
            phase=PoolPhase.READY,
            tasks=TasksView(),
            scaling=ScalingView(),
        )
        sv = SessionView(pools=MappingProxyType({"my-pool": pv}))
        assert "my-pool" in sv.pools
        assert sv.pools["my-pool"].phase == PoolPhase.READY

    def test_frozen(self) -> None:
        from skyward.api.views import SessionView

        sv = SessionView()
        with pytest.raises(AttributeError):
            sv.pools = MappingProxyType({})  # type: ignore[misc]
