"""Tests for vscode.sidecar.serialize — view-to-dict conversion."""

from __future__ import annotations

from types import MappingProxyType

from pytest import approx

from skyward.accelerators import Accelerator
from skyward.api.events import Log, Node, Task
from skyward.api.model import Instance, InstanceType, Offer
from skyward.api.views import (
    BootstrapView,
    NodeStatus,
    NodeView,
    PoolPhase,
    PoolView,
    ScalingView,
    TaskEntry,
    TasksView,
)
from vscode.sidecar.serialize import (
    _camel_to_snake,
    _node,
    _tasks,
    event_to_dict,
    pool_view_to_dict,
)

# ── Fixtures ────────────────────────────────────────────────────


def _make_instance(
    ip: str = "10.0.0.1",
    accel_name: str = "A100",
) -> Instance:
    accel = Accelerator(name=accel_name, memory="80GB")
    itype = InstanceType(
        name="p4d.24xlarge",
        accelerator=accel,
        vcpus=96,
        memory_gb=1152,
        architecture="x86_64",
        specific=None,
    )
    offer = Offer(
        id="offer-1",
        instance_type=itype,
        spot_price=10.0,
        on_demand_price=32.77,
        billing_unit="hour",
        specific=None,
    )
    return Instance(
        id="i-abc123",
        status="ready",
        offer=offer,
        ip=ip,
        private_ip="172.16.0.1",
    )


def _make_pool_view(
    *,
    phase: PoolPhase = PoolPhase.READY,
    nodes: MappingProxyType[int, NodeView] | None = None,
    tasks: TasksView | None = None,
    scaling: ScalingView | None = None,
) -> PoolView:
    if nodes is None:
        nodes = MappingProxyType(
            {0: NodeView(node_id=0, status=NodeStatus.READY)}
        )
    return PoolView(
        name="train",
        phase=phase,
        tasks=tasks or TasksView(),
        scaling=scaling or ScalingView(),
        total_nodes=len(nodes),
        nodes=nodes,
    )


# ── pool_view_to_dict ──────────────────────────────────────────


class TestPoolViewToDict:
    def test_basic_fields(self) -> None:
        view = _make_pool_view()
        d = pool_view_to_dict(view)

        assert d["name"] == "train"
        assert d["phase"] == "ready"
        assert d["total_nodes"] == 1
        assert d["started_at"] == 0.0
        assert d["ready_at"] == 0.0
        assert d["cost_per_hour"] == 0.0
        assert d["cost_total"] == 0.0

    def test_phase_enum_to_lowercase(self) -> None:
        for phase in PoolPhase:
            view = _make_pool_view(phase=phase)
            d = pool_view_to_dict(view)
            assert d["phase"] == phase.name.lower()

    def test_nodes_keyed_as_strings(self) -> None:
        nodes = MappingProxyType({
            0: NodeView(node_id=0, status=NodeStatus.READY),
            3: NodeView(node_id=3, status=NodeStatus.WAITING),
        })
        d = pool_view_to_dict(_make_pool_view(nodes=nodes))

        assert "0" in d["nodes"]
        assert "3" in d["nodes"]
        assert d["nodes"]["0"]["status"] == "ready"
        assert d["nodes"]["3"]["status"] == "waiting"

    def test_scaling_section(self) -> None:
        sc = ScalingView(
            desired=4,
            pending=1,
            draining=0,
            reconciler_state="scaling_up",
            is_elastic=True,
            min_nodes=2,
            max_nodes=8,
        )
        d = pool_view_to_dict(_make_pool_view(scaling=sc))

        assert d["scaling"]["desired"] == 4
        assert d["scaling"]["pending"] == 1
        assert d["scaling"]["is_elastic"] is True
        assert d["scaling"]["min_nodes"] == 2
        assert d["scaling"]["max_nodes"] == 8
        assert d["scaling"]["reconciler_state"] == "scaling_up"

    def test_logs_default_empty(self) -> None:
        d = pool_view_to_dict(_make_pool_view())
        assert d["logs"] == []

    def test_logs_passthrough(self) -> None:
        logs = [
            {"node_id": 0, "message": "started", "level": "info"},
            {"node_id": 1, "message": "OOM", "level": "error"},
        ]
        d = pool_view_to_dict(_make_pool_view(), logs=logs)
        assert d["logs"] == logs
        assert len(d["logs"]) == 2

    def test_tasks_with_latencies_and_fn_stats(self) -> None:
        tv = TasksView(
            queued=1,
            running=2,
            done=10,
            failed=1,
            inflight=MappingProxyType({}),
            latencies=(0.5, 1.0, 1.5),
            fn_stats=MappingProxyType({"fit": (0.5, 1.0, 1.5)}),
            fn_failed=MappingProxyType({"fit": 1}),
            first_task_at=100.0,
            throughput=5.0,
            tasks_per_node=MappingProxyType({0: 5, 1: 5}),
        )
        d = pool_view_to_dict(_make_pool_view(tasks=tv))
        t = d["tasks"]

        assert t["queued"] == 1
        assert t["running"] == 2
        assert t["done"] == 10
        assert t["failed"] == 1
        assert t["avg_latency"] == 1.0
        assert t["throughput"] == 5.0
        assert t["first_task_at"] == 100.0
        assert "fit" in t["fn_summary"]
        assert t["fn_summary"]["fit"]["calls"] == 3
        assert t["fn_summary"]["fit"]["min"] == 0.5
        assert t["fn_summary"]["fit"]["max"] == 1.5
        assert t["fn_summary"]["fit"]["avg"] == 1.0
        assert t["fn_summary"]["fit"]["failed"] == 1
        assert t["tasks_per_node"]["0"] == 5
        assert t["tasks_per_node"]["1"] == 5

    def test_raw_latencies_not_sent(self) -> None:
        tv = TasksView(latencies=(0.5, 1.0))
        d = pool_view_to_dict(_make_pool_view(tasks=tv))
        assert "latencies" not in d["tasks"]


# ── _node ───────────────────────────────────────────────────────


class TestNodeSerialization:
    def test_basic_node_no_instance(self) -> None:
        nv = NodeView(node_id=0, status=NodeStatus.READY)
        d = _node(nv)

        assert d["node_id"] == 0
        assert d["status"] == "ready"
        assert d["metrics"] == {}
        assert "ip" not in d
        assert "accelerator" not in d
        assert "bootstrap" not in d

    def test_node_with_instance(self) -> None:
        inst = _make_instance(ip="10.0.0.42", accel_name="H100")
        nv = NodeView(node_id=1, status=NodeStatus.READY, instance=inst)
        d = _node(nv)

        assert d["ip"] == "10.0.0.42"
        assert d["accelerator"] == "H100"

    def test_node_instance_falls_back_to_private_ip(self) -> None:
        inst = _make_instance(ip="")
        nv = NodeView(node_id=2, status=NodeStatus.SSH, instance=inst)
        d = _node(nv)

        # empty string is falsy, falls back to private_ip
        assert d["ip"] == "172.16.0.1"

    def test_node_instance_none_ip(self) -> None:
        accel = Accelerator(name="T4")
        itype = InstanceType(
            name="g4dn.xlarge",
            accelerator=accel,
            vcpus=4,
            memory_gb=16,
            architecture="x86_64",
            specific=None,
        )
        offer = Offer(
            id="offer-2",
            instance_type=itype,
            spot_price=None,
            on_demand_price=0.526,
            billing_unit="hour",
            specific=None,
        )
        inst = Instance(
            id="i-xyz",
            status="provisioning",
            offer=offer,
            ip=None,
            private_ip=None,
        )
        nv = NodeView(node_id=0, status=NodeStatus.WAITING, instance=inst)
        d = _node(nv)

        assert d["ip"] is None

    def test_node_instance_no_accelerator(self) -> None:
        itype = InstanceType(
            name="c5.xlarge",
            accelerator=None,
            vcpus=4,
            memory_gb=8,
            architecture="x86_64",
            specific=None,
        )
        offer = Offer(
            id="offer-cpu",
            instance_type=itype,
            spot_price=None,
            on_demand_price=0.17,
            billing_unit="hour",
            specific=None,
        )
        inst = Instance(
            id="i-cpu",
            status="ready",
            offer=offer,
            ip="10.0.0.5",
        )
        nv = NodeView(node_id=0, status=NodeStatus.READY, instance=inst)
        d = _node(nv)

        assert d["accelerator"] is None

    def test_node_with_bootstrap(self) -> None:
        bv = BootstrapView(
            phases=("apt", "uv", "pip"),
            completed=frozenset({"apt", "uv"}),
            active="pip",
            output="Installing packages...",
        )
        nv = NodeView(
            node_id=0,
            status=NodeStatus.BOOTSTRAPPING,
            bootstrap=bv,
        )
        d = _node(nv)

        assert d["bootstrap"]["phases"] == ["apt", "uv", "pip"]
        assert set(d["bootstrap"]["completed"]) == {"apt", "uv"}
        assert d["bootstrap"]["active"] == "pip"
        assert d["bootstrap"]["output"] == "Installing packages..."

    def test_node_status_enum_values(self) -> None:
        for st in NodeStatus:
            nv = NodeView(node_id=0, status=st)
            d = _node(nv)
            assert d["status"] == st.name.lower()

    def test_node_with_metrics(self) -> None:
        nv = NodeView(
            node_id=0,
            status=NodeStatus.READY,
            metrics=MappingProxyType({"gpu_util": 0.87, "mem_util": 0.63}),
        )
        d = _node(nv)

        assert d["metrics"] == {"gpu_util": 0.87, "mem_util": 0.63}
        assert isinstance(d["metrics"], dict)


# ── _tasks ──────────────────────────────────────────────────────


class TestTasksSerialization:
    def test_empty_tasks(self) -> None:
        tv = TasksView()
        d = _tasks(tv)

        assert d["queued"] == 0
        assert d["running"] == 0
        assert d["done"] == 0
        assert d["failed"] == 0
        assert d["avg_latency"] == 0.0
        assert d["throughput"] == 0.0
        assert d["fn_summary"] == {}
        assert d["inflight"] == {}
        assert d["tasks_per_node"] == {}

    def test_inflight_entries(self) -> None:
        entry = TaskEntry(
            task_id="t-1",
            name="fit",
            kind="rshift",
            started_at=1000.0,
            node_id=2,
            broadcast_total=0,
            broadcast_done=0,
        )
        tv = TasksView(
            running=1,
            inflight=MappingProxyType({"t-1": entry}),
        )
        d = _tasks(tv)

        assert "t-1" in d["inflight"]
        inf = d["inflight"]["t-1"]
        assert inf["task_id"] == "t-1"
        assert inf["name"] == "fit"
        assert inf["kind"] == "rshift"
        assert inf["started_at"] == 1000.0
        assert inf["node_id"] == 2
        assert inf["broadcast_total"] == 0

    def test_fn_stats_multiple_functions(self) -> None:
        tv = TasksView(
            fn_stats=MappingProxyType({
                "train": (1.0, 2.0, 3.0),
                "eval": (0.1, 0.2),
            }),
            fn_failed=MappingProxyType({"train": 0, "eval": 1}),
        )
        d = _tasks(tv)

        assert d["fn_summary"]["train"]["calls"] == 3
        assert d["fn_summary"]["train"]["avg"] == 2.0
        assert d["fn_summary"]["train"]["min"] == 1.0
        assert d["fn_summary"]["train"]["max"] == 3.0
        assert d["fn_summary"]["train"]["failed"] == 0

        assert d["fn_summary"]["eval"]["calls"] == 2
        assert d["fn_summary"]["eval"]["avg"] == approx(0.15)
        assert d["fn_summary"]["eval"]["min"] == 0.1
        assert d["fn_summary"]["eval"]["max"] == 0.2
        assert d["fn_summary"]["eval"]["failed"] == 1

    def test_fn_failed_default_zero(self) -> None:
        tv = TasksView(
            fn_stats=MappingProxyType({"infer": (0.5,)}),
            fn_failed=MappingProxyType({}),
        )
        d = _tasks(tv)
        assert d["fn_summary"]["infer"]["failed"] == 0

    def test_avg_latency_rounding(self) -> None:
        tv = TasksView(latencies=(1.0, 2.0, 3.0))
        d = _tasks(tv)
        assert d["avg_latency"] == 2.0

        tv2 = TasksView(latencies=(0.1, 0.2, 0.3))
        d2 = _tasks(tv2)
        assert d2["avg_latency"] == 0.2

    def test_tasks_per_node_keys_are_strings(self) -> None:
        tv = TasksView(
            tasks_per_node=MappingProxyType({0: 10, 1: 7, 2: 3}),
        )
        d = _tasks(tv)
        assert d["tasks_per_node"] == {"0": 10, "1": 7, "2": 3}


# ── event_to_dict ───────────────────────────────────────────────


class TestEventToDict:
    def test_task_completed(self) -> None:
        ev = Task.Completed(
            pool_name="train",
            task_id="t-42",
            node_id=1,
            elapsed=2.5,
        )
        d = event_to_dict(ev)

        assert d is not None
        assert d["event"] == "task.completed"
        assert d["pool"] == "train"
        assert d["data"]["task_id"] == "t-42"
        assert d["data"]["node_id"] == 1
        assert d["data"]["elapsed"] == 2.5
        assert "pool_name" not in d["data"]

    def test_node_ready(self) -> None:
        ev = Node.Ready(pool_name="train", node_id=3)
        d = event_to_dict(ev)

        assert d is not None
        assert d["event"] == "node.ready"
        assert d["pool"] == "train"
        assert d["data"]["node_id"] == 3

    def test_node_lost(self) -> None:
        ev = Node.Lost(pool_name="eval", node_id=0, reason="preempted")
        d = event_to_dict(ev)

        assert d is not None
        assert d["event"] == "node.lost"
        assert d["data"]["reason"] == "preempted"

    def test_log_emitted(self) -> None:
        ev = Log.Emitted(
            pool_name="train",
            node_id=0,
            message="epoch 1 done",
            level="info",
        )
        d = event_to_dict(ev)

        assert d is not None
        assert d["event"] == "log.emitted"
        assert d["data"]["message"] == "epoch 1 done"
        assert d["data"]["level"] == "info"
        assert d["data"]["node_id"] == 0

    def test_task_failed(self) -> None:
        ev = Task.Failed(
            pool_name="train",
            task_id="t-99",
            node_id=2,
            error="OOM",
        )
        d = event_to_dict(ev)

        assert d is not None
        assert d["event"] == "task.failed"
        assert d["data"]["error"] == "OOM"

    def test_pool_name_excluded_from_data(self) -> None:
        ev = Node.Ready(pool_name="train", node_id=0)
        d = event_to_dict(ev)

        assert d is not None
        assert "pool_name" not in d["data"]

    def test_none_for_object_without_pool_name(self) -> None:
        assert event_to_dict("not an event") is None
        assert event_to_dict(42) is None
        assert event_to_dict(object()) is None

    def test_bootstrap_nested_event(self) -> None:
        ev = Node.Bootstrap.Started(
            pool_name="train", node_id=0, phase="apt"
        )
        d = event_to_dict(ev)

        assert d is not None
        # Bootstrap.Started is nested: Node > Bootstrap > Started
        # qualname is "Bootstrap.Started", so event = "bootstrap.started"
        assert d["event"] == "bootstrap.started"
        assert d["data"]["node_id"] == 0
        assert d["data"]["phase"] == "apt"

    def test_scaling_desired_changed(self) -> None:
        from skyward.api.events import Scaling

        ev = Scaling.DesiredChanged(
            pool_name="train", desired=6, reason="pressure"
        )
        d = event_to_dict(ev)

        assert d is not None
        assert d["event"] == "scaling.desired_changed"
        assert d["data"]["desired"] == 6
        assert d["data"]["reason"] == "pressure"

    def test_error_occurred(self) -> None:
        from skyward.api.events import Error

        ev = Error.Occurred(
            pool_name="train", message="SSH timeout", fatal=True
        )
        d = event_to_dict(ev)

        assert d is not None
        assert d["event"] == "error.occurred"
        assert d["data"]["message"] == "SSH timeout"
        assert d["data"]["fatal"] is True


# ── _camel_to_snake ─────────────────────────────────────────────


class TestCamelToSnake:
    def test_simple(self) -> None:
        assert _camel_to_snake("Ready") == "ready"

    def test_two_words(self) -> None:
        assert _camel_to_snake("PhaseChanged") == "phase_changed"

    def test_three_words(self) -> None:
        assert _camel_to_snake("DesiredChanged") == "desired_changed"

    def test_already_lowercase(self) -> None:
        assert _camel_to_snake("ready") == "ready"

    def test_consecutive_caps(self) -> None:
        assert _camel_to_snake("OOMKilled") == "o_o_m_killed"

    def test_broadcast_partial(self) -> None:
        assert _camel_to_snake("BroadcastPartial") == "broadcast_partial"

    def test_provision_failed(self) -> None:
        assert _camel_to_snake("ProvisionFailed") == "provision_failed"
