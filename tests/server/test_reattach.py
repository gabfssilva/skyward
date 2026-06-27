"""Tests for server-side reattach persist + boot re-adoption glue."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("starlette")

from skyward.actors.snapshot import (  # noqa: E402
    NodeSnapshot,
    NodeStatus,
    PoolPhase,
    PoolSnapshot,
    ScalingSnapshot,
    TaskCounters,
)
from skyward.core.session_store import NodeHandle, SessionHandle, pack_payload  # noqa: E402
from skyward.server import reattach  # noqa: E402
from skyward.server.state import ServerState  # noqa: E402


class _FakeInstance:
    def __init__(self, iid: str) -> None:
        self.id = iid


class _FakeCluster:
    def __init__(self, instances: tuple[_FakeInstance, ...]) -> None:
        self.instances = instances

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _instance(iid: str, ip: str):
    from skyward.api.model import Instance, InstanceType, Offer

    itype = InstanceType(
        name="t", accelerator=None, vcpus=1, memory_gb=1,
        architecture="x86_64", specific=None,
    )
    offer = Offer(
        id="o", instance_type=itype, spot_price=1.0, on_demand_price=2.0,
        billing_unit="hour", specific=None,
    )
    return Instance(id=iid, status="ready", offer=offer, ip=ip, private_ip="10.0.0.1", ssh_port=22)


def _snapshot_pool():
    from skyward.core.pool import ComputePool

    inst0 = _instance("i-0", "1.1.1.1")
    inst1 = _instance("i-1", "2.2.2.2")
    cluster = MagicMock()
    cluster.id = "c-1"
    cluster.prebaked = False
    cluster.ssh_user = "ubuntu"
    cluster.ssh_key_path = "/k"
    snap = PoolSnapshot(
        name="p", phase=PoolPhase.READY,
        nodes=(NodeSnapshot(0, "i-0", NodeStatus.READY), NodeSnapshot(1, "i-1", NodeStatus.READY)),
        tasks=TaskCounters(), scaling=ScalingSnapshot(),
        cluster=cluster, instances=(inst0, inst1),
    )

    class _Pool(ComputePool):
        def __init__(self):
            pass

        def snapshot(self):
            return snap

    return _Pool()


def test_persist_handle_writes_rank_ordered_nodes(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("skyward.server.reattach.write_handle", lambda h, **k: captured.update(h=h))
    monkeypatch.setattr("skyward.server.reattach.pack_payload", lambda cfg, cluster: b"PAYLOAD")

    reattach.persist_handle("p", provider_config={"cfg": 1}, pool=_snapshot_pool())

    h = captured["h"]
    assert h.name == "p"
    assert [n.node_id for n in h.nodes] == [0, 1]
    assert h.nodes[0].ssh_user == "ubuntu"
    assert h.nodes[0].ip == "1.1.1.1"
    assert h.payload == b"PAYLOAD"


def _node(iid: str) -> NodeHandle:
    return NodeHandle(
        node_id=0, instance_id=iid, ip="1.1.1.1", private_ip=None,
        ssh_port=22, ssh_user="u", ssh_key_path="/k", ssh_password=None,
    )


def _handle(name: str) -> SessionHandle:
    cluster = _FakeCluster((_FakeInstance("i-0"),))
    return SessionHandle(
        version=1, name=name, created_at="t", cluster_id="c", prebaked=False,
        head_node_id=0, nodes=(_node("i-0"),), payload=pack_payload({"cfg": 1}, cluster),
    )


def test_reattach_pools_adopts_and_registers(monkeypatch):
    monkeypatch.setattr("skyward.server.reattach.list_handles", lambda **k: (_handle("a"),))
    state = ServerState(session=MagicMock())
    fake_pool = object()
    state.session.adopt = MagicMock(return_value=fake_pool)
    state.session.projection.subscribe = MagicMock(return_value=lambda: None)

    reattach.reattach_pools(state)

    entry = state.get_pool("a")
    assert entry is not None
    assert entry.pool is fake_pool
    assert entry.status == "ready"
    state.session.adopt.assert_called_once()
    kwargs = state.session.adopt.call_args.kwargs
    assert kwargs["node_ids"] == (0,)
    assert len(kwargs["instances"]) == 1


def test_reattach_pools_cleans_up_dead(monkeypatch):
    removed: list[str] = []
    monkeypatch.setattr("skyward.server.reattach.list_handles", lambda **k: (_handle("a"),))
    monkeypatch.setattr("skyward.server.reattach.remove_handle", lambda n: removed.append(n))
    state = ServerState(session=MagicMock())
    state.session.adopt = MagicMock(side_effect=RuntimeError("instances gone"))
    state.session.discard = MagicMock()

    reattach.reattach_pools(state)

    assert state.get_pool("a") is None
    assert removed == ["a"]
    state.session.discard.assert_called_once()


def test_drop_persistence_unsubscribes_and_removes(monkeypatch):
    removed: list[str] = []
    monkeypatch.setattr("skyward.server.reattach.remove_handle", lambda n: removed.append(n))
    state = ServerState(session=MagicMock())
    unsubbed = []
    state.reattach_unsubs["a"] = lambda: unsubbed.append(True)

    reattach.drop_persistence(state, "a")

    assert unsubbed == [True]
    assert removed == ["a"]
    assert "a" not in state.reattach_unsubs
