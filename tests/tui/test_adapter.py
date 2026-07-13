"""Tests for the SessionView -> UiCluster adapter and ProjectionSource (Phase 2)."""

from __future__ import annotations

import pytest

from skyward.accelerators import Accelerator
from skyward.api.events import Log, Metric, Node, Pool
from skyward.api.model import Cluster, Instance, InstanceType, Offer
from skyward.api.projection import SessionProjection
from skyward.api.spec import Nodes, PoolSpec
from skyward.tui.adapter import view_to_ui
from skyward.tui.model import UiStatus
from skyward.tui.sources import ProjectionSource

pytestmark = pytest.mark.unit


def _projection(total: int = 2) -> SessionProjection:
    proj = SessionProjection()
    proj.handle(Pool.Provisioning(pool_name="train", total_nodes=total, started_at=1000.0))
    return proj


def _instance(
    *, iid: str = "i-0", region: str = "us-east-1", spot: bool = True,
    spot_price: float | None = 1.92, on_demand_price: float | None = 7.56,
) -> Instance:
    itype = InstanceType(
        name="h100.8x",
        accelerator=Accelerator(name="H100", memory="80GB", count=8),
        vcpus=96, memory_gb=1152, architecture="x86_64", specific=None,
    )
    offer = Offer(
        id="offer-1", instance_type=itype, spot_price=spot_price,
        on_demand_price=on_demand_price, billing_unit="hour", specific=None,
    )
    return Instance(id=iid, status="ready", offer=offer, region=region, spot=spot)


def _cluster() -> Cluster:
    spec = PoolSpec(
        nodes=Nodes(desired=2),
        accelerator=Accelerator(name="H100", memory="80GB", count=8),
        region="us-east-1", allocation="spot", provider="vastai",
    )
    return Cluster(
        id="c1", status="ready", spec=spec, offer=_instance().offer,
        ssh_key_path="", ssh_user="root", use_sudo=False,
        shutdown_command="", specific=None,
    )


# ── view_to_ui: status, role, metrics ───────────────────────────


def test_status_role_and_metrics_map() -> None:
    proj = _projection(2)
    proj.handle(Node.Connected(pool_name="train", node_id=0, instance=None))
    proj.handle(Node.Ready(pool_name="train", node_id=0))
    for metric, value in (("gpu_util", 91.0), ("gpu_mem_mb", 61440.0), ("gpu_temp", 63.0)):
        proj.handle(Metric.Sampled(pool_name="train", node_id=0, name=metric, value=value))

    proj.handle(Node.Connected(pool_name="train", node_id=1, instance=None))
    proj.handle(Node.Bootstrap.Started(pool_name="train", node_id=1, phase="apt"))

    pool = proj.view.pools["train"]
    ui = view_to_ui(pool, t=5, base_elapsed=0, ready_ticks={0: 0})

    assert [n.id for n in ui.nodes] == ["node-0", "node-1"]
    n0, n1 = ui.nodes
    assert n0.status is UiStatus.READY
    assert n0.role == "coord"
    assert round(n0.gpu) == 91
    assert round(n0.mem) == 60  # 61440 MB / 1024
    assert round(n0.temp) == 63
    assert n0.started_at == 0

    # SSH + active bootstrap collapses to BOOTSTRAPPING with a phase bar
    assert n1.status is UiStatus.BOOTSTRAPPING
    assert n1.role == "worker"
    assert "apt" in n1.phases
    assert n1.phases[n1.phase_idx] == "apt"
    assert ui.ready_count == 1


def test_multi_gpu_metrics_are_averaged() -> None:
    proj = _projection(1)
    proj.handle(Node.Connected(pool_name="train", node_id=0, instance=None))
    proj.handle(Node.Ready(pool_name="train", node_id=0))
    proj.handle(Metric.Sampled(pool_name="train", node_id=0, name="gpu_util_0", value=80.0))
    proj.handle(Metric.Sampled(pool_name="train", node_id=0, name="gpu_util_1", value=90.0))

    ui = view_to_ui(proj.view.pools["train"], t=1, base_elapsed=0)
    assert round(ui.nodes[0].gpu) == 85


def test_waiting_node_has_no_phase_bar() -> None:
    proj = _projection(1)
    pool = proj.view.pools["train"]
    # node was provisioned but never connected -> not present yet
    assert pool.nodes == {} or 0 not in pool.nodes
    proj.handle(Node.Connected(pool_name="train", node_id=0, instance=None))
    ui = view_to_ui(proj.view.pools["train"], t=1, base_elapsed=0)
    node = ui.nodes[0]
    # connected, no bootstrap yet -> SSH, empty phases (phase bar hidden)
    assert node.status is UiStatus.SSH
    assert node.phases == ()
    assert node.phase_idx == 0


# ── view_to_ui: header from cluster + instances ─────────────────


def test_header_from_cluster_and_instances() -> None:
    proj = _projection(1)
    inst = _instance(iid="i-head", spot=True, spot_price=1.92, on_demand_price=7.56)
    proj.handle(Pool.Provisioned(pool_name="train", cluster=_cluster(), instances=(inst,)))
    proj.handle(Node.Connected(pool_name="train", node_id=0, instance=inst))

    ui = view_to_ui(proj.view.pools["train"], t=0, base_elapsed=0)
    assert ui.provider == "vastai"
    assert ui.accel == "H100 80GB"
    assert ui.region == "us-east-1"
    assert ui.allocation == "spot"
    assert ui.hourly_cost == pytest.approx(1.92)  # spot price chosen
    assert ui.nodes[0].ip  # instance ip surfaced


# ── ProjectionSource: logs, ready tracking, pool selection ──────


def test_projection_source_accumulates_logs_and_selects_pool() -> None:
    proj = _projection(1)
    source = ProjectionSource(proj, pool_name="train")
    try:
        proj.handle(Node.Connected(pool_name="train", node_id=0, instance=None))
        proj.handle(Node.Ready(pool_name="train", node_id=0))
        proj.handle(Log.Emitted(
            pool_name="train", node_id=0, message="cluster bootstrapped", level="ok",
        ))
        proj.handle(Log.Emitted(
            pool_name="train", node_id=0, message="step 1240 loss 1.88", level="info",
        ))

        snap = source.snapshot()
        assert snap.name == "train"
        assert snap.ready_count == 1
        msgs = [log.msg for log in snap.cluster_log]
        assert "cluster bootstrapped" in msgs
        assert snap.cluster_log[0].node == "node-0"
        assert snap.cluster_log[0].level == "OK"
        # per-node log mirrors the cluster log without the node tag
        assert [log.msg for log in snap.nodes[0].logs] == msgs
        # node seen ready -> uptime anchored (started_at set)
        assert snap.nodes[0].started_at is not None
    finally:
        source.close()


def test_projection_source_empty_before_pool_exists() -> None:
    proj = SessionProjection()
    source = ProjectionSource(proj, pool_name="train")
    try:
        snap = source.snapshot()
        assert snap.name == "train"
        assert snap.nodes == ()
        assert snap.cluster_log == ()
    finally:
        source.close()


def test_projection_source_close_unsubscribes() -> None:
    proj = _projection(1)
    source = ProjectionSource(proj, pool_name="train")
    source.close()
    # after close, new logs must not accumulate
    proj.handle(Log.Emitted(pool_name="train", node_id=0, message="late", level="info"))
    assert source.snapshot().cluster_log == ()


# ── End-to-end: the real app driven by a ProjectionSource ───────


async def test_app_renders_a_projection_source() -> None:
    from skyward.tui.app import SkywardTUI
    from skyward.tui.screens import DashboardScreen, NodeScreen

    proj = _projection(2)
    proj.handle(Pool.Provisioned(
        pool_name="train", cluster=_cluster(),
        instances=(_instance(iid="i-0"), _instance(iid="i-1")),
    ))
    proj.handle(Node.Connected(pool_name="train", node_id=0, instance=_instance(iid="i-0")))
    proj.handle(Node.Ready(pool_name="train", node_id=0))
    proj.handle(Node.Connected(pool_name="train", node_id=1, instance=_instance(iid="i-1")))
    proj.handle(Node.Bootstrap.Started(pool_name="train", node_id=1, phase="apt"))
    proj.handle(Log.Emitted(pool_name="train", node_id=0, message="rank 0 online", level="ok"))

    source = ProjectionSource(proj, pool_name="train")
    app = SkywardTUI(source, tick_interval=3600.0)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)
            cluster = app.screen._cluster
            assert cluster is not None
            assert cluster.name == "train"
            assert cluster.ready_count == 1
            assert [n.id for n in cluster.nodes] == ["node-0", "node-1"]

            # open the bootstrapping node and confirm the node screen binds to it
            await pilot.press("down", "enter")
            await pilot.pause()
            assert isinstance(app.screen, NodeScreen)
            assert app.screen.index == 1
    finally:
        source.close()
