"""Pool orchestration: start, partial readiness, scale-to-zero, recover,
drain, stop, file ops, and resize."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from skyward.api.facts import BootstrapCommand, BootstrapPhase, ConsoleOutput
from skyward.control.pool import Pool, apply_stream_event
from skyward.api.spec import Nodes
from skyward.core.errors import ProvisioningError

from .conftest import FakeNode, FakeProvider, make_cluster, make_instance, make_spec

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


async def _start(pool: Pool, spec, provider):  # noqa: ANN001, ANN202
    return await asyncio.wait_for(
        pool.start(spec, provider, (MagicMock(),)), timeout=5.0,
    )


async def test_start_provisions_and_returns_started(patch_nodes: dict[int, FakeNode]) -> None:
    spec = make_spec(Nodes(desired=2))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    started = await _start(pool, spec, provider)
    assert started.cluster_id == "c1"
    assert len(started.instances) == 2
    assert pool.current_nodes() == (2, 2)
    assert provider.provision_calls == [2]


async def test_partial_readiness_starts_at_min(patch_nodes: dict[int, FakeNode]) -> None:
    spec = make_spec(Nodes(desired=4, min=2))
    provider = FakeProvider(make_cluster(spec), per_call=2)
    pool = Pool(pool_name="p")
    started = await _start(pool, spec, provider)
    assert len(started.instances) >= 2
    assert pool.current_nodes().ready >= 2


async def test_no_offers_raises(patch_nodes: dict[int, FakeNode]) -> None:
    spec = make_spec(Nodes(desired=1))
    pool = Pool(pool_name="p")
    with pytest.raises(ProvisioningError, match="No offers"):
        await pool.start(spec, FakeProvider(make_cluster(spec)), ())


async def test_provision_exhaustion_raises_and_cleans_up(
    patch_nodes: dict[int, FakeNode],
) -> None:
    spec = make_spec(Nodes(desired=2), max_provision_attempts=2)
    cluster = make_cluster(spec)
    provider = FakeProvider(cluster, per_call=0)
    pool = Pool(pool_name="p")
    with pytest.raises(ProvisioningError):
        await _start(pool, spec, provider)
    assert provider.teardown_calls >= 1


async def test_scale_to_zero_lazy_start(patch_nodes: dict[int, FakeNode]) -> None:
    spec = make_spec(Nodes(desired=0, min=0, max=8))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    started = await _start(pool, spec, provider)
    assert started.instances == ()
    assert provider.provision_calls == []
    assert pool._tm is not None
    assert pool._reconciler is not None


async def test_wake_from_zero_via_scale_up(patch_nodes: dict[int, FakeNode]) -> None:
    spec = make_spec(Nodes(desired=0, min=0, max=8))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    await _start(pool, spec, provider)
    provisioned = await pool.scale_up(1)
    assert provisioned == 1
    assert provider.provision_calls == [1]
    await asyncio.sleep(0.05)
    assert pool.current_nodes().ready == 1


async def test_recover_adopts_instances_with_ranks(
    patch_nodes: dict[int, FakeNode],
) -> None:
    spec = make_spec(Nodes(desired=2))
    cluster = make_cluster(spec)
    provider = FakeProvider(cluster)
    pool = Pool(pool_name="p")
    instances = (make_instance("i-a"), make_instance("i-b"))
    started = await asyncio.wait_for(
        pool.recover(spec, provider, cluster, instances, node_ids=(1, 0)),
        timeout=5.0,
    )
    assert started.cluster_id == "c1"
    assert patch_nodes[1].adopted
    assert patch_nodes[0].adopted
    assert not patch_nodes[0].provisioned
    assert provider.provision_calls == []


async def test_recover_with_no_instances_raises(patch_nodes: dict[int, FakeNode]) -> None:
    spec = make_spec(Nodes(desired=2))
    cluster = make_cluster(spec)
    pool = Pool(pool_name="p")
    with pytest.raises(ProvisioningError, match="No alive instances"):
        await pool.recover(spec, FakeProvider(cluster), cluster, ())


async def test_execute_routes_through_task_manager(
    patch_nodes: dict[int, FakeNode],
) -> None:
    spec = make_spec(Nodes(desired=1))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    await _start(pool, spec, provider)
    assert await pool.execute(b"fn", (), {}, task_id="t1") == "ok"


async def test_node_exhausted_terminates_and_notifies_reconciler(
    patch_nodes: dict[int, FakeNode],
) -> None:
    spec = make_spec(Nodes(desired=2))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    await _start(pool, spec, provider)
    iid = pool.instance_map[1]
    dead = pool.nodes[1]
    pool.node_exhausted(1, "preempted", iid)
    await asyncio.sleep(0.1)
    assert (iid,) in provider.terminated
    # reconciler auto-repair provisions a replacement reusing rank 1
    assert 1 in provider.provision_calls
    assert pool.nodes[1] is not dead


async def test_drain_nodes_never_drains_head_unless_scale_to_zero(
    patch_nodes: dict[int, FakeNode],
) -> None:
    spec = make_spec(Nodes(desired=2))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    await _start(pool, spec, provider)
    drained = pool.drain_nodes(frozenset({0, 1}))
    assert drained == 1
    assert 0 in pool.nodes
    assert 1 not in pool.nodes


async def test_drain_to_zero_allowed_when_min_zero(
    patch_nodes: dict[int, FakeNode],
) -> None:
    spec = make_spec(Nodes(desired=1, min=0, max=4))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    await _start(pool, spec, provider)
    drained = pool.drain_nodes(frozenset({0}))
    assert drained == 1
    assert pool.nodes == {}


async def test_stop_terminates_all_instances(patch_nodes: dict[int, FakeNode]) -> None:
    spec = make_spec(Nodes(desired=2))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    await _start(pool, spec, provider)
    iids = set(pool.instance_map.values())
    await asyncio.wait_for(pool.stop(), timeout=5.0)
    assert provider.terminated
    assert set(provider.terminated[-1]) == iids
    assert provider.teardown_calls == 1
    assert all(n.stopped for n in patch_nodes.values())


async def test_stop_is_idempotent(patch_nodes: dict[int, FakeNode]) -> None:
    spec = make_spec(Nodes(desired=1))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    await _start(pool, spec, provider)
    await pool.stop()
    await pool.stop()
    assert provider.teardown_calls == 1


async def test_file_op_fans_out_to_selection(patch_nodes: dict[int, FakeNode]) -> None:
    spec = make_spec(Nodes(desired=2))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    await _start(pool, spec, provider)
    results = await pool.file_op("ls", "/tmp", selection="all")
    assert [r.node_id for r in results] == [0, 1]
    head_only = await pool.file_op("ls", "/tmp", selection="head")
    assert [r.node_id for r in head_only] == [0]


async def test_resize_updates_bounds_and_desired(
    patch_nodes: dict[int, FakeNode],
) -> None:
    spec = make_spec(Nodes(desired=1, min=1, max=4))
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    await _start(pool, spec, provider)
    pool.resize(Nodes(desired=3, min=1, max=4))
    await asyncio.sleep(0.1)
    assert pool.scaling.desired_nodes == 3
    assert 2 in provider.provision_calls
    await asyncio.sleep(0.1)
    assert pool.current_nodes().total == 3


async def test_head_address_fans_out_to_other_nodes(
    patch_nodes: dict[int, FakeNode],
) -> None:
    spec = make_spec(Nodes(desired=2), cluster=True)
    provider = FakeProvider(make_cluster(spec))
    pool = Pool(pool_name="p")
    await _start(pool, spec, provider)
    from skyward.control.types import HeadAddressKnown

    info = HeadAddressKnown(
        head_addr="10.0.0.0", casty_port=25520, num_nodes=2,
        worker_concurrency=1, worker_executor="thread",
    )
    pool.head_address_known(info)
    assert patch_nodes[1].head_info is info
    assert patch_nodes[0].head_info is None


class TestBootstrapTimeline:
    def _ni(self):  # noqa: ANN202
        from skyward.api.facts import NodeInstance

        return NodeInstance(
            instance=make_instance("i-1"), node=0, provider="container",
            ssh_user="root", ssh_key_path="/key",
        )

    def test_phase_started_creates_timeline(self) -> None:
        timelines: dict = {}
        apply_stream_event(
            timelines, "i-1",
            BootstrapPhase(instance=self._ni(), event="started", phase="apt"),
        )
        assert timelines["i-1"].active == "apt"
        assert timelines["i-1"].phases == ("apt",)

    def test_phase_completed_marks_done(self) -> None:
        timelines: dict = {}
        ni = self._ni()
        apply_stream_event(
            timelines, "i-1", BootstrapPhase(instance=ni, event="started", phase="apt"),
        )
        apply_stream_event(
            timelines, "i-1", BootstrapPhase(instance=ni, event="completed", phase="apt"),
        )
        assert "apt" in timelines["i-1"].completed

    def test_command_and_output_update_tail(self) -> None:
        timelines: dict = {}
        ni = self._ni()
        apply_stream_event(
            timelines, "i-1", BootstrapPhase(instance=ni, event="started", phase="apt"),
        )
        apply_stream_event(
            timelines, "i-1", BootstrapCommand(instance=ni, command="apt install x"),
        )
        assert timelines["i-1"].output == "apt install x"
        apply_stream_event(
            timelines, "i-1", ConsoleOutput(instance=ni, content="  50% done  "),
        )
        assert timelines["i-1"].output == "50% done"

    def test_pool_snapshot_carries_timeline(self) -> None:
        pool = Pool(pool_name="p")
        pool.spec = make_spec(Nodes(desired=1))
        ni = self._ni()
        pool.nodes[0] = MagicMock()
        pool.instances[0] = ni
        pool.on_stream(BootstrapPhase(instance=ni, event="started", phase="uv"))
        snap = pool.snapshot()
        assert snap.nodes[0].bootstrap is not None
        assert snap.nodes[0].bootstrap.active == "uv"
