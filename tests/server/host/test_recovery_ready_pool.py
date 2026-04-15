"""Workstream C — ``Ready`` compute recovery on PoolHost boot.

The simplified first-cut marks recovery-failed computes as ``Failed``
so queued executions surface a clear cause rather than hang. Full
provider-specific rehydration is deferred; this suite locks in the
safety invariants that the sweep guarantees today.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from skyward.accelerators import A100
from skyward.api.spec import DEFAULT_IMAGE, Nodes, Spec
from skyward.providers.container import Container
from skyward.server.host.domain import (
    Compute,
    ComputeSpec,
    Failed,
    Node,
    NodeReady,
    NodeWaiting,
    Queued,
    Ready,
    Run,
    Task,
    TaskExecution,
)
from skyward.server.host.store import Store

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _spec() -> Spec:
    return Spec(
        provider=Container(),
        accelerator=A100(),
        image=DEFAULT_IMAGE,
        nodes=Nodes(desired=1),
    )


def _ready_compute(name: str = "p") -> Compute:
    spec = _spec()
    return Compute(
        name=name,
        spec=ComputeSpec(
            specs=(spec,),
            selection="cheapest",
            nodes=Nodes(desired=1),
            allocation="spot-if-available",
            ttl=timedelta(hours=1),
        ),
        created_at=datetime.now(UTC),
        status=Ready(
            started_at=datetime.now(UTC),
            chosen=spec,
            nodes_ready=1,
            last_activity_at=datetime.now(UTC),
        ),
    )


def _node_ready(compute_name: str, nid: str = "n-0") -> Node:
    return Node(
        id=nid,
        compute=compute_name,
        instance_id="i-abc",
        provider_name="container",
        head_addr=None,
        status=NodeReady(since=datetime.now(UTC)),
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_ready_without_any_ready_node_marks_failed(tmp_path) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.pool_host import PoolHost

    store = Store(":memory:")
    await store.open()
    blobs = Blobs(store=store, root=tmp_path / "blobs")
    host = PoolHost(store=store, blobs=blobs, logs_dir=tmp_path / "logs")
    await store.put_compute(_ready_compute("orphan"))
    waiting = Node(
        id="n-0",
        compute="orphan",
        instance_id="i-x",
        provider_name="container",
        head_addr=None,
        status=NodeWaiting(),
        created_at=datetime.now(UTC),
    )
    await store.put_node(waiting)

    host._session = None  # no-op session — exercises the Failed path.
    try:
        await host._recover_ready_computes()
    except AssertionError:
        pass  # _try_recover_one asserts session is not None; we want Failed.

    row = await store.get_compute("orphan")
    assert row is not None
    assert isinstance(row.status, Failed)
    assert "recovery_failed" in row.status.reason
    await store.close()


@pytest.mark.asyncio
async def test_ready_with_ready_node_marks_failed_until_full_impl(tmp_path) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.pool_host import PoolHost

    store = Store(":memory:")
    await store.open()
    blobs = Blobs(store=store, root=tmp_path / "blobs")
    host = PoolHost(store=store, blobs=blobs, logs_dir=tmp_path / "logs")
    await store.put_compute(_ready_compute("alive"))
    await store.put_node(_node_ready("alive"))

    host._session = None
    try:
        await host._recover_ready_computes()
    except AssertionError:
        pass

    row = await store.get_compute("alive")
    assert row is not None
    assert isinstance(row.status, Failed)
    assert "recovery_failed" in row.status.reason
    await store.close()


@pytest.mark.asyncio
async def test_fail_queued_without_live_pool_requires_live_pool(tmp_path) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.domain import InterruptedExec
    from skyward.server.host.pool_host import PoolHost

    store = Store(":memory:")
    await store.open()
    blobs = Blobs(store=store, root=tmp_path / "blobs")
    host = PoolHost(store=store, blobs=blobs, logs_dir=tmp_path / "logs")
    compute = _ready_compute("ghost")
    await store.put_compute(compute)
    await store.put_task(Task(module="m", qualname="f"))
    payload_bid = await blobs.put(b"x", kind="payload")
    await store.put_execution(
        TaskExecution(
            id="e1",
            task=("m", "f"),
            compute="ghost",
            kind=Run(),
            payload=payload_bid,
            timeout=timedelta(seconds=10),
            client=None,
            submitted_at=datetime.now(UTC),
            status=Queued(),
        ),
    )

    class _FakeSession:
        _pools: dict[str, object] = {}

    host._session = _FakeSession()  # type: ignore[assignment]
    await host._fail_queued_without_live_pool()

    row = await store.get_execution("e1")
    assert row is not None
    assert isinstance(row.status, InterruptedExec)
    assert row.status.reason == "compute_failed_on_restart"

    await store.put_compute(replace(compute, status=compute.status))

    class _LiveSession:
        _pools = {"ghost": object()}

    host._session = _LiveSession()  # type: ignore[assignment]
    await store.update_execution_status("e1", Queued())
    await host._fail_queued_without_live_pool()

    row = await store.get_execution("e1")
    assert row is not None
    assert isinstance(row.status, Queued)
    await store.close()
