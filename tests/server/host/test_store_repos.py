"""Tests for provider / compute / node repository methods on ``Store``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from skyward.api.spec import Image, Nodes, Spec
from skyward.providers.container.config import Container
from skyward.server.host.domain import (
    Compute,
    ComputeSpec,
    ComputeStatus,
    Failed,
    Node,
    NodeBootstrapping,
    NodeConnecting,
    NodeLost,
    NodeReady,
    NodeStatus,
    NodeWaiting,
    Provider,
    Provisioning,
    Ready,
    Stopped,
    Stopping,
)
from skyward.server.host.store import Store


def _t(sec: int = 0) -> datetime:
    return datetime(2026, 4, 14, 12, sec // 60, sec % 60, tzinfo=UTC)


def _simple_spec() -> Spec:
    return Spec(provider=Container(), image=Image(metrics=None))


def _compute_spec() -> ComputeSpec:
    return ComputeSpec(
        specs=(_simple_spec(),),
        selection="cheapest",
        nodes=Nodes(desired=2),
        allocation="spot-if-available",
        ttl=timedelta(hours=1),
    )


def _make_compute(name: str, status: ComputeStatus) -> Compute:
    return Compute(
        name=name,
        spec=_compute_spec(),
        created_at=_t(0),
        status=status,
    )


async def _open(tmp_path: Path) -> Store:
    store = Store(str(tmp_path / "test.db"))
    await store.open()
    return store


@pytest.mark.asyncio
async def test_provider_roundtrip(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        p = Provider(
            name="aws-main",
            type="aws",
            config={"region": "us-east-1", "exclude_burstable": True},
            created_at=_t(0),
            updated_at=_t(1),
            last_used_at=None,
        )
        await store.put_provider(p)
        loaded = await store.get_provider("aws-main")
        assert loaded == p
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_providers_ordered(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        for name in ("zeta", "alpha", "mu"):
            await store.put_provider(
                Provider(
                    name=name,
                    type="container",
                    config={},
                    created_at=_t(0),
                    updated_at=_t(0),
                    last_used_at=None,
                )
            )
        loaded = await store.list_providers()
        assert [p.name for p in loaded] == ["alpha", "mu", "zeta"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_provider_missing_returns_none(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        assert await store.get_provider("ghost") is None
    finally:
        await store.close()


_COMPUTE_STATUSES: list[tuple[str, ComputeStatus]] = [
    ("provisioning", Provisioning(started_at=_t(0))),
    (
        "ready",
        Ready(
            started_at=_t(0),
            chosen=_simple_spec(),
            nodes_ready=4,
            last_activity_at=_t(30),
        ),
    ),
    ("stopping", Stopping(started_at=_t(0), stopping_since=_t(10))),
    ("stopped", Stopped(started_at=_t(0), stopped_at=_t(20))),
    ("failed", Failed(failed_at=_t(5), reason="boom")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("label,status", _COMPUTE_STATUSES, ids=[s[0] for s in _COMPUTE_STATUSES])
async def test_compute_roundtrip_each_status(
    tmp_path: Path, label: str, status: ComputeStatus
) -> None:
    store = await _open(tmp_path)
    try:
        c = _make_compute("c-" + label, status)
        await store.put_compute(c)
        loaded = await store.get_compute(c.name)
        assert loaded == c
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compute_chosen_spec_roundtrip(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        chosen = Spec(provider=Container(), accelerator=None, ttl=123, image=Image(metrics=None))
        c = _make_compute(
            "c-ready",
            Ready(
                started_at=_t(0),
                chosen=chosen,
                nodes_ready=2,
                last_activity_at=_t(5),
            ),
        )
        await store.put_compute(c)
        loaded = await store.get_compute(c.name)
        assert loaded is not None
        match loaded.status:
            case Ready(chosen=loaded_chosen):
                assert loaded_chosen == chosen
            case _:
                pytest.fail("expected Ready status")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_compute_filter_by_status(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await store.put_compute(_make_compute("a", Provisioning(started_at=_t(0))))
        await store.put_compute(
            _make_compute(
                "b",
                Ready(
                    started_at=_t(0),
                    chosen=_simple_spec(),
                    nodes_ready=1,
                    last_activity_at=_t(1),
                ),
            )
        )
        await store.put_compute(
            _make_compute(
                "c",
                Ready(
                    started_at=_t(0),
                    chosen=_simple_spec(),
                    nodes_ready=3,
                    last_activity_at=_t(2),
                ),
            )
        )
        await store.put_compute(_make_compute("d", Failed(failed_at=_t(1), reason="x")))

        all_items = await store.list_compute()
        assert [c.name for c in all_items] == ["a", "b", "c", "d"]

        ready_items = await store.list_compute(status="ready")
        assert [c.name for c in ready_items] == ["b", "c"]

        failed_items = await store.list_compute(status="failed")
        assert [c.name for c in failed_items] == ["d"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_compute_missing_returns_none(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        assert await store.get_compute("nope") is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_delete_compute(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        c = _make_compute("gone", Provisioning(started_at=_t(0)))
        await store.put_compute(c)
        assert await store.get_compute("gone") is not None
        await store.delete_compute("gone")
        assert await store.get_compute("gone") is None
    finally:
        await store.close()


_NODE_STATUSES: list[tuple[str, NodeStatus]] = [
    ("waiting", NodeWaiting()),
    ("connecting", NodeConnecting(since=_t(1))),
    ("bootstrapping", NodeBootstrapping(since=_t(2), phase="apt")),
    ("ready", NodeReady(since=_t(3))),
    ("lost", NodeLost(at=_t(9), reason="preempted")),
]


async def _seed_compute(store: Store, name: str) -> None:
    await store.put_compute(_make_compute(name, Provisioning(started_at=_t(0))))


@pytest.mark.asyncio
@pytest.mark.parametrize("label,status", _NODE_STATUSES, ids=[s[0] for s in _NODE_STATUSES])
async def test_node_roundtrip_each_status(
    tmp_path: Path, label: str, status: NodeStatus
) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        n = Node(
            id=f"n-{label}",
            compute="c1",
            instance_id="i-1",
            provider_name="aws-main",
            head_addr="10.0.0.1" if label == "ready" else None,
            status=status,
            created_at=_t(0),
        )
        await store.put_node(n)
        loaded = await store.list_nodes(compute="c1")
        assert loaded == [n]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_nodes_filters_by_compute(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "a")
        await _seed_compute(store, "b")
        for compute, ids in (("a", ("a1", "a2")), ("b", ("b1", "b2"))):
            for nid in ids:
                await store.put_node(
                    Node(
                        id=nid,
                        compute=compute,
                        instance_id=f"i-{nid}",
                        provider_name="aws-main",
                        head_addr=None,
                        status=NodeWaiting(),
                        created_at=_t(0),
                    )
                )
        a_nodes = await store.list_nodes(compute="a")
        b_nodes = await store.list_nodes(compute="b")
        assert [n.id for n in a_nodes] == ["a1", "a2"]
        assert [n.id for n in b_nodes] == ["b1", "b2"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_decode_equals_encode_invariant(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        c = Compute(
            name="invariant",
            spec=_compute_spec(),
            created_at=_t(0),
            status=Ready(
                started_at=_t(0),
                chosen=_simple_spec(),
                nodes_ready=7,
                last_activity_at=_t(42),
            ),
        )
        await store.put_compute(c)
        loaded = await store.get_compute("invariant")
        assert loaded == c
    finally:
        await store.close()
