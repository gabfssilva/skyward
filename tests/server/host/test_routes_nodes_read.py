"""Tests for the nodes read route: ``GET /v1/compute/{compute_name}/nodes``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from skyward.api.spec import Image, Nodes, Spec
from skyward.providers.container.config import Container
from skyward.server.host import Store
from skyward.server.host.app import create_app
from skyward.server.host.domain import (
    Compute,
    ComputeSpec,
    Node,
    NodeReady,
    NodeWaiting,
    Provisioning,
)
from skyward.server.wire import to_dict

_HEADERS = {"X-Skyward-Api": "1"}


def _t(sec: int = 0) -> datetime:
    return datetime(2026, 4, 14, 12, sec // 60, sec % 60, tzinfo=UTC)


def _compute_spec() -> ComputeSpec:
    return ComputeSpec(
        specs=(Spec(provider=Container(), image=Image(metrics=None)),),
        selection="cheapest",
        nodes=Nodes(desired=2),
        allocation="spot-if-available",
        ttl=timedelta(hours=1),
    )


@pytest.fixture
async def seeded_store(tmp_path: Path):
    store = Store(path=str(tmp_path / "t.db"))
    await store.open()
    try:
        await store.put_compute(
            Compute(
                name="pool-a",
                spec=_compute_spec(),
                created_at=_t(0),
                status=Provisioning(started_at=_t(0)),
            )
        )
        await store.put_compute(
            Compute(
                name="pool-b",
                spec=_compute_spec(),
                created_at=_t(0),
                status=Provisioning(started_at=_t(0)),
            )
        )
        await store.put_node(
            Node(
                id="n-a1",
                compute="pool-a",
                instance_id="i-a1",
                provider_name="aws-main",
                head_addr="10.0.0.1",
                status=NodeReady(since=_t(5)),
                created_at=_t(0),
            )
        )
        await store.put_node(
            Node(
                id="n-a2",
                compute="pool-a",
                instance_id="i-a2",
                provider_name="aws-main",
                head_addr=None,
                status=NodeWaiting(),
                created_at=_t(1),
            )
        )
        await store.put_node(
            Node(
                id="n-b1",
                compute="pool-b",
                instance_id="i-b1",
                provider_name="aws-main",
                head_addr=None,
                status=NodeWaiting(),
                created_at=_t(0),
            )
        )
        yield store
    finally:
        await store.close()


@pytest.fixture
def client(seeded_store):
    app = create_app(seeded_store)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_list_nodes_for_compute(seeded_store, client):
    async with client as c:
        resp = await c.get("/v1/compute/pool-a/nodes", headers=_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert [item["id"] for item in body] == ["n-a1", "n-a2"]
    expected = await seeded_store.list_nodes(compute="pool-a")
    assert body == [to_dict(n) for n in expected]


async def test_list_nodes_other_compute_isolated(client):
    async with client as c:
        resp = await c.get("/v1/compute/pool-b/nodes", headers=_HEADERS)
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()] == ["n-b1"]


async def test_list_nodes_unknown_compute_is_empty(client):
    async with client as c:
        resp = await c.get("/v1/compute/nope/nodes", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == []
