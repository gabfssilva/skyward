"""Tests for the compute read routes: ``GET /v1/compute`` and detail."""

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
    ComputeStatus,
    Failed,
    Ready,
)
from skyward.server.wire import to_dict

_HEADERS = {"X-Skyward-Api": "1"}


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


@pytest.fixture
async def seeded_store(tmp_path: Path):
    store = Store(path=str(tmp_path / "t.db"))
    await store.open()
    try:
        await store.put_compute(
            _make_compute(
                "ready-a",
                Ready(
                    started_at=_t(0),
                    chosen=_simple_spec(),
                    nodes_ready=2,
                    last_activity_at=_t(5),
                ),
            )
        )
        await store.put_compute(
            _make_compute("failed-b", Failed(failed_at=_t(1), reason="boom"))
        )
        yield store
    finally:
        await store.close()


@pytest.fixture
def client(seeded_store):
    app = create_app(seeded_store)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_list_compute_returns_both(client):
    async with client as c:
        resp = await c.get("/v1/compute", headers=_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert [item["name"] for item in body] == ["failed-b", "ready-a"]


async def test_list_compute_filter_by_status(client):
    async with client as c:
        resp = await c.get("/v1/compute", params={"status": "ready"}, headers=_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert [item["name"] for item in body] == ["ready-a"]


async def test_get_compute_by_name(seeded_store, client):
    expected = await seeded_store.get_compute("ready-a")
    assert expected is not None
    async with client as c:
        resp = await c.get("/v1/compute/ready-a", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == to_dict(expected)


async def test_get_compute_missing_returns_404(client):
    async with client as c:
        resp = await c.get("/v1/compute/nope", headers=_HEADERS)
    assert resp.status_code == 404
    body = resp.json()
    assert body == {"error": "not_found", "resource": "compute", "name": "nope"}
