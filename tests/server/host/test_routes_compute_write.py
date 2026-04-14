"""Tests for the compute write routes: ``POST /v1/compute`` and ``DELETE``.

The write handlers are a Phase D stub that violates the single-writer
invariant on purpose; Phase G2 replaces them with ``PoolHost`` mediation and
these tests will be rewritten then.
"""

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
    Provisioning,
    Ready,
    Stopped,
    Stopping,
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
async def store(tmp_path: Path):
    s = Store(path=str(tmp_path / "t.db"))
    await s.open()
    try:
        yield s
    finally:
        await s.close()


@pytest.fixture
def client(store):
    app = create_app(store)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_post_compute_creates_provisioning(store, client):
    body = {"name": "foo", "spec": to_dict(_compute_spec())}
    async with client as c:
        resp = await c.post("/v1/compute", json=body, headers=_HEADERS)
    assert resp.status_code == 202
    payload = resp.json()
    assert payload["name"] == "foo"
    assert payload["status"]["type"] == "provisioning"
    assert "started_at" in payload["status"]
    persisted = await store.get_compute("foo")
    assert persisted is not None
    assert isinstance(persisted.status, Provisioning)


async def test_post_compute_existing_name_conflict(store, client):
    await store.put_compute(_make_compute("foo", Provisioning(started_at=_t(0))))
    body = {"name": "foo", "spec": to_dict(_compute_spec())}
    async with client as c:
        resp = await c.post("/v1/compute", json=body, headers=_HEADERS)
    assert resp.status_code == 409
    assert resp.json() == {"error": "exists", "name": "foo"}


async def test_post_compute_bad_body_returns_422(client):
    async with client as c:
        resp_missing_name = await c.post(
            "/v1/compute",
            json={"spec": to_dict(_compute_spec())},
            headers=_HEADERS,
        )
        resp_missing_spec = await c.post(
            "/v1/compute",
            json={"name": "bar"},
            headers=_HEADERS,
        )
    assert resp_missing_name.status_code == 422
    assert resp_missing_spec.status_code == 422


async def test_post_compute_invalid_json_returns_422(client):
    async with client as c:
        resp = await c.post(
            "/v1/compute",
            content=b"not-json",
            headers={**_HEADERS, "Content-Type": "application/json"},
        )
    assert resp.status_code == 422


async def test_delete_compute_marks_stopping(store, client):
    await store.put_compute(
        _make_compute(
            "foo",
            Ready(
                started_at=_t(0),
                chosen=_simple_spec(),
                nodes_ready=2,
                last_activity_at=_t(5),
            ),
        )
    )
    async with client as c:
        resp = await c.delete("/v1/compute/foo", headers=_HEADERS)
    assert resp.status_code == 202
    payload = resp.json()
    assert payload["status"]["type"] == "stopping"
    persisted = await store.get_compute("foo")
    assert persisted is not None
    assert isinstance(persisted.status, Stopping)


async def test_delete_compute_missing_returns_404(client):
    async with client as c:
        resp = await c.delete("/v1/compute/nope", headers=_HEADERS)
    assert resp.status_code == 404
    assert resp.json() == {
        "error": "not_found",
        "resource": "compute",
        "name": "nope",
    }


async def test_delete_compute_already_terminal_conflicts(store, client):
    await store.put_compute(
        _make_compute("foo", Stopped(started_at=_t(0), stopped_at=_t(5)))
    )
    await store.put_compute(
        _make_compute("bar", Failed(failed_at=_t(5), reason="boom"))
    )
    async with client as c:
        resp_stopped = await c.delete("/v1/compute/foo", headers=_HEADERS)
        resp_failed = await c.delete("/v1/compute/bar", headers=_HEADERS)
    assert resp_stopped.status_code == 409
    assert resp_stopped.json() == {"error": "already_terminal"}
    assert resp_failed.status_code == 409
    assert resp_failed.json() == {"error": "already_terminal"}
