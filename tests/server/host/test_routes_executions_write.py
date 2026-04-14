"""Tests for the executions write route: ``POST /v1/compute/{name}/tasks``.

The write handler is a Phase D stub that violates the single-writer
invariant on purpose; Phase G3 replaces direct ``put_execution`` with
task-manager mediation and these tests will be rewritten then.
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
from skyward.server.host.blobs import Blobs
from skyward.server.host.domain import (
    Broadcast,
    Compute,
    ComputeSpec,
    Provisioning,
    Queued,
    Run,
)

_HEADERS = {
    "X-Skyward-Api": "1",
    "Content-Type": "application/octet-stream",
    "X-Task-Module": "mymod",
    "X-Task-Qualname": "myfn",
    "X-Timeout": "30.5",
    "X-Client-Id": "client-xyz",
    "X-Kind": "run",
}


def _t(sec: int = 0) -> datetime:
    return datetime(2026, 4, 14, 12, sec // 60, sec % 60, tzinfo=UTC)


def _compute_spec() -> ComputeSpec:
    return ComputeSpec(
        specs=(Spec(provider=Container(), image=Image(metrics=None)),),
        selection="cheapest",
        nodes=Nodes(desired=1),
        allocation="spot-if-available",
        ttl=timedelta(hours=1),
    )


@pytest.fixture
async def store(tmp_path: Path):
    s = Store(path=str(tmp_path / "t.db"))
    await s.open()
    try:
        await s.put_compute(
            Compute(
                name="c1",
                spec=_compute_spec(),
                created_at=_t(0),
                status=Provisioning(started_at=_t(0)),
            )
        )
        yield s
    finally:
        await s.close()


@pytest.fixture
def blobs(store, tmp_path: Path) -> Blobs:
    return Blobs(store=store, root=tmp_path / "blobs")


@pytest.fixture
def client(store, blobs):
    app = create_app(store=store, blobs=blobs)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_post_execution_creates_queued(client, store, blobs):
    payload = b"\x00\x01\x02payload"
    async with client as c:
        resp = await c.post(
            "/v1/compute/c1/tasks", content=payload, headers=_HEADERS
        )
    assert resp.status_code == 202
    body = resp.json()
    assert "id" in body
    exec_id = body["id"]

    exec_row = await store.get_execution(exec_id)
    assert exec_row is not None
    assert isinstance(exec_row.status, Queued)
    assert exec_row.timeout == timedelta(seconds=30.5)
    assert exec_row.compute == "c1"
    assert exec_row.task == ("mymod", "myfn")
    assert exec_row.client == "client-xyz"
    assert isinstance(exec_row.kind, Run)
    assert await blobs.read(exec_row.payload) == payload


async def test_post_execution_kind_broadcast(client, store):
    headers = {**_HEADERS, "X-Kind": "broadcast"}
    async with client as c:
        resp = await c.post(
            "/v1/compute/c1/tasks", content=b"payload", headers=headers
        )
    assert resp.status_code == 202
    exec_row = await store.get_execution(resp.json()["id"])
    assert exec_row is not None
    assert isinstance(exec_row.kind, Broadcast)


async def test_post_execution_missing_module_returns_422(client):
    headers = {k: v for k, v in _HEADERS.items() if k != "X-Task-Module"}
    async with client as c:
        resp = await c.post(
            "/v1/compute/c1/tasks", content=b"payload", headers=headers
        )
    assert resp.status_code == 422
    assert resp.json() == {"error": "missing_task_headers"}


async def test_post_execution_unknown_compute_returns_404(client):
    async with client as c:
        resp = await c.post(
            "/v1/compute/nope/tasks", content=b"payload", headers=_HEADERS
        )
    assert resp.status_code == 404
    assert resp.json() == {
        "error": "not_found",
        "resource": "compute",
        "name": "nope",
    }


async def test_post_execution_invalid_timeout_returns_422(client):
    headers = {**_HEADERS, "X-Timeout": "not-a-float"}
    async with client as c:
        resp = await c.post(
            "/v1/compute/c1/tasks", content=b"payload", headers=headers
        )
    assert resp.status_code == 422
    assert resp.json() == {"error": "invalid_timeout"}


async def test_post_execution_missing_kind_defaults_to_run(client, store):
    headers = {k: v for k, v in _HEADERS.items() if k != "X-Kind"}
    async with client as c:
        resp = await c.post(
            "/v1/compute/c1/tasks", content=b"payload", headers=headers
        )
    assert resp.status_code == 202
    exec_row = await store.get_execution(resp.json()["id"])
    assert exec_row is not None
    assert isinstance(exec_row.kind, Run)
