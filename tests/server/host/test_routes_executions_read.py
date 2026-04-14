"""Tests for the executions read routes: list (with filters) and detail."""

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
    GroupMember,
    Provisioning,
    Queued,
    Run,
    RunningExec,
    SucceededExec,
    Task,
    TaskExecution,
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
        for name in ("c1", "c2"):
            await store.put_compute(
                Compute(
                    name=name,
                    spec=_compute_spec(),
                    created_at=_t(0),
                    status=Provisioning(started_at=_t(0)),
                )
            )
        await store.put_task(Task(module="m1", qualname="f"))
        await store.put_task(Task(module="m2", qualname="g"))
        blob = await store.put_blob(path="/tmp/p.bin", size=1, kind="payload")
        await store.put_execution(
            TaskExecution(
                id="a", task=("m1", "f"), compute="c1", kind=Run(),
                payload=blob, timeout=None, client=None,
                submitted_at=_t(0), status=Queued(),
            )
        )
        await store.put_execution(
            TaskExecution(
                id="b", task=("m1", "f"), compute="c2", kind=Run(),
                payload=blob, timeout=None, client=None,
                submitted_at=_t(1), status=RunningExec(),
            )
        )
        await store.put_execution(
            TaskExecution(
                id="c", task=("m2", "g"), compute="c1",
                kind=GroupMember(group="g1"),
                payload=blob, timeout=None, client=None,
                submitted_at=_t(2),
                status=SucceededExec(finished_at=_t(5)),
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


async def test_list_executions_all(client):
    async with client as c:
        resp = await c.get("/v1/executions", headers=_HEADERS)
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()] == ["a", "b", "c"]


async def test_list_executions_filter_by_compute(client):
    async with client as c:
        resp = await c.get(
            "/v1/executions", params={"compute": "c1"}, headers=_HEADERS
        )
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()] == ["a", "c"]


async def test_list_executions_filter_by_status(client):
    async with client as c:
        resp = await c.get(
            "/v1/executions", params={"status": "running"}, headers=_HEADERS
        )
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()] == ["b"]


async def test_list_executions_filter_by_task(client):
    async with client as c:
        resp = await c.get(
            "/v1/executions", params={"task": "m1:f"}, headers=_HEADERS
        )
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()] == ["a", "b"]


async def test_list_executions_filter_by_group(client):
    async with client as c:
        resp = await c.get(
            "/v1/executions", params={"group": "g1"}, headers=_HEADERS
        )
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()] == ["c"]


async def test_list_executions_bad_task_key(client):
    async with client as c:
        resp = await c.get(
            "/v1/executions", params={"task": "no-colon"}, headers=_HEADERS
        )
    assert resp.status_code == 400
    assert resp.json() == {"error": "bad_task_key"}


async def test_get_execution_detail(seeded_store, client):
    expected = await seeded_store.get_execution("a")
    assert expected is not None
    async with client as c:
        resp = await c.get("/v1/executions/a", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == to_dict(expected)


async def test_get_execution_missing_returns_404(client):
    async with client as c:
        resp = await c.get("/v1/executions/nope", headers=_HEADERS)
    assert resp.status_code == 404
    assert resp.json() == {
        "error": "not_found",
        "resource": "execution",
        "name": "nope",
    }
