"""Tests for the tasks read route: executions scoped by task key."""

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
    Provisioning,
    Queued,
    Run,
    Task,
    TaskExecution,
)

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
                name="c1",
                spec=_compute_spec(),
                created_at=_t(0),
                status=Provisioning(started_at=_t(0)),
            )
        )
        await store.put_task(Task(module="pkg.mod", qualname="fn"))
        await store.put_task(Task(module="pkg.mod", qualname="other"))
        blob = await store.put_blob(path="/tmp/p.bin", size=1, kind="payload")
        await store.put_execution(
            TaskExecution(
                id="exec-1",
                task=("pkg.mod", "fn"),
                compute="c1",
                kind=Run(),
                payload=blob,
                timeout=None,
                client=None,
                submitted_at=_t(0),
                status=Queued(),
            )
        )
        await store.put_execution(
            TaskExecution(
                id="exec-2",
                task=("pkg.mod", "fn"),
                compute="c1",
                kind=Run(),
                payload=blob,
                timeout=None,
                client=None,
                submitted_at=_t(1),
                status=Queued(),
            )
        )
        await store.put_execution(
            TaskExecution(
                id="exec-3",
                task=("pkg.mod", "other"),
                compute="c1",
                kind=Run(),
                payload=blob,
                timeout=None,
                client=None,
                submitted_at=_t(2),
                status=Queued(),
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


async def test_list_executions_for_task(client):
    async with client as c:
        resp = await c.get(
            "/v1/tasks/pkg.mod/fn/executions", headers=_HEADERS
        )
    assert resp.status_code == 200
    body = resp.json()
    assert [item["id"] for item in body] == ["exec-1", "exec-2"]


async def test_list_executions_for_other_task(client):
    async with client as c:
        resp = await c.get(
            "/v1/tasks/pkg.mod/other/executions", headers=_HEADERS
        )
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()] == ["exec-3"]


async def test_list_executions_for_unknown_task_is_empty(client):
    async with client as c:
        resp = await c.get(
            "/v1/tasks/ghost.mod/fn/executions", headers=_HEADERS
        )
    assert resp.status_code == 200
    assert resp.json() == []
