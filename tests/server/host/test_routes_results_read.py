"""Tests for the results read route: ``GET /v1/executions/{id}/results``."""

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
    PendingRes,
    Provisioning,
    Queued,
    Run,
    SucceededRes,
    Task,
    TaskExecution,
    TaskResult,
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
                name="c1", spec=_compute_spec(), created_at=_t(0),
                status=Provisioning(started_at=_t(0)),
            )
        )
        await store.put_task(Task(module="m", qualname="f"))
        payload = await store.put_blob(path="/tmp/p.bin", size=1, kind="payload")
        result_blob = await store.put_blob(path="/tmp/r.bin", size=2, kind="result")
        for eid in ("e1", "e2"):
            await store.put_execution(
                TaskExecution(
                    id=eid, task=("m", "f"), compute="c1", kind=Run(),
                    payload=payload, timeout=None, client=None,
                    submitted_at=_t(0), status=Queued(),
                )
            )
        await store.put_result(
            TaskResult(id=0, execution="e1", shard=0, status=PendingRes())
        )
        await store.put_result(
            TaskResult(
                id=0, execution="e1", shard=1,
                status=SucceededRes(
                    dispatched_at=_t(1), started_at=_t(2),
                    finished_at=_t(3), node="n1", blob=result_blob,
                ),
            )
        )
        await store.put_result(
            TaskResult(id=0, execution="e2", shard=0, status=PendingRes())
        )
        yield store
    finally:
        await store.close()


@pytest.fixture
def client(seeded_store):
    app = create_app(seeded_store)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_list_results_for_execution(client):
    async with client as c:
        resp = await c.get("/v1/executions/e1/results", headers=_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert [item["shard"] for item in body] == [0, 1]


async def test_list_results_scoped_per_execution(client):
    async with client as c:
        resp = await c.get("/v1/executions/e2/results", headers=_HEADERS)
    assert resp.status_code == 200
    assert [item["shard"] for item in resp.json()] == [0]


async def test_list_results_unknown_execution_is_empty(client):
    async with client as c:
        resp = await c.get("/v1/executions/nope/results", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == []
