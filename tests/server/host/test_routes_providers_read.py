"""Tests for the providers read routes: ``GET /v1/providers`` and detail."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from skyward.server.host import Store
from skyward.server.host.app import create_app
from skyward.server.host.domain import Provider
from skyward.server.wire import to_dict

_HEADERS = {"X-Skyward-Api": "1"}


def _t(sec: int = 0) -> datetime:
    return datetime(2026, 4, 14, 12, sec // 60, sec % 60, tzinfo=UTC)


@pytest.fixture
async def seeded_store(tmp_path: Path):
    store = Store(path=str(tmp_path / "t.db"))
    await store.open()
    try:
        await store.put_provider(
            Provider(
                name="aws-main",
                type="aws",
                config={"region": "us-east-1"},
                created_at=_t(0),
                updated_at=_t(1),
                last_used_at=None,
            )
        )
        await store.put_provider(
            Provider(
                name="container-local",
                type="container",
                config={},
                created_at=_t(0),
                updated_at=_t(1),
                last_used_at=_t(5),
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


async def test_list_providers(client):
    async with client as c:
        resp = await c.get("/v1/providers", headers=_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert [item["name"] for item in body] == ["aws-main", "container-local"]


async def test_get_provider_detail(seeded_store, client):
    expected = await seeded_store.get_provider("aws-main")
    assert expected is not None
    async with client as c:
        resp = await c.get("/v1/providers/aws-main", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == to_dict(expected)


async def test_get_provider_missing_returns_404(client):
    async with client as c:
        resp = await c.get("/v1/providers/nope", headers=_HEADERS)
    assert resp.status_code == 404
    assert resp.json() == {
        "error": "not_found",
        "resource": "provider",
        "name": "nope",
    }
