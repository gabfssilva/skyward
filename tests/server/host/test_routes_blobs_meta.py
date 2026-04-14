"""Tests for blob metadata read route: ``GET /v1/blobs/{id}/meta``.

Binary streaming (``GET /v1/blobs/{id}``) is owned by Phase E (Task E2);
C2 only exposes the JSON metadata shape.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from skyward.server.host import Store
from skyward.server.host.app import create_app

_HEADERS = {"X-Skyward-Api": "1"}


@pytest.fixture
async def seeded_store(tmp_path: Path):
    store = Store(path=str(tmp_path / "t.db"))
    await store.open()
    try:
        blob_id = await store.put_blob(
            path="/var/blobs/payload-1.bin",
            size=1234,
            sha256="deadbeef",
            kind="payload",
        )
        store._seeded_blob_id = blob_id  # type: ignore[attr-defined]
        yield store
    finally:
        await store.close()


@pytest.fixture
def client(seeded_store):
    app = create_app(seeded_store)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_get_blob_meta(seeded_store, client):
    blob_id = seeded_store._seeded_blob_id  # type: ignore[attr-defined]
    async with client as c:
        resp = await c.get(f"/v1/blobs/{blob_id}/meta", headers=_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == blob_id
    assert body["size"] == 1234
    assert body["sha256"] == "deadbeef"
    assert body["kind"] == "payload"
    assert body["evicted_at"] is None
    assert "created_at" in body
    assert "path" in body


async def test_get_blob_meta_missing_returns_404(client):
    async with client as c:
        resp = await c.get("/v1/blobs/9999/meta", headers=_HEADERS)
    assert resp.status_code == 404
    assert resp.json() == {
        "error": "not_found",
        "resource": "blob",
        "name": "9999",
    }


async def test_get_blob_meta_non_integer_id_returns_404(client):
    async with client as c:
        resp = await c.get("/v1/blobs/abc/meta", headers=_HEADERS)
    assert resp.status_code == 404
    assert resp.json() == {
        "error": "not_found",
        "resource": "blob",
        "name": "abc",
    }
