"""Tests for blob binary streaming route: ``GET /v1/blobs/{id}``.

Live rows stream ``application/octet-stream``; evicted rows return ``410 Gone``
with ``{"error":"evicted","at":<iso>}``; unknown ids return ``404``.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from skyward.server.host import Store
from skyward.server.host.app import create_app
from skyward.server.host.blobs import Blobs

_HEADERS = {"X-Skyward-Api": "1"}


@pytest.fixture
async def store(tmp_path: Path):
    s = Store(path=str(tmp_path / "t.db"))
    await s.open()
    try:
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


async def test_get_blob_streams_bytes(blobs, client):
    blob_id = await blobs.put(b"hello", kind="payload")
    async with client as c:
        resp = await c.get(f"/v1/blobs/{blob_id}", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.content == b"hello"


async def test_get_blob_evicted_returns_410(blobs, client):
    blob_id = await blobs.put(b"hello", kind="payload")
    await blobs.evict(blob_id)
    async with client as c:
        resp = await c.get(f"/v1/blobs/{blob_id}", headers=_HEADERS)
    assert resp.status_code == 410
    body = resp.json()
    assert body["error"] == "evicted"
    assert isinstance(body["at"], str)
    assert body["at"]


async def test_get_blob_missing_returns_404(client):
    async with client as c:
        resp = await c.get("/v1/blobs/99999", headers=_HEADERS)
    assert resp.status_code == 404
    assert resp.json() == {"error": "not_found", "resource": "blob", "id": 99999}


async def test_get_blob_non_integer_id_returns_404(client):
    async with client as c:
        resp = await c.get("/v1/blobs/notanint", headers=_HEADERS)
    assert resp.status_code == 404
    assert resp.json() == {
        "error": "not_found",
        "resource": "blob",
        "id": "notanint",
    }
