"""Tests for the HTTP host skeleton: /v1/health, /v1/info, API-version middleware, SSE helper."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

import skyward
from skyward.server.host import Store
from skyward.server.host.app import create_app
from skyward.server.host.sse import sse_response


@pytest.fixture
async def store(tmp_path):
    s = Store(path=str(tmp_path / "state.db"))
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


async def test_health_ok(client):
    async with client as c:
        resp = await c.get("/v1/health", headers={"X-Skyward-Api": "1"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_info_reports_python_and_version(client):
    async with client as c:
        resp = await c.get("/v1/info", headers={"X-Skyward-Api": "1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["python_version"] == sys.version
    assert body["api_version"] == 1
    assert body["skyward_version"] == skyward.__version__
    assert isinstance(body["pid"], int)
    assert body["pid"] > 0


async def test_api_version_mismatch_returns_409(client):
    async with client as c:
        resp = await c.get("/v1/health", headers={"X-Skyward-Api": "2"})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "api_version_mismatch"
    assert body["required"] == 1
    assert body["got"] == 2


async def test_missing_api_version_header_returns_409(client):
    async with client as c:
        resp = await c.get("/v1/health")
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "api_version_mismatch"
    assert body["required"] == 1
    assert body["got"] is None


async def test_sse_response_yields_events():
    async def gen() -> AsyncIterator[Any]:
        yield {"type": "phase", "name": "ready"}
        yield {"type": "phase", "name": "done"}

    resp = sse_response(gen())
    chunks: list[bytes] = []
    async for chunk in resp.body_iterator:
        chunks.append(bytes(chunk) if not isinstance(chunk, (bytes, str)) else chunk.encode() if isinstance(chunk, str) else chunk)
    body = b"".join(chunks).decode()
    assert f"data: {json.dumps({'type': 'phase', 'name': 'ready'})}\n\n" in body
    assert f"data: {json.dumps({'type': 'phase', 'name': 'done'})}\n\n" in body
    assert resp.media_type == "text/event-stream"


async def test_sse_response_accepts_preformatted_strings():
    async def gen() -> AsyncIterator[Any]:
        yield "event: phase\ndata: hello\n\n"

    resp = sse_response(gen())
    chunks: list[bytes] = []
    async for chunk in resp.body_iterator:
        chunks.append(bytes(chunk) if not isinstance(chunk, (bytes, str)) else chunk.encode() if isinstance(chunk, str) else chunk)
    body = b"".join(chunks).decode()
    assert "event: phase\ndata: hello\n\n" in body
