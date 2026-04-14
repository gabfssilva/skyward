"""Tests for the providers write route: ``PUT /v1/providers/{name}``.

The upsert handler is a Phase D stub that writes directly to the store.
Later phases may mediate provider management through a host actor; these
tests pin the wire contract and persistence round-trip regardless.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from skyward.providers.aws.config import AWS
from skyward.server.host import Store
from skyward.server.host.app import create_app
from skyward.server.host.domain import Provider

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
def client(store):
    app = create_app(store)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_put_provider_creates_row(store, client):
    body = {"type": "aws", "config": {"type": "aws", "region": "us-east-1"}}
    async with client as c:
        resp = await c.put(
            "/v1/providers/aws-main", json=body, headers=_HEADERS
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["name"] == "aws-main"
    assert payload["type"] == "aws"
    persisted = await store.get_provider("aws-main")
    assert persisted is not None
    assert isinstance(persisted, Provider)
    assert persisted.name == "aws-main"
    assert persisted.type == "aws"
    assert persisted.config["region"] == "us-east-1"


async def test_put_provider_updates_existing(store, client):
    async with client as c:
        first = await c.put(
            "/v1/providers/aws-main",
            json={"type": "aws", "config": {"type": "aws", "region": "us-east-1"}},
            headers=_HEADERS,
        )
        assert first.status_code == 200
        second = await c.put(
            "/v1/providers/aws-main",
            json={"type": "aws", "config": {"type": "aws", "region": "us-west-2"}},
            headers=_HEADERS,
        )
    assert second.status_code == 200
    persisted = await store.get_provider("aws-main")
    assert persisted is not None
    assert persisted.config["region"] == "us-west-2"


async def test_put_provider_missing_type_returns_422(client):
    async with client as c:
        resp = await c.put(
            "/v1/providers/aws-main",
            json={"config": {"type": "aws", "region": "us-east-1"}},
            headers=_HEADERS,
        )
    assert resp.status_code == 422


async def test_put_provider_invalid_config_returns_422(client):
    async with client as c:
        resp = await c.put(
            "/v1/providers/aws-main",
            json={"type": "aws", "config": {"type": "nonexistent-provider"}},
            headers=_HEADERS,
        )
    assert resp.status_code == 422


async def test_put_provider_invalid_json_returns_422(client):
    async with client as c:
        resp = await c.put(
            "/v1/providers/aws-main",
            content=b"not-json",
            headers={**_HEADERS, "Content-Type": "application/json"},
        )
    assert resp.status_code == 422


# Ensure AWS config import is not tree-shaken when unused by tests above.
_ = AWS
