"""H1: ``ServerClient`` speaks the host API over HTTP (in-process ASGI)."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.mark.asyncio
async def test_server_client_health_and_info(tmp_path: Path) -> None:
    from skyward.server.host.app import create_app
    from skyward.server.host.store import Store
    from skyward.server.driver import ServerClient

    store = Store(str(tmp_path / "db"))
    await store.open()
    try:
        app = create_app(store)
        transport = httpx.ASGITransport(app=app)
        client = ServerClient("/tmp/ignored")
        client._client = httpx.AsyncClient(
            transport=transport, base_url="http://test",
            headers={"X-Skyward-Api": "1"},
        )
        async with client:
            body = await client.health()
            assert body["status"] == "ok"
            info = await client.info()
            assert info["api_version"] == 1
    finally:
        await store.close()
