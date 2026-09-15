"""The browser console, served by the daemon beside the API it drives."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from skyward.server.http.app import create_app, services
from skyward.server.persistence.db import connect

pytestmark = pytest.mark.local


@pytest.fixture
def build(tmp_path: Path) -> Path:
    directory = tmp_path / "console"
    (directory / "assets").mkdir(parents=True)
    (directory / "index.html").write_text("<!doctype html><title>Skyward</title>")
    (directory / "assets" / "index-a1b2.js").write_text("export {}")
    return directory


@pytest.fixture
async def http(tmp_path: Path, build: Path) -> AsyncIterator[AsyncTestClient]:
    await connect(tmp_path / "skyward.sqlite")
    async with AsyncTestClient(app=create_app(services(), logging=False, console_at=build)) as client:
        yield client


def describe_a_daemon_with_a_console() -> None:
    async def it_answers_every_route_the_app_draws_with_the_page(http: AsyncTestClient) -> None:
        for route in ("/", "/market", "/computes/c_1/nodes/0"):
            response = await http.get(route)

            assert response.status_code == 200, route
            assert response.headers["content-type"].startswith("text/html")
            assert response.headers["cache-control"] == "no-cache"
            assert "<title>Skyward</title>" in response.text

    async def it_serves_the_built_assets(http: AsyncTestClient) -> None:
        response = await http.get("/assets/index-a1b2.js")

        assert response.status_code == 200 and response.text == "export {}"

    async def a_missing_asset_is_not_answered_with_the_page(http: AsyncTestClient) -> None:
        response = await http.get("/assets/index-gone.js")

        assert response.status_code == 404

    async def the_api_keeps_its_own_answers_under_v1(http: AsyncTestClient) -> None:
        live = await http.get("/v1/health/live")
        missing = await http.get("/v1/nothing-here")

        assert live.status_code == 200 and live.json()["live"] is True
        assert missing.status_code == 404 and missing.headers["content-type"].startswith("application/json")


def describe_a_daemon_without_one() -> None:
    async def its_root_is_not_found(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        async with AsyncTestClient(app=create_app(services(), logging=False)) as http:
            response = await http.get("/")

        assert response.status_code == 404
