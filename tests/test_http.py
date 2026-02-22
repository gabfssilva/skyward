from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from skyward.infra.http import BearerAuth, HttpClient, HttpError, OAuth2Auth, Response

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def make_app(
    *,
    token: str = "valid-token",
    oauth_access_token: str = "fresh-token",
) -> web.Application:
    app = web.Application()

    async def json_echo(request: web.Request) -> web.Response:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {token}":
            return web.Response(status=401, text="unauthorized")
        body = await request.json() if request.can_read_body else {}
        return web.json_response({"echo": body, "params": dict(request.query)})

    async def text_endpoint(_: web.Request) -> web.Response:
        return web.Response(text="plain-text-response")

    async def empty_json(_: web.Request) -> web.Response:
        return web.Response(status=204, body=b"")

    async def error_endpoint(_: web.Request) -> web.Response:
        return web.json_response({"error": "not found"}, status=404)

    async def server_error(_: web.Request) -> web.Response:
        return web.Response(status=500, text="internal server error")

    call_count: dict[str, int] = {"n": 0}

    async def auth_refresh(request: web.Request) -> web.Response:
        call_count["n"] += 1
        auth = request.headers.get("Authorization", "")
        if call_count["n"] == 1:
            return web.Response(status=401, text="expired")
        if auth != f"Bearer {oauth_access_token}":
            return web.Response(status=401, text="bad token")
        return web.json_response({"ok": True})

    async def oauth_token(request: web.Request) -> web.Response:
        data = await request.post()
        if data.get("grant_type") != "client_credentials":
            return web.Response(status=400, text="bad grant_type")
        return web.json_response({"access_token": oauth_access_token})

    async def no_auth_endpoint(_: web.Request) -> web.Response:
        return web.json_response({"public": True})

    app.router.add_route("*", "/echo", json_echo)
    app.router.add_post("/text", text_endpoint)
    app.router.add_get("/empty", empty_json)
    app.router.add_get("/not-found", error_endpoint)
    app.router.add_get("/server-error", server_error)
    app.router.add_route("*", "/auth-refresh", auth_refresh)
    app.router.add_post("/oauth2/token", oauth_token)
    app.router.add_get("/public", no_auth_endpoint)
    return app


@pytest.fixture
async def server():
    app = make_app()
    srv = TestServer(app)
    await srv.start_server()
    yield srv
    await srv.close()


@pytest.fixture
def base_url(server: TestServer) -> str:
    return f"http://{server.host}:{server.port}"


# ─── BearerAuth ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bearer_auth_headers():
    auth = BearerAuth("my-token")
    h = await auth.headers()
    assert h["Authorization"] == "Bearer my-token"
    assert h["Accept"] == "application/json"


@pytest.mark.asyncio
async def test_bearer_on_401_is_noop():
    auth = BearerAuth("t")
    await auth.on_401()


# ─── HttpClient basic requests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_get_json(base_url: str):
    async with HttpClient(base_url, BearerAuth("valid-token")) as http:
        result = await http.request("GET", "/echo", params={"a": "1"})
    assert result["params"]["a"] == "1"


@pytest.mark.asyncio
async def test_post_json(base_url: str):
    async with HttpClient(base_url, BearerAuth("valid-token")) as http:
        result = await http.request("POST", "/echo", json={"key": "value"})
    assert result["echo"]["key"] == "value"


@pytest.mark.asyncio
async def test_text_format(base_url: str):
    async with HttpClient(base_url) as http:
        result = await http.request("POST", "/text", format="text")
    assert result == "plain-text-response"


@pytest.mark.asyncio
async def test_empty_body_returns_none(base_url: str):
    async with HttpClient(base_url) as http:
        result = await http.request("GET", "/empty")
    assert result is None


# ─── Errors ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_http_error_on_4xx(base_url: str):
    async with HttpClient(base_url) as http:
        with pytest.raises(HttpError) as exc_info:
            await http.request("GET", "/not-found")
    assert exc_info.value.status == 404


@pytest.mark.asyncio
async def test_http_error_on_5xx(base_url: str):
    async with HttpClient(base_url) as http:
        with pytest.raises(HttpError) as exc_info:
            await http.request("GET", "/server-error")
    assert exc_info.value.status == 500


def test_http_error_str():
    err = HttpError(status=429, body="rate limited")
    assert str(err) == "HTTP 429: rate limited"


# ─── No auth ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_auth(base_url: str):
    async with HttpClient(base_url) as http:
        result = await http.request("GET", "/public")
    assert result["public"] is True


# ─── Default headers ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_headers(base_url: str):
    async with HttpClient(
        base_url,
        BearerAuth("valid-token"),
        default_headers={"X-Custom": "yes"},
    ) as http:
        result = await http.request("GET", "/echo")
    assert result is not None


# ─── 401 retry with OAuth2 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_oauth2_401_retry(base_url: str):
    auth = OAuth2Auth("cid", "csecret", f"{base_url}/oauth2/token")
    async with HttpClient(base_url, auth) as http:
        result = await http.request("GET", "/auth-refresh")
    assert result["ok"] is True


# ─── OAuth2Auth unit tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_oauth2_fetches_token(base_url: str):
    auth = OAuth2Auth("cid", "csecret", f"{base_url}/oauth2/token")
    h = await auth.headers()
    assert h["Authorization"] == "Bearer fresh-token"


@pytest.mark.asyncio
async def test_oauth2_caches_token(base_url: str):
    auth = OAuth2Auth("cid", "csecret", f"{base_url}/oauth2/token")
    h1 = await auth.headers()
    h2 = await auth.headers()
    assert h1 == h2


@pytest.mark.asyncio
async def test_oauth2_on_401_clears_token(base_url: str):
    auth = OAuth2Auth("cid", "csecret", f"{base_url}/oauth2/token")
    await auth.headers()
    assert auth._token is not None
    await auth.on_401()
    assert auth._token is None


# ─── Context manager & session lifecycle ─────────────────────────────


@pytest.mark.asyncio
async def test_close_idempotent(base_url: str):
    http = HttpClient(base_url)
    await http.close()
    await http.close()


@pytest.mark.asyncio
async def test_session_created_lazily(base_url: str):
    http = HttpClient(base_url)
    assert http._session is None
    await http.request("GET", "/public")
    assert http._session is not None
    await http.close()


@pytest.mark.asyncio
async def test_base_url_trailing_slash_stripped():
    http = HttpClient("http://example.com/")
    assert http._base_url == "http://example.com"
    await http.close()


# ─── Typed convenience methods ───────────────────────────────────────


@pytest.mark.asyncio
async def test_get_typed(base_url: str):
    async with HttpClient(base_url, BearerAuth("valid-token")) as http:
        resp = await http.get("/echo", params={"x": "1"}, response_type=dict)
    assert isinstance(resp, Response)
    assert resp.status == 200
    assert resp.data["params"]["x"] == "1"


@pytest.mark.asyncio
async def test_post_typed(base_url: str):
    async with HttpClient(base_url, BearerAuth("valid-token")) as http:
        resp = await http.post("/echo", json={"k": "v"}, response_type=dict)
    assert resp.status == 200
    assert resp.data["echo"]["k"] == "v"


@pytest.mark.asyncio
async def test_put_typed(base_url: str):
    async with HttpClient(base_url, BearerAuth("valid-token")) as http:
        resp = await http.put("/echo", json={"a": 1}, response_type=dict)
    assert resp.status == 200
    assert resp.data["echo"]["a"] == 1


@pytest.mark.asyncio
async def test_patch_typed(base_url: str):
    async with HttpClient(base_url, BearerAuth("valid-token")) as http:
        resp = await http.patch("/echo", json={"b": 2}, response_type=dict)
    assert resp.status == 200
    assert resp.data["echo"]["b"] == 2


@pytest.mark.asyncio
async def test_delete_typed(base_url: str):
    async with HttpClient(base_url, BearerAuth("valid-token")) as http:
        resp = await http.delete("/echo", response_type=dict)
    assert resp.status == 200


@pytest.mark.asyncio
async def test_typed_error_still_raises(base_url: str):
    async with HttpClient(base_url) as http:
        with pytest.raises(HttpError) as exc_info:
            await http.get("/not-found", response_type=dict)
    assert exc_info.value.status == 404


@pytest.mark.asyncio
async def test_response_is_frozen():
    resp: Response[int] = Response(status=200, data=42, headers={"Content-Type": "text/plain"})
    assert resp.status == 200
    assert resp.data == 42
    assert resp.headers["Content-Type"] == "text/plain"
    with pytest.raises(AttributeError):
        resp.status = 201  # type: ignore[misc]


@pytest.mark.asyncio
async def test_get_typed_has_headers(base_url: str):
    async with HttpClient(base_url, BearerAuth("valid-token")) as http:
        resp = await http.get("/echo", response_type=dict)
    assert "Content-Type" in resp.headers
