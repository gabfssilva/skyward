from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal, Protocol, overload, runtime_checkable

import aiohttp

from skyward.observability.logger import logger

# ─── Errors ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HttpError(Exception):
    status: int
    body: str

    def __str__(self) -> str:
        return f"HTTP {self.status}: {self.body}"


# ─── Response ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Response[T]:
    status: int
    data: T
    headers: dict[str, str]


# ─── Auth ────────────────────────────────────────────────────────────


@runtime_checkable
class Auth(Protocol):
    async def headers(self) -> dict[str, str]: ...
    async def on_401(self) -> None: ...


class BearerAuth:
    def __init__(self, token: str) -> None:
        self._token = token

    async def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    async def on_401(self) -> None:
        pass


class OAuth2Auth:
    def __init__(
        self, client_id: str, client_secret: str, token_url: str
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._token: str | None = None
        self._lock = asyncio.Lock()

    async def _fetch_token(self) -> str:
        logger.bind(component="http").debug("Fetching OAuth2 token")
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session, session.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logger.bind(component="http").error(
                    "OAuth2 token fetch failed: status={status} body={body}",
                    status=resp.status, body=body[:200],
                )
            resp.raise_for_status()
            data = await resp.json()
            return data["access_token"]

    async def headers(self) -> dict[str, str]:
        async with self._lock:
            if self._token is None:
                self._token = await self._fetch_token()
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    async def on_401(self) -> None:
        async with self._lock:
            self._token = None


# ─── Client ──────────────────────────────────────────────────────────


class HttpClient:
    def __init__(
        self,
        base_url: str,
        auth: Auth | None = None,
        *,
        timeout: float = 30,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = auth
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._default_headers = default_headers or {}
        self._session: aiohttp.ClientSession | None = None
        self._log = logger.bind(component="http")

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def _build_headers(self) -> dict[str, str]:
        headers = dict(self._default_headers)
        if self._auth:
            headers.update(await self._auth.headers())
        return headers

    async def _send(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        format: Literal["json", "text"] = "json",
    ) -> tuple[int, Any, dict[str, str]]:
        session = await self._ensure_session()
        headers = await self._build_headers()
        self._log.debug("{method} {path}", method=method, path=path)

        try:
            async with session.request(
                method, self._url(path), headers=headers, json=json, params=params
            ) as resp:
                if resp.status == 401 and self._auth:
                    self._log.debug("401 received, refreshing auth and retrying")
                    await self._auth.on_401()
                    retry_headers = await self._build_headers()
                    async with session.request(
                        method,
                        self._url(path),
                        headers=retry_headers,
                        json=json,
                        params=params,
                    ) as retry_resp:
                        return await self._parse(retry_resp, format)

                return await self._parse(resp, format)
        except aiohttp.ClientResponseError as e:
            raise HttpError(status=e.status, body=e.message) from e
        except aiohttp.ClientError as e:
            raise HttpError(status=0, body=str(e)) from e

    async def _parse(
        self, resp: aiohttp.ClientResponse, format: Literal["json", "text"]
    ) -> tuple[int, Any, dict[str, str]]:
        if resp.status >= 400:
            body = await resp.text()
            self._log.warning(
                "HTTP {status} from {url}: {body}",
                status=resp.status, url=str(resp.url), body=body[:500],
            )
            raise HttpError(status=resp.status, body=body)
        resp_headers = dict(resp.headers)
        match format:
            case "json":
                body = await resp.read()
                return resp.status, (await resp.json() if body else None), resp_headers
            case "text":
                return resp.status, await resp.text(), resp_headers

    # ─── Untyped low-level ───────────────────────────────────────────

    @overload
    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        format: Literal["json"] = "json",
    ) -> Any: ...

    @overload
    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        format: Literal["text"],
    ) -> str: ...

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        format: Literal["json", "text"] = "json",
    ) -> Any:
        _, data, _ = await self._send(method, path, json=json, params=params, format=format)
        return data

    # ─── Typed convenience ───────────────────────────────────────────

    async def _typed[T](
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        response_type: type[T],
    ) -> Response[T]:
        _ = response_type
        status, data, headers = await self._send(method, path, json=json, params=params)
        return Response(status=status, data=data, headers=headers)

    async def get[T](
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        response_type: type[T],
    ) -> Response[T]:
        return await self._typed("GET", path, params=params, response_type=response_type)

    async def post[T](
        self,
        path: str,
        *,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        response_type: type[T],
    ) -> Response[T]:
        return await self._typed(
            "POST", path, json=json, params=params,
            response_type=response_type,
        )

    async def put[T](
        self,
        path: str,
        *,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        response_type: type[T],
    ) -> Response[T]:
        return await self._typed("PUT", path, json=json, params=params, response_type=response_type)

    async def patch[T](
        self,
        path: str,
        *,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        response_type: type[T],
    ) -> Response[T]:
        return await self._typed(
            "PATCH", path, json=json, params=params,
            response_type=response_type,
        )

    async def delete[T](
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        response_type: type[T],
    ) -> Response[T]:
        return await self._typed("DELETE", path, params=params, response_type=response_type)

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def close(self) -> None:
        if self._session and not self._session.closed:
            self._log.debug("Closing HTTP session")
            await self._session.close()

    async def __aenter__(self) -> HttpClient:
        await self._ensure_session()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
