"""The one way the SDK reaches the control plane.

Every call is an HTTP call. Only the transport knows whether it left the
process: ``embedded`` runs the daemon here and reaches it through ASGI — no
socket, no port, no server — while ``remote`` dials one. Nothing above this
module can tell the difference, which is the point: there is a single client,
and running locally is a transport, not a second product.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, MutableMapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import httpx
import msgspec

from skyward2.protocol.schemas import Error
from skyward2.sdk.errors import raised

if TYPE_CHECKING:
    from litestar import Litestar

BLOB = "application/vnd.skyward.blob"
JSON = "application/json"

type Message = MutableMapping[str, Any]
type Asgi = Callable[
    [Message, Callable[[], Awaitable[Message]], Callable[[Message], Awaitable[None]]],
    Coroutine[None, None, None],
]


class Client:
    def __init__(self, http: httpx.AsyncClient, stack: AsyncExitStack) -> None:
        self._http = http
        self._stack = stack

    @classmethod
    async def embedded(cls, database: Path) -> Self:
        """The whole control plane, in this process, behind the same HTTP surface.

        The imports are here rather than at module scope because the daemon's
        dependencies are an extra: a program that only talks to a remote Skyward
        should not need a web framework to do it.
        """
        from skyward2.persistence.db import connect
        from skyward2.server.app import create_app, services

        await connect(database)
        app = create_app(services())

        stack = AsyncExitStack()
        await stack.enter_async_context(app.lifespan())
        http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=asgi(app)),
            base_url="http://skyward",
            timeout=None,
        )
        return cls(await stack.enter_async_context(http), stack)

    @classmethod
    async def remote(cls, url: str) -> Self:
        stack = AsyncExitStack()
        http = httpx.AsyncClient(base_url=url, timeout=None)
        return cls(await stack.enter_async_context(http), stack)

    async def close(self) -> None:
        await self._stack.aclose()

    async def call[T](
        self,
        method: str,
        path: str,
        kind: type[T],
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        **query: object,
    ) -> T:
        response = await self._send(method, path, body, JSON, headers, query)
        return msgspec.json.decode(response.content, type=kind)

    async def blob(self, path: str, **query: object) -> bytes | None:
        """The bytes, or None if the server had nothing to give yet (204)."""
        response = await self._send("GET", path, None, JSON, None, query)
        return response.content if response.status_code == 200 else None

    async def upload(self, path: str, body: bytes, headers: dict[str, str] | None = None) -> None:
        await self._send("PUT", path, body, BLOB, headers, {})

    async def _send(
        self,
        method: str,
        path: str,
        body: bytes | None,
        content_type: str,
        headers: dict[str, str] | None,
        query: dict[str, object],
    ) -> httpx.Response:
        response = await self._http.request(
            method,
            path,
            content=body,
            params={key: str(value) for key, value in query.items() if value is not None},
            headers={"Content-Type": content_type, **(headers or {})},
        )
        if response.status_code >= 400:
            raise raised(msgspec.json.decode(response.content, type=Error))
        return response


def asgi(app: Litestar) -> Asgi:
    """The same object, spelled the way httpx spells one.

    Litestar types the ASGI scope and messages as TypedDicts; httpx types them as
    plain mappings. They are the same dicts — the ASGI spec's — and a Litestar
    app is an ASGI app. The two vocabularies meet here and nowhere else.
    """
    return cast(Asgi, app)
