"""The one way the SDK reaches the control plane.

Every call is an HTTP call. Only the transport knows whether it left the
process: ``embedded`` runs the daemon here and reaches it through ASGI — no
socket, no port, no server — while ``remote`` dials one. Nothing above this
module can tell the difference, which is the point: there is a single client,
and running locally is a transport, not a second product.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine, MutableMapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import httpx
import msgspec

from skyward2.protocol.schemas import Error
from skyward2.sdk.errors import raised

if TYPE_CHECKING:
    from litestar import Litestar

logger = logging.getLogger(__name__)

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
            transport=Embedded(asgi(app)),
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

    async def frames(self, path: str) -> AsyncGenerator[bytes]:
        """The frames of a stream, as they arrive.

        The response is pulled one chunk at a time and only as the caller asks for a
        frame, which is what carries the backpressure the whole way: a consumer that
        stops reading stops the generator on the machine. Buffering the response here
        to hand back a list would throw that away and call it convenience.
        """
        async with self._http.stream("GET", path) as response:
            if response.status_code >= 400:
                await response.aread()
                raise raised(msgspec.json.decode(response.content, type=Error))

            buffer = bytearray()
            async for chunk in response.aiter_bytes():
                buffer += chunk
                while len(buffer) >= 4 and len(buffer) >= 4 + int.from_bytes(buffer[:4], "big"):
                    size = int.from_bytes(buffer[:4], "big")
                    yield bytes(buffer[4 : 4 + size])
                    del buffer[: 4 + size]

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


class Embedded(httpx.AsyncBaseTransport):
    """The in-process control plane, reached as if it were across a network.

    httpx ships an ASGI transport already, and it runs the application to completion
    and hands back the body as a list. For a request that is not a stream that is
    merely eager; for one that is, it is the whole thing gone — the generator on the
    machine would be pulled as fast as it produced, into a buffer nobody had asked
    to fill.

    So the application runs as a task and its body chunks go through a queue of one.
    The queue is the backpressure: the app cannot send a chunk until the reader has
    taken the last, and the reader is the user's ``for`` loop, six frames and one SSH
    tunnel away. Closing the response cancels the task, which closes the generator on
    the node — the same thing a dropped connection does to a real daemon.
    """

    def __init__(self, app: Asgi) -> None:
        self._app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        chunks: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=1)
        start: asyncio.Future[Message] = asyncio.get_running_loop().create_future()

        scope: Message = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.1"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(key.lower(), value) for key, value in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": ("127.0.0.1", 123),
            "root_path": "",
        }

        consumed = False
        disconnected = asyncio.Event()

        async def receive() -> Message:
            """The request once, and then nothing until the caller goes away.

            Answering ``http.request`` a second time is what a server would never do,
            and what a streaming response takes as an invitation to ask again: it is
            listening for the disconnect that this is supposed to be waiting for.
            """
            nonlocal consumed
            if consumed:
                await disconnected.wait()
                return {"type": "http.disconnect"}

            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: Message) -> None:
            match message["type"]:
                case "http.response.start" if not start.done():
                    start.set_result(message)
                case "http.response.body":
                    await chunks.put(message.get("body", b""))
                    if not message.get("more_body", False):
                        await chunks.put(None)

        async def run() -> None:
            """The application, as a task, because the response is read while it runs.

            A failure after the headers have gone out cannot become a status code, and
            it must not become silence either: nothing is awaiting this task, so an
            exception here would be swallowed and the body would simply stop.
            """
            try:
                await self._app(scope, receive, send)
            except Exception as exc:
                if not start.done():
                    start.set_exception(exc)
                    return
                logger.exception("the control plane failed while streaming a response")
            finally:
                if start.done() and not disconnected.is_set():
                    await chunks.put(None)

        task = asyncio.get_running_loop().create_task(run())

        async def stream() -> AsyncGenerator[bytes]:
            """The body, and the hang-up when the reader stops reading.

            A reader that walks away is a disconnected client, and that is what the
            application is told — the same message a real server would deliver, which
            is what makes the response generator on the other side close its own
            resources instead of being killed holding them.
            """
            try:
                while (chunk := await chunks.get()) is not None:
                    if chunk:
                        yield chunk
            finally:
                disconnected.set()

        headers: Message = await start
        return httpx.Response(
            headers["status"],
            headers=headers.get("headers", []),
            stream=Body(stream(), task),
        )


class Body(httpx.AsyncByteStream):
    """The response body, and the request that is still producing it."""

    def __init__(self, chunks: AsyncGenerator[bytes], request: asyncio.Task[None]) -> None:
        self._chunks = chunks
        self._request = request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        """Hang up, then make sure the request is really over.

        The hang-up is the polite half and does the useful work — the application
        sees a disconnected client and unwinds. The cancel is the backstop: nothing
        is awaiting the request task, so one that decided to ignore the disconnect
        would go on producing into a queue nobody is draining.
        """
        await self._chunks.aclose()
        self._request.cancel()


def asgi(app: Litestar) -> Asgi:
    """The same object, spelled the way httpx spells one.

    Litestar types the ASGI scope and messages as TypedDicts; httpx types them as
    plain mappings. They are the same dicts — the ASGI spec's — and a Litestar
    app is an ASGI app. The two vocabularies meet here and nowhere else.
    """
    return cast(Asgi, app)
