"""The one way the SDK reaches the control plane.

Every call is an HTTP call. Only the transport knows whether it left the
process: ``embedded`` runs the daemon here and reaches it through ASGI — no
socket, no port, no server — while ``remote`` dials one. Nothing above this
module can tell the difference, which is the point: there is a single client,
and running locally is a transport, not a second product.

:func:`connect` is what a pool actually calls, and what it usually ends up with is
``remote``: a daemon at the default address, started here if there was none. The
embedded transport is for the caller who named a database, which is the one thing
a daemon on that address cannot serve.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine, MutableMapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast
from urllib.parse import urlsplit

import httpx
import msgspec

from skyward.core.errors import DaemonError, SkywardError, UnexpectedResponseError, refused
from skyward.shared.observability import logger
from skyward.shared.schemas import Liveness
from skyward.shared.version import current

if TYPE_CHECKING:
    from litestar import Litestar

logger = logger.bind(component="client")

BLOB = "application/vnd.skyward.blob"
JSON = "application/json"

HOST = "127.0.0.1"
PORT = 17590
DEFAULT_URL = f"http://{HOST}:{PORT}"
"""Where ``sky server start`` binds, and so where a pool looks for one already up."""

START_TIMEOUT = 30.0
POLL_SECONDS = 0.2

RETRY_SECONDS = 30.0
"""How long a request outlives the daemon it was talking to.

A daemon bounce is seconds, and every mutating route was built to be re-asked —
submits and cancels carry an ``Idempotency-Key``, blobs are addressed by their
content, a lease renew renews. A daemon that is actually gone is a different
thing, and past this the error is the answer.
"""

NOT_A_DAEMON = (httpx.HTTPError, OSError, msgspec.DecodeError, SkywardError, UnexpectedResponseError)
"""What asking an address whether it holds a daemon can fail with.

None of it is a bug here. A closed port, a hung connection, something else
listening and answering in a shape of its own — each one means the same thing,
which is that there is no daemon of ours there to talk to.
"""

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
        from skyward.server.http.app import create_app, services
        from skyward.server.persistence.db import connect

        await connect(database)
        app = create_app(services(), logging=False)

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

    async def liveness(self) -> Liveness | None:
        """What the daemon says about itself, or nothing when none answers.

        No patience here: this is the probe :func:`connect` uses to decide whether
        to start a daemon, and a probe that retries for half a minute is a daemon
        that takes half a minute to be found missing.
        """
        try:
            response = await self._send("GET", "/v1/health/live", None, JSON, None, {}, patience=0.0)
            return msgspec.json.decode(response.content, type=Liveness)
        except NOT_A_DAEMON:
            return None

    async def close(self) -> None:
        await self._stack.aclose()

    async def call[T](
        self,
        method: str,
        path: str,
        kind: type[T],
        /,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        **query: object,
    ) -> T:
        """The route is positional, so a query of its own may be called ``path``."""
        response = await self._send(method, path, body, JSON, headers, query)
        return msgspec.json.decode(response.content, type=kind)

    async def blob(self, path: str, **query: object) -> bytes | None:
        """The bytes, or None if the server had nothing to give yet (204)."""
        response = await self._send("GET", path, None, JSON, None, query)
        return response.content if response.status_code == 200 else None

    async def upload(self, path: str, body: bytes, headers: dict[str, str] | None = None) -> None:
        await self._send("PUT", path, body, BLOB, headers, {})

    async def delete(self, path: str) -> None:
        """For the endpoints that answer 204 — nothing to decode, so not ``call``."""
        await self._send("DELETE", path, None, JSON, None, {})

    async def download(self, path: str, /, **query: object) -> AsyncGenerator[bytes]:
        """A response body as raw bytes, as it arrives.

        :meth:`blob` for a body that has no length to wait for. A file on a node is
        as large as the user made it, and reading it into memory here to hand back
        one ``bytes`` would size the client to the file. Pulled a chunk at a time
        and only as the caller writes them onward, so the machine's own read is
        what the caller's disk paces.
        """
        params = {key: str(value) for key, value in query.items() if value is not None}
        async with self._http.stream("GET", path, params=params) as response:
            if response.status_code >= 400:
                await response.aread()
                raise refused(response.status_code, response.content)
            async for chunk in response.aiter_bytes():
                yield chunk

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
                raise refused(response.status_code, response.content)

            buffer = bytearray()
            async for chunk in response.aiter_bytes():
                buffer += chunk
                while len(buffer) >= 4 and len(buffer) >= 4 + int.from_bytes(buffer[:4], "big"):
                    size = int.from_bytes(buffer[:4], "big")
                    yield bytes(buffer[4 : 4 + size])
                    del buffer[: 4 + size]

    async def events(self, compute: str) -> AsyncGenerator[tuple[str, bytes]]:
        """One compute's event log, replayed and then followed.

        The replay is what makes subscribing late harmless: the log is in the store,
        and the stream starts at the beginning of it. Nothing said before anybody was
        listening is lost — which is the only reason the pool can print a bootstrap
        it did not subscribe to in time.
        """
        async with self._http.stream("GET", "/v1/events", params={"compute": compute}) as response:
            event = ""
            async for line in response.aiter_lines():
                match line.split(": ", 1):
                    case ["event", name]:
                        event = name
                    case ["data", payload]:
                        yield event, payload.encode()
                    case _:
                        continue

    async def forward_up(self, compute: str, cid: str, port: int, route: str, chunks: AsyncIterator[bytes]) -> None:
        """Send one connection's bytes up to a node, as a streaming request body.

        The request stands open for the life of the connection: its body is the
        socket, and it ends when the socket does. The daemon opens the channel on
        the id this shares with :meth:`forward_down` and pumps the rest into the
        node. Backpressure is the body itself — the next chunk is pulled only once
        the node has taken the last.
        """
        response = await self._http.request(
            "POST",
            f"/v1/computes/{compute}/forward/up",
            content=chunks,
            params={"cid": cid, "port": str(port), "route": route},
            headers={"Content-Type": "application/octet-stream"},
        )
        if response.status_code >= 400:
            raise refused(response.status_code, response.content)

    async def forward_down(self, compute: str, cid: str) -> AsyncGenerator[bytes]:
        """The node's bytes for this connection, raw and as they arrive.

        No framing: a byte proxy has no frames to find. The stream is pulled one
        chunk at a time and only as the caller writes them onward, which is what
        carries the node's backpressure the whole way to the local socket.
        """
        async with self._http.stream("GET", f"/v1/computes/{compute}/forward/down", params={"cid": cid}) as response:
            if response.status_code >= 400:
                await response.aread()
                raise refused(response.status_code, response.content)
            async for chunk in response.aiter_bytes():
                yield chunk

    async def shell_up(
        self,
        compute: str,
        cid: str,
        node: str | None,
        command: str | None,
        term: str,
        size: tuple[int, int],
        chunks: AsyncIterator[bytes],
    ) -> None:
        """Send one session's keystrokes up to a node's terminal, as a streaming body.

        :meth:`forward_up` with a terminal on the far end: the request stands open for
        the life of the session, and the daemon opens the pseudo-terminal on the id
        this shares with :meth:`shell_down`.
        """
        columns, rows = size
        node_at = {"node": node} if node else {}
        running = {"command": command} if command else {}
        response = await self._http.request(
            "POST",
            f"/v1/computes/{compute}/shell/up",
            content=chunks,
            params={"cid": cid, "term": term, "columns": str(columns), "rows": str(rows), **node_at, **running},
            headers={"Content-Type": "application/octet-stream"},
        )
        if response.status_code >= 400:
            raise refused(response.status_code, response.content)

    async def shell_down(self, compute: str, cid: str) -> AsyncGenerator[bytes]:
        """What the terminal paints, raw and as it arrives."""
        async with self._http.stream("GET", f"/v1/computes/{compute}/shell/down", params={"cid": cid}) as response:
            if response.status_code >= 400:
                await response.aread()
                raise refused(response.status_code, response.content)
            async for chunk in response.aiter_bytes():
                yield chunk

    async def _send(
        self,
        method: str,
        path: str,
        body: bytes | None,
        content_type: str,
        headers: dict[str, str] | None,
        query: dict[str, object],
        *,
        patience: float = RETRY_SECONDS,
    ) -> httpx.Response:
        """One request, re-sent through a daemon bounce.

        Only transport failures are retried — a connection refused or reset is the
        daemon being restarted under us, and every route is safe to re-ask (see
        :data:`RETRY_SECONDS`). An HTTP refusal is an answer, and it stands.
        """
        deadline = asyncio.get_running_loop().time() + patience
        delay = POLL_SECONDS
        while True:
            try:
                response = await self._http.request(
                    method,
                    path,
                    content=body,
                    params={key: str(value) for key, value in query.items() if value is not None},
                    headers={"Content-Type": content_type, **(headers or {})},
                )
            except httpx.TransportError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)
                continue
            if response.status_code >= 400:
                raise refused(response.status_code, response.content)
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

        source = request.stream
        assert isinstance(source, httpx.AsyncByteStream)
        body = aiter(source)
        drained = False
        disconnected = asyncio.Event()

        async def receive() -> Message:
            """The request body a chunk at a time, then nothing until the caller leaves.

            Reading the whole body up front and handing it over in one message is the
            eager case, and for a request that streams — a forwarded socket, an upload
            with no end in sight — it is the wrong one: it would wait for a body that
            closes only when the caller does. So the chunks are pulled as the app asks
            and the body is declared over only when the iterator is.

            Once it is, answering ``http.request`` again is what a server would never
            do, and what a streaming response takes as an invitation to ask once more:
            it is listening for the disconnect this is now waiting to deliver.
            """
            nonlocal drained
            if drained:
                await disconnected.wait()
                return {"type": "http.disconnect"}
            try:
                return {"type": "http.request", "body": await anext(body), "more_body": True}
            except StopAsyncIteration:
                drained = True
                return {"type": "http.request", "body": b"", "more_body": False}

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


async def connect(url: str | None, database: Path | None) -> Client:
    """The daemon a pool speaks to: one it was named, one already up, or one it starts.

    A pool that names no daemon looks where ``sky server start`` binds, and starts
    a daemon there when nothing answers. It does not embed one: an embedded plane
    would open the database that daemon owns the moment somebody starts one, and
    two control planes over one database are two reconcilers buying the same
    machine. The daemon it starts outlives it, which is the point of a daemon — the
    machines it bought are still running when the script ends.

    Naming a database is the one way to be the plane instead of reaching one. That
    argument says which file to serve, and a daemon holding a different one has no
    answer to it.
    """
    if target := url or os.environ.get("SKYWARD_URL"):
        return await dial(target, start=False)

    if database is not None:
        return await Client.embedded(database)

    return await dial(DEFAULT_URL, start=True)


async def dial(url: str, *, start: bool) -> Client:
    """A client on the daemon at ``url``, once it has answered for itself.

    The handshake is the liveness call, and what it is really asking is the version:
    a daemon of another skyward serves other wire types behind the same routes, and
    what that becomes is a decode error somewhere in the middle of a provision, long
    after machines are billing. Refused here, it costs a round trip.

    ``start`` is for the address nobody named — a daemon may be started there
    because that is where one belongs. A named url is somewhere the caller says a
    daemon already is, and binding a port over that answer would be inventing one.
    """
    client = await Client.remote(url)
    try:
        if (live := await client.liveness()) is None:
            if not start:
                raise DaemonError(f"no daemon answers at {url}")
            print("skyward: no server is running, starting it now", file=sys.stderr, flush=True)
            live = await started(client, url)
            print(f"skyward: the daemon at {url} stays up after this run — `sky server stop` ends it", file=sys.stderr, flush=True)

        if live.version != (here := current()):
            theirs = f"skyward {live.version}" if live.version else "a skyward too old to say which"
            raise DaemonError(
                f"the daemon at {url} runs {theirs}, this process runs skyward {here} — "
                "stop it with `sky server stop` and run again, or point at a daemon on this version"
            )
    except BaseException:
        await client.close()
        raise
    return client


async def started(client: Client, url: str) -> Liveness:
    """Start a daemon at ``url`` and wait for it to answer.

    The pid is written down only once something has answered, and only if the
    process that was started is the one still there: two pools starting at the same
    instant both spawn, one takes the port and the other dies of it, and a pidfile
    naming the loser is a ``sky server stop`` that stops nothing.
    """
    from skyward.server import daemon

    host, port = address(url)
    process = daemon.spawn(host, port)
    deadline = time.monotonic() + START_TIMEOUT

    while True:
        if (live := await client.liveness()) is not None:
            if daemon.alive(process):
                daemon.record(process)
            return live
        if not daemon.alive(process):
            raise DaemonError(f"the daemon exited without answering — see {daemon.LOG_FILE}")
        if time.monotonic() >= deadline:
            raise DaemonError(f"no answer from {url} within {START_TIMEOUT:.0f}s — see {daemon.LOG_FILE}")
        await asyncio.sleep(POLL_SECONDS)


def address(url: str) -> tuple[str, int]:
    """Where a daemon has to bind for ``url`` to reach it."""
    parsed = urlsplit(url)
    return parsed.hostname or HOST, parsed.port or PORT
