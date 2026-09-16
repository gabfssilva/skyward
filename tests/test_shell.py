"""A terminal on a machine, and which machine it lands on.

Opening one used to need a *ready* node: a node with a tunnel, which means a
worker, which means the bootstrap finished. That excluded the minutes somebody
most wants to be inside a machine — a driver installing, a wheel resolving, a
bootstrap that is never going to finish — and in the last case it excluded them
forever: the machine that needs looking at is exactly the one that never becomes
ready.

So the choice is over the machines this daemon holds a link to, which is every
machine that has answered SSH, and the wait for one that has not answered yet is
the channel's, not the caller's.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import suppress

import asyncssh
import httpx
import pytest

from skyward.server.application.node import Node
from skyward.server.application.runtimes import Runtime, Runtimes, Terminal, keypair
from skyward.server.application.source import Source
from skyward.server.http.app import create_app, with_real
from skyward.shared.errors import ComputeNotConnectedError
from skyward.shared.provider import Machine
from skyward.shared.schemas import Image, Options

pytestmark = pytest.mark.local

KEY, _ = keypair()

type Session = Callable[[int], Callable[[asyncssh.SSHServerProcess[str]], Coroutine[None, None, None]]] | None
"""What one machine does with a session, given its rank."""


def describe_a_machine_that_is_still_bootstrapping() -> None:
    async def it_takes_a_terminal_anyway() -> None:
        async with _Holding(0) as runtime:
            reader, _ = (await runtime.open_shell(size=(100, 40))).channel

            assert await reader.read(65536) == b"xterm-256color 100x40 rank 0\r\n", "a pty, sized as asked, on a node with no worker on it"

    async def one_that_gives_up_while_it_waits_is_refused_by_name() -> None:
        """Held is not connected. A machine can still be dialling when somebody asks to be let in, and
        the wait belongs to the channel — so the channel giving up is what ends the wait, and it has to
        end it as a refusal rather than as whatever the ssh layer called it."""
        runtime = Runtime("cmp_1", "pypi", private_key=KEY)
        node = _node(_unanswered(), rank=0, connect_timeout=0.5)
        runtime.track("nod_0", node)
        dialling = asyncio.create_task(node._ssh.connect())

        with pytest.raises(ComputeNotConnectedError, match="lost its link"):
            await runtime.open_shell()

        with suppress(Exception):
            await dialling
        await runtime.close()

    async def it_is_told_apart_from_one_this_daemon_has_let_go_of() -> None:
        async with _Holding(0, 1) as runtime:
            await runtime.detach(runtime.held[0])

            reader, _ = (await runtime.open_shell()).channel

            assert b"rank 1" in await reader.read(65536), "a closed channel is not a machine anybody can be sent to"


def describe_choosing_the_machine() -> None:
    async def it_takes_the_lowest_rank_when_none_is_named() -> None:
        async with _Holding(2, 0, ready=(2,)) as runtime:
            reader, _ = (await runtime.open_shell()).channel

            assert b"rank 0" in await reader.read(65536), "lowest rank, not the first one tracked and not the ready one"

    async def it_takes_the_rank_it_was_given() -> None:
        async with _Holding(0, 2, ready=(0,)) as runtime:
            reader, _ = (await runtime.open_shell(rank=2)).channel

            assert b"rank 2" in await reader.read(65536)

    async def a_rank_it_is_not_holding_is_refused_by_name() -> None:
        async with _Holding(0, 1) as runtime:
            with pytest.raises(ComputeNotConnectedError) as refused:
                await runtime.open_shell(rank=7)

        assert "rank 7" in refused.value.message
        assert "rank(s) 0, 1" in refused.value.message, "the ranks it does hold are what the caller needs to hear"
        assert refused.value.retryable, "the machine may be minutes from answering"


def describe_a_compute_with_no_machine_up_yet() -> None:
    async def it_is_told_rather_than_raised_through() -> None:
        runtime = Runtime("cmp_1", "pypi", private_key=KEY)

        with pytest.raises(ComputeNotConnectedError) as refused:
            await runtime.open_shell()

        assert refused.value.status == 409
        assert refused.value.retryable


def describe_the_two_halves_of_a_session() -> None:
    async def they_carry_a_terminal_between_them() -> None:
        async with _Holding(0) as runtime, _daemon(runtime) as http:
            up, down = await asyncio.gather(
                http.post("/v1/computes/cmp_1/shell/up", params={"cid": "s1", "node": 0, "columns": 90, "rows": 30}, content=_nothing()),
                http.get("/v1/computes/cmp_1/shell/down", params={"cid": "s1"}),
            )

        assert up.status_code == 200, up.text
        assert down.content == b"xterm-256color 90x30 rank 0\r\n"

    async def a_rank_nobody_is_holding_is_refused_on_both_of_them() -> None:
        """The down half used to answer 200 and then die mid-chunk, which reaches the caller as a protocol error."""
        async with _Holding(0) as runtime, _daemon(runtime) as http:
            up, down = await asyncio.gather(
                http.post("/v1/computes/cmp_1/shell/up", params={"cid": "s2", "node": 9}, content=_nothing()),
                http.get("/v1/computes/cmp_1/shell/down", params={"cid": "s2"}),
            )

        assert up.status_code == 409, up.text
        assert down.status_code == 409, "a refusal has to arrive as an answer, not as a body that stops"
        assert up.json()["code"] == down.json()["code"] == "compute_not_connected"


def describe_a_terminal_over_one_socket() -> None:
    """What the browser uses, because it cannot write a request body it is still reading the answer to."""

    async def it_carries_both_directions_without_an_id_to_tie_them() -> None:
        async with _Holding(0, session=_echo) as runtime, _Socket(runtime, "columns=100&rows=40") as session:
            assert await session.next() == {"type": "websocket.accept", "subprotocol": None, "headers": []}
            assert "100x40" in await session.painted("100x40"), "the pty opened at the size the query asked for"

            await session.typed(b"knocked\n")

            assert "knocked" in await session.painted("knocked"), "a binary frame is the keyboard, and it came back painted"

    async def a_window_that_moves_is_carried_mid_session() -> None:
        """The one thing the paired halves cannot say: they carry the size once, in the query that opens them."""
        async with _Holding(0, session=_echo) as runtime, _Socket(runtime, "columns=80&rows=24") as session:
            await session.next()
            await session.painted("80x24")

            await session.said('{"columns": 132, "rows": 50}')

            assert "132x50" in await session.painted("132x50"), "the far end was told, and said so"

    async def a_frame_that_is_not_a_shape_does_not_end_the_session() -> None:
        async with _Holding(0, session=_echo) as runtime, _Socket(runtime, "") as session:
            await session.next()
            await session.painted("80x24")

            await session.said("what?")
            await session.typed(b"still here\n")

            assert "still here" in await session.painted("still here")

    async def a_rank_nobody_holds_is_an_error_frame_and_then_a_close() -> None:
        """A rejected handshake would tell a browser nothing: onerror carries no status and no body."""
        async with _Holding(0) as runtime, _Socket(runtime, "node=9") as session:
            assert await session.next() == {"type": "websocket.accept", "subprotocol": None, "headers": []}

            match await session.next():
                case {"text": str(refusal)}:
                    assert '"code":"compute_not_connected"' in refusal.replace(" ", "")
                    assert "rank 9" in refusal, "and it says which rank, and which ranks there were"
                case other:
                    raise AssertionError(f"the refusal has to arrive before the close: {other}")

            assert await session.next() == {"type": "websocket.close", "code": 4409, "reason": "compute_not_connected"}

    async def a_shell_that_exits_closes_the_socket() -> None:
        async with _Holding(0) as runtime, _Socket(runtime, "") as session:
            await session.next()
            await session.painted("rank 0")

            match await session.next():
                case {"type": "websocket.close"}:
                    pass
                case other:
                    raise AssertionError(f"the session is over when the shell is: {other}")


def _daemon(runtime: Runtime) -> httpx.AsyncClient:
    """The daemon's app, reached over ASGI in this test's own event loop.

    Not the test client: that one runs the app in a loop of its own, and an SSH
    channel belongs to the loop that dialled it.
    """
    runtimes = Runtimes(listener=lambda *_: None, output=_quiet, sample=_quiet, phase=_quiet)
    runtimes._runtimes[runtime.compute] = runtime
    app = create_app(with_real(runtimes=runtimes, shell=Terminal(runtimes)), logging=False)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://daemon")


async def _quiet(*_: object) -> None:
    pass


async def _nothing() -> AsyncIterator[bytes]:
    return
    yield b""


class _Holding:
    """A runtime holding one machine per rank, each an SSH server in this process.

    The ranks are given in the order the daemon took hold of them, which is not
    their order: a machine replaced mid-run is tracked last and still answers to
    its rank. ``ready`` names the ones that finished bootstrapping and have a
    tunnel — what the selection used to require, and what these tests are about
    not requiring.
    """

    def __init__(self, *ranks: int, ready: tuple[int, ...] = (), session: Session = None) -> None:
        self._ranks = ranks
        self._ready = ready
        self._session = session or _terminal
        self._acceptors: list[asyncssh.SSHAcceptor] = []
        self._runtime = Runtime("cmp_1", "pypi", private_key=KEY)

    async def __aenter__(self) -> Runtime:
        for rank in self._ranks:
            acceptor = await asyncssh.listen(
                "127.0.0.1",
                0,
                server_host_keys=[asyncssh.generate_private_key("ssh-ed25519")],
                server_factory=_Doorman,
                process_factory=self._session(rank),
            )
            self._acceptors.append(acceptor)
            node = _node(acceptor.get_port(), rank)
            await node._ssh.connect()
            if rank in self._ready:
                node.tunnel = 40000 + rank
            self._runtime.track(f"nod_{rank}", node)
        return self._runtime

    async def __aexit__(self, *_: object) -> None:
        await self._runtime.close()
        for acceptor in self._acceptors:
            acceptor.close()


class _Doorman(asyncssh.SSHServer):
    """A machine that admits the key it was given, the way a provisioned one does."""

    def begin_auth(self, username: str) -> bool:
        return True

    def public_key_auth_supported(self) -> bool:
        return True

    def validate_public_key(self, username: str, key: asyncssh.SSHKey) -> bool:
        return True


def _terminal(rank: int) -> Callable[[asyncssh.SSHServerProcess[str]], Coroutine[None, None, None]]:
    """A session that says what terminal it was given, and on which machine."""

    async def session(process: asyncssh.SSHServerProcess[str]) -> None:
        columns, lines = process.get_terminal_size()[:2]
        process.stdout.write(f"{process.get_terminal_type()} {columns}x{lines} rank {rank}\n")
        process.exit(0)

    return session


def _unanswered() -> int:
    """A port nothing is listening on, for a machine that is never going to answer."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _node(port: int, rank: int, connect_timeout: float = 10.0) -> Node:
    return Node(
        Machine(id=f"m{rank}", state="running", host="127.0.0.1", port=port),
        compute="cmp_1",
        private_key=KEY,
        image=Image(),
        source=Source(arguments=("skyward",)),
        listener=lambda *_: None,
        output=_quiet,
        sample=_quiet,
        phase=_quiet,
        rank=rank,
        options=Options(ssh_connect_timeout=connect_timeout, ssh_retry_delay=0.05),
    )


def _echo(rank: int) -> Callable[[asyncssh.SSHServerProcess[str]], Coroutine[None, None, None]]:
    """A session that says its shape, repeats what is typed, and says so again when it changes.

    A resize reaches a server-side session as an exception raised into its read,
    which is asyncssh's way of saying the screen moved under a program that was
    waiting on the keyboard.
    """

    async def session(process: asyncssh.SSHServerProcess[str]) -> None:
        columns, lines = process.get_terminal_size()[:2]
        process.stdout.write(f"{columns}x{lines}\n")
        while True:
            try:
                typed = await process.stdin.readline()
            except asyncssh.TerminalSizeChanged as moved:
                process.stdout.write(f"{moved.width}x{moved.height}\n")
                continue
            if not typed:
                break
            process.stdout.write(typed)
        process.exit(0)

    return session


class _Socket:
    """One WebSocket against the daemon, spoken in ASGI in this test's own loop.

    The same reason :func:`_daemon` avoids the test client: a channel belongs to the
    loop that dialled it. A WebSocket on the server's side is a scope and two
    queues, so that is what this is — no client library, and nothing running
    anywhere else.
    """

    def __init__(self, runtime: Runtime, query: str) -> None:
        runtimes = Runtimes(listener=lambda *_: None, output=_quiet, sample=_quiet, phase=_quiet)
        runtimes._runtimes[runtime.compute] = runtime
        self._app = create_app(with_real(runtimes=runtimes, shell=Terminal(runtimes)), logging=False)
        self._query = query
        self._to_daemon: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._from_daemon: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def __aenter__(self) -> _Socket:
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "ws",
            "path": "/v1/computes/cmp_1/shell/attach",
            "raw_path": b"/v1/computes/cmp_1/shell/attach",
            "query_string": self._query.encode(),
            "root_path": "",
            "headers": [(b"host", b"daemon")],
            "client": ("127.0.0.1", 51234),
            "server": ("daemon", 80),
            "subprotocols": [],
            "state": {},
        }
        self._serving = asyncio.create_task(self._app(scope, self._to_daemon.get, self._from_daemon.put))
        await self._to_daemon.put({"type": "websocket.connect"})
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._to_daemon.put({"type": "websocket.disconnect", "code": 1000})
        with suppress(asyncio.TimeoutError):
            async with asyncio.timeout(5):
                await self._serving
        self._serving.cancel()

    async def typed(self, data: bytes) -> None:
        await self._to_daemon.put({"type": "websocket.receive", "bytes": data, "text": None})

    async def said(self, message: str) -> None:
        await self._to_daemon.put({"type": "websocket.receive", "bytes": None, "text": message})

    async def next(self) -> dict[str, object]:
        async with asyncio.timeout(10):
            return await self._from_daemon.get()

    async def painted(self, until: str) -> str:
        """Everything painted up to and including a line, however many frames it took."""
        seen = ""
        while until not in seen:
            match await self.next():
                case {"bytes": bytes(data)}:
                    seen += data.decode()
                case event:
                    raise AssertionError(f"the session ended before it painted {until!r}: {event}")
        return seen
