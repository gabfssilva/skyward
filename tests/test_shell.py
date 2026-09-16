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

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine

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


def describe_a_machine_that_is_still_bootstrapping() -> None:
    async def it_takes_a_terminal_anyway() -> None:
        async with _Holding(0) as runtime:
            reader, _ = await runtime.open_shell(size=(100, 40))

            assert await reader.read(65536) == b"xterm-256color 100x40 rank 0\r\n", "a pty, sized as asked, on a node with no worker on it"

    async def it_is_told_apart_from_one_this_daemon_has_let_go_of() -> None:
        async with _Holding(0, 1) as runtime:
            await runtime.detach(runtime.held[0])

            reader, _ = await runtime.open_shell()

            assert b"rank 1" in await reader.read(65536), "a closed channel is not a machine anybody can be sent to"


def describe_choosing_the_machine() -> None:
    async def it_takes_the_lowest_rank_when_none_is_named() -> None:
        async with _Holding(2, 0, ready=(2,)) as runtime:
            reader, _ = await runtime.open_shell()

            assert b"rank 0" in await reader.read(65536), "lowest rank, not the first one tracked and not the ready one"

    async def it_takes_the_rank_it_was_given() -> None:
        async with _Holding(0, 2, ready=(0,)) as runtime:
            reader, _ = await runtime.open_shell(rank=2)

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

    def __init__(self, *ranks: int, ready: tuple[int, ...] = ()) -> None:
        self._ranks = ranks
        self._ready = ready
        self._acceptors: list[asyncssh.SSHAcceptor] = []
        self._runtime = Runtime("cmp_1", "pypi", private_key=KEY)

    async def __aenter__(self) -> Runtime:
        for rank in self._ranks:
            acceptor = await asyncssh.listen(
                "127.0.0.1",
                0,
                server_host_keys=[asyncssh.generate_private_key("ssh-ed25519")],
                server_factory=_Doorman,
                process_factory=_terminal(rank),
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


def _node(port: int, rank: int) -> Node:
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
        options=Options(ssh_connect_timeout=10.0, ssh_retry_delay=0.05),
    )
