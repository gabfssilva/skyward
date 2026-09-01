"""Which control plane a pool ends up talking to.

A pool that names nothing does not merely look for a daemon, it starts one and
leaves it behind: the machines it bought outlive the block, and something has to
be reconciling them afterwards. So the questions here are whether it finds the
daemon already up, whether it starts one when none answers, whether it leaves
that one running, and whether it refuses one that is running another skyward.
"""

import asyncio
import os
import signal
import socket
import time
from collections.abc import Callable, Iterator
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import msgspec
import pytest

from skyward.core.client import Client, connect
from skyward.core.errors import DaemonError
from skyward.server import daemon
from skyward.shared.schemas import Page, Provider, ProviderCreate

pytestmark = pytest.mark.local


async def _register(url: str, name: str) -> None:
    """Write a provider row through the daemon at ``url``, so it can be looked for."""
    client = await Client.remote(url)
    try:
        body = ProviderCreate(name=name, kind="container", credentials={}, config={})
        await client.call("POST", "/v1/providers", Provider, body=msgspec.json.encode(body))
    finally:
        await client.close()


async def _accounts(client: Client) -> list[str]:
    """What that client can see, and then let it go."""
    try:
        page = await client.call("GET", "/v1/providers", Page[Provider])
        return [provider.name for provider in page.items]
    finally:
        await client.close()


async def _answers(url: str) -> bool:
    client = await Client.remote(url)
    try:
        return await client.liveness() is not None
    finally:
        await client.close()


def _local(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.delenv("SKYWARD_URL", raising=False)
    monkeypatch.setattr("skyward.core.client.DEFAULT_URL", url)


@pytest.fixture
def nowhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A free address with nothing at it, and somewhere of its own to put a daemon.

    The pidfile, the log and the database are this test's, so a daemon started
    here is one the test can end and one that never touches ``~/.skyward``.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    _local(monkeypatch, f"http://127.0.0.1:{port}")
    monkeypatch.setenv("SKYWARD_DATABASE", str(tmp_path / "started.sqlite"))
    monkeypatch.setattr(daemon, "PID_FILE", tmp_path / "server.pid")
    monkeypatch.setattr(daemon, "LOG_FILE", tmp_path / "server.log")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        if (process := daemon.pid()) and daemon.alive(process):
            os.kill(process, signal.SIGTERM)


def describe_a_pool_that_names_no_daemon() -> None:
    def it_reaches_the_one_already_running(alone: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _local(monkeypatch, alone)

        async def resolved() -> list[str]:
            await _register(alone, "written-through-the-daemon")
            return await _accounts(await connect(None, None))

        assert "written-through-the-daemon" in asyncio.run(resolved()), "the pool hosted its own plane instead of using the daemon"

    def it_starts_one_when_nothing_answers(nowhere: str, capsys: pytest.CaptureFixture[str]) -> None:
        async def started() -> bool:
            await (await connect(None, None)).close()
            return await _answers(nowhere)

        assert asyncio.run(started()), "nothing was serving where the pool had just started a daemon"
        assert "no server is running, starting it now" in capsys.readouterr().err
        assert daemon.pid() is not None, "the daemon was started without a pid to stop it by"

    def it_leaves_the_daemon_it_started_running(nowhere: str) -> None:
        async def start_and_leave() -> bool:
            await (await connect(None, None)).close()
            return await _answers(nowhere)

        assert asyncio.run(start_and_leave()), "the pool took the daemon with it"


def describe_a_pool_that_names_a_daemon() -> None:
    def it_starts_none_when_that_one_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(daemon, "PID_FILE", tmp_path / "server.pid")

        with pytest.raises(DaemonError, match="no daemon answers"):
            asyncio.run(connect("http://127.0.0.1:1", None))

        assert not (tmp_path / "server.pid").exists(), "a named daemon that is not there is an error, not an address to bind"


def describe_a_pool_that_names_a_database() -> None:
    def it_stays_embedded_even_with_a_daemon_up(alone: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _local(monkeypatch, alone)

        async def resolved() -> list[str]:
            await _register(alone, "only-in-the-daemon")
            return await _accounts(await connect(None, tmp_path / "own.sqlite"))

        assert asyncio.run(resolved()) == [], "a named database is the file to be the plane for, not a hint"


def describe_a_daemon_running_another_skyward() -> None:
    def it_is_refused_before_a_single_machine_is_bought(alone: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _local(monkeypatch, alone)
        monkeypatch.setattr("skyward.core.client.current", lambda: "0.0.0+other")

        with pytest.raises(DaemonError, match="0.0.0\\+other") as refusal:
            asyncio.run(connect(None, None))

        assert "sky server stop" in str(refusal.value), "the refusal has to say what to do about it"


def describe_a_daemon_that_bounces_mid_request() -> None:
    def it_is_asked_again_until_it_comes_back() -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(200, content=msgspec.json.encode(Page(items=[], next_cursor=None)))

        async def survive_the_bounce() -> Page[Provider]:
            client = _mocked(handler)
            try:
                return await client.call("GET", "/v1/providers", Page[Provider])
            finally:
                await client.close()

        page = asyncio.run(survive_the_bounce())

        assert attempts == 3, "the request must outlive the bounce, not fail on the first refused connection"
        assert page.items == ()

    def it_probes_liveness_without_waiting_for_one() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        async def probe() -> object:
            client = _mocked(handler)
            try:
                return await client.liveness()
            finally:
                await client.close()

        started = time.monotonic()

        assert asyncio.run(probe()) is None
        assert time.monotonic() - started < 1.0, "the probe decides whether to start a daemon; it must not wait for one"


def _mocked(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://skyward")
    return Client(http, AsyncExitStack())


def describe_leaving_a_pool_that_borrowed_a_daemon() -> None:
    def it_does_not_take_the_daemon_with_it(alone: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _local(monkeypatch, alone)

        async def borrow_and_return() -> bool:
            await (await connect(None, None)).close()
            return await _answers(alone)

        assert asyncio.run(borrow_and_return()), "closing a borrowed client must not end the daemon it borrowed"
