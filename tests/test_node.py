"""How a node notices that the machine under it is gone.

The channel to a machine heals on its own, and the node above it waits while it
does. But a channel that gives up healing, and one that heals onto a machine that
no longer has the worker — a container rescheduled onto another host, a preempted
VM restarted from its image — both used to leave the node ``ready`` forever: the
health probe died of the channel's error, the event tail swallowed it, and nothing
else was watching.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from skyward.server.application.node import DEFAULT_OPTIONS, Node
from skyward.server.application.source import Source
from skyward.server.application.ssh import Result, Ssh, SshUnavailableError
from skyward.shared.provider import Machine
from skyward.shared.schemas import Image, NodeState, Options

pytestmark = pytest.mark.local

type Report = tuple[NodeState, str | None]


class _Link(Ssh):
    """A channel scripted by its tails: each ``stream`` call plays the next one."""

    def __init__(self, tails: list[list[str] | SshUnavailableError], worker_alive: bool = True) -> None:
        self._tails = tails
        self._worker_alive = worker_alive
        self.commands: list[str] = []

    async def run(self, command: str, *, timeout: float | None = None) -> Result:
        self.commands.append(command)
        if not self._tails:
            raise SshUnavailableError("127.0.0.1: reconnection exhausted")
        return Result(exit_code=0 if self._worker_alive else 1, stdout="", stderr="")

    async def put(self, path: str, content: bytes) -> None:
        pass

    async def stream(self, command: str) -> AsyncIterator[str]:
        match self._tails.pop(0) if self._tails else SshUnavailableError("127.0.0.1: reconnection exhausted"):
            case SshUnavailableError() as gone:
                raise gone
            case lines:
                for line in lines:
                    yield line

    async def forward(self, remote_port: int, remote_host: str = "127.0.0.1") -> int:
        return 0

    async def close(self) -> None:
        pass

    @property
    def connected(self) -> bool:
        return bool(self._tails)


def _node(link: _Link, reports: list[Report], ready: bool = True, options: Options = DEFAULT_OPTIONS) -> Node:
    node = Node(
        Machine(id="m-1", state="running", host="10.0.0.1"),
        compute="cmp_test",
        private_key="key",
        image=Image(),
        source=Source(arguments=("skyward",)),
        listener=lambda state, error: reports.append((state, error)),
        output=lambda content, task: None,
        sample=lambda name, value: None,
        phase=lambda event, phase, error: None,
        options=options,
    )
    node._ssh = link
    if ready:
        node.tunnel = 40000
    return node


def describe_a_link_that_gives_up_reconnecting() -> None:
    async def it_reports_the_node_lost() -> None:
        reports: list[Report] = []
        node = _node(_Link(tails=[]), reports)

        async with asyncio.timeout(5):
            await node._watch()

        assert reports == [("lost", "127.0.0.1: reconnection exhausted")]

    async def the_probe_stops_quietly_instead_of_dying() -> None:
        reports: list[Report] = []
        node = _node(_Link(tails=[]), reports, options=Options(health_interval=0.0))

        async with asyncio.timeout(5):
            await node._health("true")

        assert reports == []


def describe_a_link_that_heals() -> None:
    async def onto_a_machine_without_the_worker_it_reports_the_node_lost() -> None:
        reports: list[Report] = []
        link = _Link(tails=[[], []], worker_alive=False)
        node = _node(link, reports)

        async with asyncio.timeout(5):
            await node._watch()

        assert reports == [("lost", "the machine came back without its worker")]
        assert any("pgrep" in command for command in link.commands)

    async def onto_the_same_machine_it_keeps_following_the_log() -> None:
        reports: list[Report] = []
        phases: list[str] = []
        link = _Link(tails=[[], ['{"type":"phase","event":"completed","phase":"later"}']], worker_alive=True)
        node = _node(link, reports)
        node._phase = lambda event, phase, error: phases.append(phase)

        async with asyncio.timeout(5):
            await node._watch()

        assert phases == ["later"]
        assert reports == [("lost", "127.0.0.1: reconnection exhausted")]

    async def before_the_node_is_ready_it_asks_nothing_of_the_machine() -> None:
        reports: list[Report] = []
        link = _Link(tails=[[], []], worker_alive=False)
        node = _node(link, reports, ready=False)

        async with asyncio.timeout(5):
            await node._watch()

        assert link.commands == []
        assert reports == [("lost", "127.0.0.1: reconnection exhausted")]
