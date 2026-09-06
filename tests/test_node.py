"""How a node notices that the machine under it is gone.

The channel to a machine heals on its own, and the node above it waits while it
does. But a channel that gives up healing, and one that heals onto a machine that
no longer has the worker — a container rescheduled onto another host, a preempted
VM restarted from its image — both used to leave the node ``ready`` forever: the
health probe died of the channel's error, the event tail swallowed it, and nothing
else was watching.
"""

import asyncio
import random
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


async def _quiet(*_: object) -> None:
    pass


def _node(link: _Link, reports: list[Report], ready: bool = True, options: Options = DEFAULT_OPTIONS) -> Node:
    node = Node(
        Machine(id="m-1", state="running", host="10.0.0.1"),
        compute="cmp_test",
        private_key="key",
        image=Image(),
        source=Source(arguments=("skyward",)),
        listener=lambda state, error: reports.append((state, error)),
        output=_quiet,
        sample=_quiet,
        phase=_quiet,
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

        async def noted(event: str, phase: str, error: str | None) -> None:
            phases.append(phase)

        node._phase = noted

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


def describe_what_the_log_says() -> None:
    async def is_reported_in_the_order_it_was_read_however_long_each_report_takes() -> None:
        lines = [
            '{"type":"phase","event":"started","phase":"apt"}',
            '{"type":"console","content":"Reading package lists..."}',
            '{"type":"phase","event":"completed","phase":"apt"}',
            '{"type":"phase","event":"started","phase":"uv"}',
            '{"type":"metric","name":"cpu","value":3.5}',
            '{"type":"phase","event":"completed","phase":"uv"}',
            '{"type":"phase","event":"started","phase":"venv"}',
            '{"type":"phase","event":"completed","phase":"venv"}',
        ]
        said: list[str] = []
        node = _node(_Link(tails=[lines]), [])

        async def slowly(*words: object) -> None:
            await asyncio.sleep(random.uniform(0, 0.01))
            said.append(" ".join(str(word) for word in words if word is not None))

        node._output = slowly
        node._sample = slowly
        node._phase = slowly

        async with asyncio.timeout(5):
            await node._watch()

        assert said == [
            "started apt",
            "Reading package lists...",
            "completed apt",
            "started uv",
            "cpu 3.5",
            "completed uv",
            "started venv",
            "completed venv",
        ]
