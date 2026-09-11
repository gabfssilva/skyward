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
import re
from collections.abc import AsyncIterator

import pytest

import skyward.server.application.node as node_module
from skyward.server.application.node import DEFAULT_OPTIONS, Node
from skyward.server.application.source import Source
from skyward.server.application.ssh import Result, Ssh, SshUnavailableError
from skyward.shared.provider import Machine
from skyward.shared.schemas import Image, NodeState, Options
from skyward.worker.journal import LOCK, Console

pytestmark = pytest.mark.local

type Report = tuple[NodeState, str | None]


class _Link(Ssh):
    """A channel scripted by its tails: each ``stream`` call plays the next one, as the lines it wrote."""

    def __init__(self, tails: list[list[str] | SshUnavailableError], worker_alive: bool = True) -> None:
        self._tails = tails
        self._worker_alive = worker_alive
        self.commands: list[str] = []
        self.followed: list[str] = []

    async def run(self, command: str, *, timeout: float | None = None) -> Result:
        self.commands.append(command)
        if not self._tails:
            raise SshUnavailableError("127.0.0.1: reconnection exhausted")
        return Result(exit_code=0 if self._worker_alive else 1, stdout="", stderr="")

    async def put(self, path: str, content: bytes) -> None:
        pass

    async def stream(self, command: str) -> AsyncIterator[bytes]:
        self.followed.append(command)
        match self._tails.pop(0) if self._tails else SshUnavailableError("127.0.0.1: reconnection exhausted"):
            case SshUnavailableError() as gone:
                raise gone
            case lines:
                for line in lines:
                    yield line.encode()

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
        link = _Link(tails=[[], ['{"type":"phase","event":"completed","phase":"later"}\n']], worker_alive=True)
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

    async def after_a_drop_cut_a_line_in_half_it_reads_that_line_again_whole() -> None:
        apt = '{"type":"phase","event":"completed","phase":"apt"}\n'
        uv = '{"type":"phase","event":"completed","phase":"uv"}\n'
        phases: list[str] = []
        link = _Link(tails=[[apt, uv[:20]], [uv]], worker_alive=True)
        node = _node(link, [])

        async def noted(event: str, phase: str, error: str | None) -> None:
            phases.append(phase)

        node._phase = noted

        async with asyncio.timeout(5):
            await node._watch()

        assert phases == ["apt", "uv"], "every phase once, the one the drop cut included"
        assert f"-c +{len(apt.encode()) + 1} " in link.followed[1], "the second tail starts right after the last whole line the first one read"


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
        node = _node(_Link(tails=[[f"{line}\n" for line in lines]]), [])

        async def slowly(*words: object) -> None:
            await asyncio.sleep(random.uniform(0, 0.01))
            said.append(" ".join(str(word) for word in words if word is not None))

        async def printed(lines: tuple[Console, ...]) -> None:
            await asyncio.sleep(random.uniform(0, 0.01))
            said.extend(line.content for line in lines)

        node._output = printed
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


class _File(_Link):
    """A machine whose log is a real byte string: tails read it from their offset, and the lock-held check empties it."""

    def __init__(self, lines: list[str], refills: list[list[str]] | None = None, grows: bool = False, clock: list[float] | None = None) -> None:
        super().__init__(tails=[[]])
        self.content = "".join(lines).encode()
        self._refills = refills or []
        self._grows = grows
        self._clock = clock
        self.truncations: list[tuple[int, bool]] = []
        self.tails: list[tuple[int, list[str]]] = []

    async def run(self, command: str, *, timeout: float | None = None) -> Result:
        self.commands.append(command)
        if f"flock {LOCK} " not in command:
            return Result(exit_code=0, stdout="", stderr="")
        match re.search(r"= (\d+) ", command):
            case None:
                raise AssertionError(f"no size in {command}")
            case found:
                offset = int(found.group(1))
        size = len(self.content) + (1 if self._grows else 0)
        emptied = size == offset
        self.truncations.append((offset, emptied))
        if emptied:
            self.content = "".join(self._refills.pop(0) if self._refills else []).encode()
        return Result(exit_code=0 if emptied else 1, stdout="", stderr="")

    async def stream(self, command: str) -> AsyncIterator[bytes]:
        self.followed.append(command)
        match re.search(r"-c \+(\d+) ", command):
            case None:
                raise AssertionError(f"no offset in {command}")
            case found:
                offset = int(found.group(1)) - 1
        unread = self.content[offset:]
        if not unread:
            raise SshUnavailableError("127.0.0.1: reconnection exhausted")
        read: list[str] = []
        self.tails.append((offset, read))
        for line in unread.decode().splitlines(keepends=True):
            if self._clock is not None:
                self._clock[0] += 4.0
            read.append(line)
            yield line.encode()


def _phase(name: str) -> str:
    return f'{{"type":"phase","event":"completed","phase":"{name}"}}\n'


def _console(content: str) -> str:
    return f'{{"type":"console","content":"{content}"}}\n'


def _heard(node: Node) -> list[str]:
    phases: list[str] = []

    async def noted(event: str, phase: str, error: str | None) -> None:
        phases.append(phase)

    node._phase = noted
    return phases


def describe_a_log_the_daemon_has_read_to_its_end() -> None:
    async def is_emptied_under_the_lock_and_followed_again_from_byte_zero(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(node_module, "ROTATE_BYTES", len(_phase("a")) + len(_phase("b")))
        monkeypatch.setattr(node_module, "ROTATE_SECONDS", 0.0)
        link = _File([_phase("a"), _phase("b")], refills=[[_phase("c"), _phase("d")]])
        node = _node(link, [])
        phases = _heard(node)

        async with asyncio.timeout(5):
            await node._watch()

        whole = len(_phase("a")) + len(_phase("b"))
        assert link.truncations[0] == (whole, True)
        assert [offset for offset, _ in link.tails[:2]] == [0, 0]
        assert phases == ["a", "b", "c", "d"], "every line once, across the truncation"

    async def is_left_alone_while_it_holds_lines_not_yet_read(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(node_module, "ROTATE_BYTES", 1)
        monkeypatch.setattr(node_module, "ROTATE_SECONDS", 0.0)
        link = _File([_phase("a"), _phase("b")], grows=True)
        node = _node(link, [])
        phases = _heard(node)

        async with asyncio.timeout(5):
            await node._watch()

        assert link.truncations == [(len(_phase("a")), False), (len(_phase("a")) + len(_phase("b")), False)]
        assert link.content == (_phase("a") + _phase("b")).encode()
        assert link.tails[0] == (0, [_phase("a"), _phase("b")]), "the tail went on past a refused truncation"
        assert phases == ["a", "b"]

    async def is_tried_at_most_once_per_rotate_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
        clock = [0.0]
        monkeypatch.setattr(node_module, "monotonic", lambda: clock[0])
        monkeypatch.setattr(node_module, "ROTATE_BYTES", 1)
        monkeypatch.setattr(node_module, "ROTATE_SECONDS", 10.0)
        link = _File([_phase(str(index)) for index in range(6)], grows=True, clock=clock)
        node = _node(link, [])
        _heard(node)

        async with asyncio.timeout(5):
            await node._watch()

        attempts = [offset for offset, _ in link.truncations]
        assert attempts == [len(_phase("0")) * 3, len(_phase("0")) * 6], "lines read at t=4..24 try at t=12 and t=24 only"


def describe_console_lines() -> None:
    async def reach_the_output_together_around_a_phase_that_keeps_its_place() -> None:
        lines = [_console("one"), _console("two"), _console("three"), _phase("apt"), _console("four"), _console("five")]
        said: list[tuple[str, ...] | str] = []
        node = _node(_File(lines), [])

        async def printed(batch: tuple[Console, ...]) -> None:
            said.append(tuple(line.content for line in batch))

        async def noted(event: str, phase: str, error: str | None) -> None:
            said.append(phase)

        node._output = printed
        node._phase = noted

        async with asyncio.timeout(5):
            await node._watch()

        assert said == [("one", "two", "three"), "apt", ("four", "five")]

    async def arrive_in_writes_of_at_most_batch(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(node_module, "BATCH", 2)
        batches: list[tuple[str, ...]] = []
        node = _node(_File([_console(str(index)) for index in range(5)]), [])

        async def printed(batch: tuple[Console, ...]) -> None:
            batches.append(tuple(line.content for line in batch))

        node._output = printed

        async with asyncio.timeout(5):
            await node._watch()

        assert batches == [("0", "1"), ("2", "3"), ("4",)]
