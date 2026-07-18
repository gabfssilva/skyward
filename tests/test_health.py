"""The health probe, driven by a scripted machine — no node, no cluster.

The probe is the one thing in the node that decides something, and what it decides
is that a machine is gone. So what is pinned here is the arithmetic: how many
refusals in a row it takes, that it is not one fewer, and that a single good answer
puts the count back to zero.

The listener records how many probes had run when it was called, because "after
exactly three" and "not before" are the same assertion written once.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from skyward.application.provider import Machine
from skyward.protocol.schemas import Image, NodeState, Options
from skyward.runtime.node import Node
from skyward.runtime.source import Source
from skyward.runtime.ssh import Result

pytestmark = pytest.mark.unit

PROBE = "nvidia-smi"


class Exhausted(RuntimeError):
    """The probe asked more times than the test scripted an answer for."""


class FakeSsh:
    """A machine whose answers are written down in advance."""

    def __init__(self, *codes: int) -> None:
        self.runs: list[str] = []
        self._codes = list(codes)

    async def run(self, command: str, *, timeout: float | None = None) -> Result:
        if not self._codes:
            raise Exhausted(command)
        code = self._codes.pop(0)
        self.runs.append(command)
        return Result(exit_code=code, stdout="", stderr="unhealthy" if code else "")


def _node(ssh: FakeSsh, *, command: str | None = PROBE, failures: int = 3) -> tuple[Node, list[tuple[NodeState, int]]]:
    seen: list[tuple[NodeState, int]] = []
    node = Node(
        Machine(id="mch", state="running", host="127.0.0.1", user="root"),
        compute="cmp",
        private_key="",
        image=Image(),
        source=Source(argument="skyward"),
        listener=lambda state, _: seen.append((state, len(ssh.runs))),
        output=lambda *_: None,
        sample=lambda *_: None,
        phase=lambda *_: None,
        options=Options(health_command=command, health_interval=0.0, health_failures=failures),
    )
    node._ssh = ssh  # type: ignore[assignment]
    return node, seen


async def test_the_node_is_lost_on_the_failure_that_completes_the_streak():
    ssh = FakeSsh(1, 1, 1)
    node, seen = _node(ssh)

    await node._health(PROBE)

    assert seen == [("lost", 3)], "not on the second, and not once more after the third"


async def test_a_shorter_streak_is_not_enough():
    ssh = FakeSsh(1, 1)
    node, seen = _node(ssh)

    with pytest.raises(Exhausted):
        await node._health(PROBE)

    assert seen == [], "two of the three refusals is a machine still worth having"


async def test_one_good_answer_puts_the_count_back_to_zero():
    ssh = FakeSsh(1, 1, 0, 1, 1, 1)
    node, seen = _node(ssh)

    await node._health(PROBE)

    assert seen == [("lost", 6)], "the streak restarts at the success, so it takes three more"


async def test_the_streak_is_as_long_as_the_options_asked_for():
    node, seen = _node(FakeSsh(1), failures=1)

    await node._health(PROBE)

    assert seen == [("lost", 1)]


async def test_the_probe_starts_when_the_node_says_it_is_ready():
    node, seen = _node(FakeSsh())

    node._ready()

    assert seen == [("ready", 0)]
    assert node._probe is not None
    node._probe.cancel()
    with suppress(asyncio.CancelledError):
        await node._probe


def test_a_node_asked_for_no_command_probes_nothing():
    """The default: a compute that never heard of health checks behaves as it always did."""
    node, seen = _node(FakeSsh(), command=None)

    node._ready()

    assert seen == [("ready", 0)] and node._probe is None
