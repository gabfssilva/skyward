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
import time
from contextlib import suppress
from pathlib import Path

import pytest

from skyward.shared.provider import Machine
from skyward.shared import codec
from skyward.shared.schemas import Image, NodeState, Options
from skyward.worker import journal
from skyward.worker import worker as worker_module
from skyward.server.application.node import Node
from skyward.server.application.source import Source
from skyward.server.application.ssh import Result

pytestmark = pytest.mark.unit

PROBE = "nvidia-smi"


def test_the_worker_exposes_the_remote_health_loop() -> None:
    assert getattr(worker_module, "health", None) is not None


async def test_remote_health_probe_returns_the_predicate_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYWARD_NODE", "node")
    monkeypatch.setenv("SKYWARD_COMPUTE", "compute")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "127.0.0.1")

    result = await anext(worker_module.health(lambda info: info.node == "node", 0.01, 1.0, 0.0))

    assert result == (True, None)


async def test_remote_health_probe_preserves_a_failure_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYWARD_NODE", "node")
    monkeypatch.setenv("SKYWARD_COMPUTE", "compute")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "127.0.0.1")

    result = await anext(worker_module.health(lambda _: "GPU unavailable", 0.01, 1.0, 0.0))

    assert result == (False, "GPU unavailable")


async def test_remote_health_probe_reports_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYWARD_NODE", "node")
    monkeypatch.setenv("SKYWARD_COMPUTE", "compute")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "127.0.0.1")

    def fail(_: object) -> bool:
        raise RuntimeError("broken")

    healthy, reason = await anext(worker_module.health(fail, 0.01, 1.0, 0.0))

    assert not healthy
    assert reason is not None and "RuntimeError" in reason and "broken" in reason


async def test_remote_health_probe_enforces_its_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYWARD_NODE", "node")
    monkeypatch.setenv("SKYWARD_COMPUTE", "compute")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "127.0.0.1")

    def slow(_: object) -> bool:
        time.sleep(0.2)
        return True

    healthy, reason = await anext(worker_module.health(slow, 0.01, 0.01, 0.0))

    assert not healthy
    assert reason is not None and "timeout" in reason


async def test_remote_health_probe_waits_before_its_first_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYWARD_NODE", "node")
    monkeypatch.setenv("SKYWARD_COMPUTE", "compute")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "127.0.0.1")
    started = time.monotonic()

    await anext(worker_module.health(lambda _: True, 0.01, 1.0, 0.05))

    assert time.monotonic() - started >= 0.05


async def _checks(*results: tuple[bool, str | None]):
    for result in results:
        yield result


async def test_warming_waits_for_the_first_success() -> None:
    checks = _checks((False, "one"), (False, "two"), (True, None))

    await worker_module.warm(checks, consecutive_failures=3)

    with pytest.raises(StopAsyncIteration):
        await anext(checks)


async def test_warming_fails_when_the_streak_reaches_the_limit() -> None:
    checks = _checks((False, "one"), (False, "GPU unavailable"), (True, None))

    with pytest.raises(RuntimeError, match="GPU unavailable"):
        await worker_module.warm(checks, consecutive_failures=2)


async def test_steady_health_resets_the_failure_streak() -> None:
    checks = _checks((False, "one"), (True, None), (False, "two"), (False, "three"))

    reason = await worker_module.unhealthy(checks, consecutive_failures=2)

    assert "three" in reason


async def test_the_worker_loads_the_serialized_health_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "health.bin"
    path.write_bytes(codec.dumps(lambda info: info.compute == "compute"))
    monkeypatch.setenv("SKYWARD_NODE", "node")
    monkeypatch.setenv("SKYWARD_COMPUTE", "compute")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "127.0.0.1")
    monkeypatch.setenv("SKYWARD_HEALTH", str(path))
    monkeypatch.setenv("SKYWARD_HEALTH_INTERVAL", "4")
    monkeypatch.setenv("SKYWARD_HEALTH_TIMEOUT", "1.5")
    monkeypatch.setenv("SKYWARD_HEALTH_FAILURES", "5")
    monkeypatch.setenv("SKYWARD_HEALTH_INITIAL_DELAY", "0")

    configured = worker_module.health_checks()

    assert configured is not None
    checks, failures = configured
    assert failures == 5
    assert await anext(checks) == (True, None)


async def test_health_monitor_starts_only_after_the_warming_success() -> None:
    configured = (
        _checks((False, "warming"), (True, None), (False, "one"), (False, "terminal")),
        2,
    )

    monitor = await worker_module.start_health(configured)

    assert monitor is not None
    assert "terminal" in await monitor


def test_a_terminal_remote_health_event_marks_the_node_lost() -> None:
    node, seen = _node(FakeSsh(), command=None)

    node._observe(journal.Health(reason="health check failed 3 times: GPU unavailable"))

    assert seen == [("lost", 0)]


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
        source=Source(arguments=("skyward",)),
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
