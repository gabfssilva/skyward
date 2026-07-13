"""Tests for the instance-monitor task (consumes transport events)."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from skyward.actors.instance_monitor import monitor_instance
from skyward.actors.messages import BootstrapDone, ConsoleOutput, NodeInstance
from skyward.api import events as domain
from skyward.infra.ssh import RawBootstrapPhase, RawConsoleOutput, RawLogEvent, RawStreamEvent
from skyward.infra.ssh_transport import StreamEvent

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _make_ni() -> NodeInstance:
    from skyward.core.model import Instance

    offer = MagicMock()
    inst = Instance(id="i-123", ip="10.0.0.1", status="provisioned", offer=offer)
    return NodeInstance(
        instance=inst,
        node=0,
        provider="aws",
        ssh_user="ubuntu",
        ssh_key_path="/tmp/key",
    )


class _FakeTransport:
    def __init__(self, raw_events: list[RawStreamEvent]) -> None:
        self._raw_events = raw_events

    async def events(self, start_line: int = 0) -> AsyncIterator[StreamEvent]:
        for i, raw in enumerate(self._raw_events, start=start_line + 1):
            yield StreamEvent(lines_read=i, event=raw)


class _Probe:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def tell(self, msg: object) -> None:
        self.messages.append(msg)


async def _run_monitor(
    raw_events: list[RawStreamEvent],
    emit: domain.Emit | None = None,
) -> tuple[_Probe, _Probe]:
    listener, reply_to = _Probe(), _Probe()
    await asyncio.wait_for(
        monitor_instance(
            _FakeTransport(raw_events),  # type: ignore[arg-type]
            _make_ni(),
            event_listener=listener,  # type: ignore[arg-type]
            reply_to=reply_to,  # type: ignore[arg-type]
            emit=emit,
            pool_name="pool",
        ),
        timeout=2.0,
    )
    return listener, reply_to


async def test_emits_log_events() -> None:
    emitted: list[domain.SessionEvent] = []
    await _run_monitor([RawLogEvent(content="hello", stream="stdout")], emit=emitted.append)

    log_events = [e for e in emitted if isinstance(e, domain.Log.Emitted)]
    assert len(log_events) == 1
    assert log_events[0].message == "hello"
    assert log_events[0].pool_name == "pool"


async def test_forwards_console_output_to_listener() -> None:
    listener, _ = await _run_monitor([RawConsoleOutput(content="progress 50%")])

    console = [m for m in listener.messages if isinstance(m, ConsoleOutput)]
    assert len(console) == 1
    assert console[0].content == "progress 50%"


async def test_signals_bootstrap_done_once() -> None:
    _, reply_to = await _run_monitor([
        RawBootstrapPhase(event="completed", phase="bootstrap"),
        RawBootstrapPhase(event="completed", phase="bootstrap"),
    ])

    done = [m for m in reply_to.messages if isinstance(m, BootstrapDone)]
    assert len(done) == 1
    assert done[0].success is True


async def test_signals_bootstrap_failure() -> None:
    _, reply_to = await _run_monitor([
        RawBootstrapPhase(event="failed", phase="uv", error="pip exploded"),
    ])

    done = [m for m in reply_to.messages if isinstance(m, BootstrapDone)]
    assert len(done) == 1
    assert done[0].success is False
    assert done[0].error == "pip exploded"


async def test_ends_when_transport_stream_ends() -> None:
    listener, reply_to = await _run_monitor([])
    assert listener.messages == []
    assert reply_to.messages == []
