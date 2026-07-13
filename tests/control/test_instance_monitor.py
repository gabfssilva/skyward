"""Tests for the instance-monitor task (consumes transport events)."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from skyward.control.instance_monitor import monitor_instance
from skyward.api.facts import ConsoleOutput, NodeInstance
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


async def _run_monitor(
    raw_events: list[RawStreamEvent],
    emit: domain.Emit | None = None,
) -> tuple[list[object], list[tuple[bool, str | None]]]:
    stream: list[object] = []
    done: list[tuple[bool, str | None]] = []
    await asyncio.wait_for(
        monitor_instance(
            _FakeTransport(raw_events),  # type: ignore[arg-type]
            _make_ni(),
            on_stream=stream.append,
            on_bootstrap_done=lambda ok, err: done.append((ok, err)),
            emit=emit,
            pool_name="pool",
        ),
        timeout=2.0,
    )
    return stream, done


async def test_emits_log_events() -> None:
    emitted: list[domain.SessionEvent] = []
    await _run_monitor([RawLogEvent(content="hello", stream="stdout")], emit=emitted.append)

    log_events = [e for e in emitted if isinstance(e, domain.Log.Emitted)]
    assert len(log_events) == 1
    assert log_events[0].message == "hello"
    assert log_events[0].pool_name == "pool"


async def test_forwards_console_output_to_stream() -> None:
    stream, _ = await _run_monitor([RawConsoleOutput(content="progress 50%")])

    console = [m for m in stream if isinstance(m, ConsoleOutput)]
    assert len(console) == 1
    assert console[0].content == "progress 50%"


async def test_signals_bootstrap_done_once() -> None:
    _, done = await _run_monitor([
        RawBootstrapPhase(event="completed", phase="bootstrap"),
        RawBootstrapPhase(event="completed", phase="bootstrap"),
    ])

    assert done == [(True, None)]


async def test_signals_bootstrap_failure() -> None:
    _, done = await _run_monitor([
        RawBootstrapPhase(event="failed", phase="uv", error="pip exploded"),
    ])

    assert done == [(False, "pip exploded")]


async def test_ends_when_transport_stream_ends() -> None:
    stream, done = await _run_monitor([])
    assert stream == []
    assert done == []
