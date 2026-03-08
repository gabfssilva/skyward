"""Tests for the simplified streaming actor (subscriber pattern)."""
import asyncio
from unittest.mock import MagicMock

import pytest
from casty import ActorSystem

from skyward.actors.messages import (
    BootstrapDone,
    BootstrapPhase,
    Log,
    NodeInstance,
)
from skyward.actors.streaming.messages import StopMonitor
from skyward.actors.streaming import instance_monitor
from skyward.infra.ssh import RawBootstrapPhase, RawLogEvent
from skyward.infra.ssh_actor import StreamEvent, SubscribeEvents


def _make_ni() -> NodeInstance:
    from unittest.mock import MagicMock
    from skyward.api.model import Instance
    offer = MagicMock()
    inst = Instance(id="i-123", ip="10.0.0.1", status="provisioned", offer=offer)
    return NodeInstance(
        instance=inst,
        node=0,
        provider="aws",
        ssh_user="ubuntu",
        ssh_key_path="/tmp/key",
    )


@pytest.fixture
async def system():
    s = ActorSystem("test-streaming")
    yield s
    await s.shutdown()


class TestStreamingActorSubscriber:
    @pytest.mark.asyncio
    async def test_subscribes_to_transport_on_start(self, system: ActorSystem) -> None:
        ni = _make_ni()
        subscribe_msgs: list[SubscribeEvents] = []

        transport = MagicMock()
        transport.tell = lambda msg: subscribe_msgs.append(msg) if isinstance(msg, SubscribeEvents) else None

        ref = system.spawn(
            instance_monitor(
                info=ni,
                transport=transport,
                event_listener=MagicMock(),
                reply_to=MagicMock(),
            ),
            "monitor",
        )
        await asyncio.sleep(0.2)

        assert len(subscribe_msgs) == 1
        assert subscribe_msgs[0].start_line == 0

    @pytest.mark.asyncio
    async def test_forwards_log_events(self, system: ActorSystem) -> None:
        ni = _make_ni()
        events: list[object] = []

        transport = MagicMock()
        transport.tell = MagicMock()

        event_listener = MagicMock()
        event_listener.tell = lambda msg: events.append(msg)

        ref = system.spawn(
            instance_monitor(
                info=ni,
                transport=transport,
                event_listener=event_listener,
                reply_to=MagicMock(),
            ),
            "monitor",
        )
        await asyncio.sleep(0.2)

        ref.tell(StreamEvent(
            lines_read=1,
            event=RawLogEvent(content="hello", stream="stdout"),
        ))
        await asyncio.sleep(0.1)

        log_events = [e for e in events if isinstance(e, Log)]
        assert len(log_events) == 1
        assert log_events[0].line == "hello"

    @pytest.mark.asyncio
    async def test_signals_bootstrap_done(self, system: ActorSystem) -> None:
        ni = _make_ni()
        replies: list[BootstrapDone] = []

        transport = MagicMock()
        transport.tell = MagicMock()

        reply_to = MagicMock()
        reply_to.tell = lambda msg: replies.append(msg) if isinstance(msg, BootstrapDone) else None

        ref = system.spawn(
            instance_monitor(
                info=ni,
                transport=transport,
                event_listener=MagicMock(tell=MagicMock()),
                reply_to=reply_to,
            ),
            "monitor",
        )
        await asyncio.sleep(0.2)

        ref.tell(StreamEvent(
            lines_read=10,
            event=RawBootstrapPhase(event="completed", phase="bootstrap"),
        ))
        await asyncio.sleep(0.1)

        assert len(replies) == 1
        assert replies[0].success is True
