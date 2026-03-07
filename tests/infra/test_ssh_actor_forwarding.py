"""Tests for port forwarding and event streaming subscription."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from casty import ActorSystem

from skyward.infra.ssh_actor import (
    ForwardPort,
    PortForwarded,
    StreamEvent,
    SubscribeEvents,
    ssh_transport,
)


def _mock_conn_with_forward(local_port: int = 12345) -> MagicMock:
    conn = MagicMock()

    async def mock_run(cmd: str, timeout=None, check=False) -> MagicMock:
        result = MagicMock()
        result.exit_status = 0
        result.stdout = ""
        result.stderr = ""
        return result

    conn.run = mock_run

    listener = MagicMock()
    listener.get_port = MagicMock(return_value=local_port)
    listener.close = MagicMock()

    async def mock_forward(bind_host, bind_port, dest_host, dest_port):
        return listener

    conn.forward_local_port = mock_forward
    conn.close = MagicMock()
    conn.wait_closed = AsyncMock()

    return conn


@pytest.fixture
async def system():
    s = ActorSystem("test-forward")
    yield s
    await s.shutdown()


class TestForwardPort:
    @pytest.mark.asyncio
    async def test_forward_port_returns_local_port(self, system: ActorSystem) -> None:
        conn = _mock_conn_with_forward(local_port=54321)

        ref = system.spawn(
            ssh_transport(host="x", user="u", key_path="k", connect_fn=lambda: _as_coro(conn)),
            "transport",
        )
        await asyncio.sleep(0.2)

        future = asyncio.get_event_loop().create_future()
        reply = MagicMock()
        reply.tell = lambda msg: future.set_result(msg) if not future.done() else None

        ref.tell(ForwardPort(remote_host="127.0.0.1", remote_port=25520, reply_to=reply))

        result = await asyncio.wait_for(future, timeout=2.0)
        assert isinstance(result, PortForwarded)
        assert result.local_port == 54321


class TestSubscribeEvents:
    @pytest.mark.asyncio
    async def test_subscriber_receives_events(self, system: ActorSystem) -> None:
        conn = _mock_conn_with_forward()
        events_received: list[StreamEvent] = []

        log_line = json.dumps({"type": "log", "content": "hello from worker"})

        # Mock conn.run for file existence check
        async def mock_run(cmd: str, timeout=None, check=False) -> MagicMock:
            result = MagicMock()
            result.exit_status = 0
            result.stdout = ""
            result.stderr = ""
            return result

        conn.run = mock_run

        # Mock create_process for tail -F
        proc = MagicMock()
        stdout_reader = AsyncMock()

        read_count = 0

        async def mock_read(n: int) -> str:
            nonlocal read_count
            read_count += 1
            if read_count == 1:
                return log_line + "\n"
            await asyncio.sleep(10)
            return ""

        stdout_reader.read = mock_read
        proc.stdout = stdout_reader
        proc.__aenter__ = AsyncMock(return_value=proc)
        proc.__aexit__ = AsyncMock(return_value=False)
        conn.create_process = MagicMock(return_value=proc)

        ref = system.spawn(
            ssh_transport(host="x", user="u", key_path="k", connect_fn=lambda: _as_coro(conn)),
            "transport",
        )
        await asyncio.sleep(0.2)

        subscriber = MagicMock()
        subscriber.tell = lambda msg: events_received.append(msg)

        ref.tell(SubscribeEvents(start_line=0, subscriber=subscriber))
        await asyncio.sleep(1.0)

        assert len(events_received) >= 1
        assert isinstance(events_received[0], StreamEvent)
        assert events_received[0].event.content == "hello from worker"  # type: ignore[union-attr]


async def _as_coro(val: MagicMock) -> MagicMock:
    return val
