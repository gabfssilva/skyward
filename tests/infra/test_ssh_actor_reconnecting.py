"""Tests for the reconnecting chapter."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from casty import ActorSystem

from skyward.infra.ssh_actor import (
    CommandResult,
    ConnectionFailed,
    ConnectionLost,
    ConnectionRestored,
    ForwardPort,
    PortForwarded,
    PortReForwarded,
    RunCommand,
    ssh_transport,
)


@pytest.fixture
async def system():
    s = ActorSystem("test-reconnect")
    yield s
    await s.shutdown()


class TestReconnecting:
    @pytest.mark.asyncio
    async def test_reconnects_after_command_failure(self, system: ActorSystem) -> None:
        """Connection error on command triggers reconnect and notifies parent."""
        call_count = 0
        parent_msgs: list[object] = []

        def make_conn(fail_run: bool = False) -> MagicMock:
            conn = MagicMock()

            async def mock_run(cmd: str, timeout=None, check=False) -> MagicMock:
                if fail_run:
                    raise OSError("connection reset")
                result = MagicMock()
                result.exit_status = 0
                result.stdout = "ok"
                result.stderr = ""
                return result

            conn.run = mock_run
            conn.close = MagicMock()
            conn.wait_closed = AsyncMock()
            return conn

        conn1 = make_conn(fail_run=True)
        conn2 = make_conn(fail_run=False)

        async def connect() -> MagicMock:
            nonlocal call_count
            call_count += 1
            return conn1 if call_count == 1 else conn2

        parent = MagicMock()
        parent.tell = lambda msg: parent_msgs.append(msg)

        ref = system.spawn(
            ssh_transport(
                host="x", user="u", key_path="k",
                connect_fn=connect,
                parent=parent,
                retry_delay=0.05,
            ),
            "transport",
        )
        await asyncio.sleep(0.2)

        # Send command that will fail (triggers reconnect)
        future = asyncio.get_event_loop().create_future()
        reply = MagicMock()
        reply.tell = lambda msg: future.set_result(msg) if not future.done() else None
        ref.tell(RunCommand(command=("test",), reply_to=reply))

        # Wait for error result + reconnection
        result = await asyncio.wait_for(future, timeout=2.0)
        assert isinstance(result, CommandResult)
        assert result.exit_code == -1

        await asyncio.sleep(0.5)

        # Verify parent got ConnectionLost + ConnectionRestored
        lost = [m for m in parent_msgs if isinstance(m, ConnectionLost)]
        restored = [m for m in parent_msgs if isinstance(m, ConnectionRestored)]
        assert len(lost) >= 1
        assert len(restored) >= 1

    @pytest.mark.asyncio
    async def test_permanent_reconnect_failure_notifies_parent(self, system: ActorSystem) -> None:
        """If reconnection exhausts retries, parent gets ConnectionFailed."""
        call_count = 0
        parent_msgs: list[object] = []

        def make_conn(fail_run: bool = False) -> MagicMock:
            conn = MagicMock()

            async def mock_run(cmd: str, timeout=None, check=False) -> MagicMock:
                if fail_run:
                    raise OSError("connection reset")
                result = MagicMock()
                result.exit_status = 0
                result.stdout = "ok"
                result.stderr = ""
                return result

            conn.run = mock_run
            conn.close = MagicMock()
            conn.wait_closed = AsyncMock()
            return conn

        conn1 = make_conn(fail_run=True)

        async def connect() -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return conn1
            raise ConnectionError("permanently unreachable")

        parent = MagicMock()
        parent.tell = lambda msg: parent_msgs.append(msg)

        ref = system.spawn(
            ssh_transport(
                host="x", user="u", key_path="k",
                connect_fn=connect,
                parent=parent,
                retry_delay=0.05,
                retry_max_attempts=2,
            ),
            "transport",
        )
        await asyncio.sleep(0.2)

        future = asyncio.get_event_loop().create_future()
        reply = MagicMock()
        reply.tell = lambda msg: future.set_result(msg) if not future.done() else None
        ref.tell(RunCommand(command=("test",), reply_to=reply))

        await asyncio.wait_for(future, timeout=2.0)
        await asyncio.sleep(1.0)

        failed = [m for m in parent_msgs if isinstance(m, ConnectionFailed)]
        assert len(failed) >= 1
