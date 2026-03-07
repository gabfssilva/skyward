"""Tests for the connecting chapter of the ssh_transport actor."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from casty import ActorSystem

from skyward.infra.ssh_actor import (
    ConnectionFailed,
    RunCommand,
    CommandResult,
    StopTransport,
    ssh_transport,
)


@pytest.fixture
async def system():
    s = ActorSystem("test-transport")
    yield s
    await s.shutdown()


class TestConnecting:
    @pytest.mark.asyncio
    async def test_connects_successfully(self, system: ActorSystem) -> None:
        mock_conn = MagicMock()
        connect_calls = 0

        async def fake_connect() -> MagicMock:
            nonlocal connect_calls
            connect_calls += 1
            return mock_conn

        ref = system.spawn(
            ssh_transport(
                host="10.0.0.1",
                user="ubuntu",
                key_path="/tmp/key",
                connect_fn=fake_connect,
            ),
            "transport",
        )

        await asyncio.sleep(0.2)
        assert connect_calls == 1

        ref.tell(StopTransport())
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_retries_on_connect_failure(self, system: ActorSystem) -> None:
        attempt = 0
        mock_conn = MagicMock()

        async def fail_then_succeed() -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ConnectionError(f"attempt {attempt}")
            return mock_conn

        ref = system.spawn(
            ssh_transport(
                host="10.0.0.1",
                user="ubuntu",
                key_path="/tmp/key",
                retry_delay=0.05,
                connect_fn=fail_then_succeed,
            ),
            "transport",
        )

        await asyncio.sleep(0.5)
        assert attempt == 3

    @pytest.mark.asyncio
    async def test_permanent_failure_notifies_parent(self, system: ActorSystem) -> None:
        received: list[object] = []

        async def always_fail() -> MagicMock:
            raise ConnectionError("nope")

        parent_ref = MagicMock()
        parent_ref.tell = lambda msg: received.append(msg)

        ref = system.spawn(
            ssh_transport(
                host="10.0.0.1",
                user="ubuntu",
                key_path="/tmp/key",
                retry_max_attempts=2,
                retry_delay=0.05,
                connect_fn=always_fail,
                parent=parent_ref,
            ),
            "transport",
        )

        await asyncio.sleep(0.5)
        failures = [m for m in received if isinstance(m, ConnectionFailed)]
        assert len(failures) == 1
        assert "nope" in failures[0].error

    @pytest.mark.asyncio
    async def test_requests_queued_during_connecting(self, system: ActorSystem) -> None:
        mock_conn = MagicMock()
        connect_event = asyncio.Event()

        async def delayed_connect() -> MagicMock:
            await connect_event.wait()
            return mock_conn

        async def mock_run(cmd: str, timeout: float | None = None, check: bool = False) -> MagicMock:
            result = MagicMock()
            result.exit_status = 0
            result.stdout = "queued-result"
            result.stderr = ""
            return result

        mock_conn.run = mock_run

        ref = system.spawn(
            ssh_transport(
                host="10.0.0.1",
                user="ubuntu",
                key_path="/tmp/key",
                connect_fn=delayed_connect,
            ),
            "transport",
        )

        result_future = asyncio.get_event_loop().create_future()

        reply_mock = MagicMock()
        reply_mock.tell = lambda msg: result_future.set_result(msg) if not result_future.done() else None

        ref.tell(RunCommand(command=("echo", "hi"), reply_to=reply_mock))

        connect_event.set()

        cmd_result = await asyncio.wait_for(result_future, timeout=2.0)
        assert isinstance(cmd_result, CommandResult)
        assert cmd_result.stdout == "queued-result"
