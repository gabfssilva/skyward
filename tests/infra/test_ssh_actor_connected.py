"""Tests for the connected chapter — commands and file ops."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from casty import ActorSystem

from skyward.infra.ssh_actor import (
    CommandResult,
    RunCommand,
    StopTransport,
    Upload,
    UploadResult,
    WriteBytes,
    WriteFile,
    WriteResult,
    ssh_transport,
)


def _mock_conn() -> MagicMock:
    """Create a mock asyncssh connection."""
    conn = MagicMock()

    async def mock_run(cmd: str, timeout=None, check=False) -> MagicMock:
        result = MagicMock()
        result.exit_status = 0
        result.stdout = f"ran: {cmd}"
        result.stderr = ""
        return result

    conn.run = mock_run

    sftp = AsyncMock()
    file_mock = AsyncMock()
    file_mock.__aenter__ = AsyncMock(return_value=file_mock)
    file_mock.__aexit__ = AsyncMock(return_value=False)
    sftp.open = MagicMock(return_value=file_mock)
    sftp.__aenter__ = AsyncMock(return_value=sftp)
    sftp.__aexit__ = AsyncMock(return_value=False)
    conn.start_sftp_client = MagicMock(return_value=sftp)

    conn.close = MagicMock()
    conn.wait_closed = AsyncMock()

    return conn


@pytest.fixture
async def system():
    s = ActorSystem("test-connected")
    yield s
    await s.shutdown()


class TestRunCommand:
    @pytest.mark.asyncio
    async def test_run_command_returns_result(self, system: ActorSystem) -> None:
        conn = _mock_conn()

        ref = system.spawn(
            ssh_transport(host="x", user="u", key_path="k", connect_fn=lambda: _as_coro(conn)),
            "transport",
        )
        await asyncio.sleep(0.2)

        future = asyncio.get_event_loop().create_future()
        reply = MagicMock()
        reply.tell = lambda msg: future.set_result(msg) if not future.done() else None

        ref.tell(RunCommand(command=("echo", "hi"), reply_to=reply))

        result = await asyncio.wait_for(future, timeout=2.0)
        assert isinstance(result, CommandResult)
        assert result.exit_code == 0
        assert "ran: echo hi" in result.stdout


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_file_returns_success(self, system: ActorSystem) -> None:
        conn = _mock_conn()

        ref = system.spawn(
            ssh_transport(host="x", user="u", key_path="k", connect_fn=lambda: _as_coro(conn)),
            "transport",
        )
        await asyncio.sleep(0.2)

        future = asyncio.get_event_loop().create_future()
        reply = MagicMock()
        reply.tell = lambda msg: future.set_result(msg) if not future.done() else None

        ref.tell(WriteFile(remote="/tmp/test.txt", content="hello", reply_to=reply))

        result = await asyncio.wait_for(future, timeout=2.0)
        assert isinstance(result, WriteResult)
        assert result.success is True


class TestWriteBytes:
    @pytest.mark.asyncio
    async def test_write_bytes_returns_success(self, system: ActorSystem) -> None:
        conn = _mock_conn()

        ref = system.spawn(
            ssh_transport(host="x", user="u", key_path="k", connect_fn=lambda: _as_coro(conn)),
            "transport",
        )
        await asyncio.sleep(0.2)

        future = asyncio.get_event_loop().create_future()
        reply = MagicMock()
        reply.tell = lambda msg: future.set_result(msg) if not future.done() else None

        ref.tell(WriteBytes(remote="/tmp/test.bin", content=b"\x00\x01", reply_to=reply))

        result = await asyncio.wait_for(future, timeout=2.0)
        assert isinstance(result, WriteResult)
        assert result.success is True


class TestUpload:
    @pytest.mark.asyncio
    async def test_upload_returns_success(self, system: ActorSystem) -> None:
        conn = _mock_conn()

        ref = system.spawn(
            ssh_transport(host="x", user="u", key_path="k", connect_fn=lambda: _as_coro(conn)),
            "transport",
        )
        await asyncio.sleep(0.2)

        future = asyncio.get_event_loop().create_future()
        reply = MagicMock()
        reply.tell = lambda msg: future.set_result(msg) if not future.done() else None

        with patch.dict("sys.modules", {"asyncssh": MagicMock(scp=AsyncMock())}):
            ref.tell(Upload(local="/tmp/a.txt", remote="/tmp/b.txt", reply_to=reply))

            result = await asyncio.wait_for(future, timeout=2.0)
            assert isinstance(result, UploadResult)
            assert result.success is True


async def _as_coro(val: MagicMock) -> MagicMock:
    return val
