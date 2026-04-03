import pytest

from skyward.daemon.protocol import (
    DaemonRequest,
    DaemonResponse,
    EnsurePool,
    PoolReady,
    PoolFailed,
    SubmitTask,
    SubmitBroadcast,
    TaskSucceeded,
    TaskFailed,
    BroadcastSucceeded,
    GetNodeCount,
    NodeCount,
    Disconnect,
    Disconnected,
    ShutdownPool,
    PoolShutdown,
    Ping,
    Pong,
)


class TestProtocolMessages:
    def test_ensure_pool_is_frozen(self) -> None:
        msg = EnsurePool(name="train", spec_bytes=b"data")
        assert msg.name == "train"
        with pytest.raises(AttributeError):
            msg.name = "other"  # type: ignore[misc]

    def test_submit_task_carries_client_id(self) -> None:
        msg = SubmitTask(pool_name="train", payload=b"fn", timeout=300.0, client_id="abc123")
        assert msg.client_id == "abc123"

    def test_submit_task_client_id_defaults_empty(self) -> None:
        msg = SubmitTask(pool_name="train", payload=b"fn", timeout=300.0)
        assert msg.client_id == ""

    def test_submit_broadcast_carries_client_id(self) -> None:
        msg = SubmitBroadcast(pool_name="train", payload=b"fn", timeout=60.0, client_id="x")
        assert msg.client_id == "x"

    def test_task_succeeded_carries_result_bytes(self) -> None:
        msg = TaskSucceeded(payload=b"result")
        assert isinstance(msg.payload, bytes)

    def test_task_failed_carries_error(self) -> None:
        msg = TaskFailed(error="ZeroDivisionError", traceback="...")
        assert msg.error == "ZeroDivisionError"

    def test_request_response_unions(self) -> None:
        req: DaemonRequest = Ping()
        resp: DaemonResponse = Pong()
        assert isinstance(req, Ping)
        assert isinstance(resp, Pong)
