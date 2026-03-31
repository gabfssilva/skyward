"""Task manager retry logic for interrupted tasks."""

from types import MappingProxyType
from unittest.mock import MagicMock

import pytest

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeSlots,
    SubmitTask,
    TaskFailed,
    TaskInterrupted,
    TaskSucceeded,
    TaskSubmitted,
)
from skyward.actors.task_manager.state import (
    _dispatch,
    _State,
)


class TestStateRetryFields:
    def test_state_has_retries_field(self):
        s = _State(
            nodes=MappingProxyType({}),
            queue=(),
            round_robin=0,
            inflight=MappingProxyType({}),
            broadcasts=MappingProxyType({}),
        )
        assert s.retries == MappingProxyType({})

    def test_state_has_retry_on_interruption_field(self):
        s = _State(
            nodes=MappingProxyType({}),
            queue=(),
            round_robin=0,
            inflight=MappingProxyType({}),
            broadcasts=MappingProxyType({}),
            retry_on_interruption=5,
        )
        assert s.retry_on_interruption == 5

    def test_state_retry_on_interruption_defaults_to_3(self):
        s = _State(
            nodes=MappingProxyType({}),
            queue=(),
            round_robin=0,
            inflight=MappingProxyType({}),
            broadcasts=MappingProxyType({}),
        )
        assert s.retry_on_interruption == 3


class TestDispatchStoresFullTask:
    def test_inflight_contains_submit_task(self):
        node_ref = MagicMock()
        tm_ref = MagicMock()
        caller = MagicMock()
        task = SubmitTask(fn=b"payload", args=(), kwargs={}, reply_to=caller, task_id="t1")
        nodes = MappingProxyType({
            1: NodeSlots(ref=node_ref, total=2, used=0),
        })
        inflight: MappingProxyType[str, SubmitTask] = MappingProxyType({})

        new_nodes, new_inflight = _dispatch(1, task, nodes, tm_ref, inflight)

        assert "t1" in new_inflight
        assert isinstance(new_inflight["t1"], SubmitTask)
        assert new_inflight["t1"].reply_to is caller
        assert new_inflight["t1"].fn == b"payload"

    def test_dispatch_increments_slot_used(self):
        node_ref = MagicMock()
        tm_ref = MagicMock()
        caller = MagicMock()
        task = SubmitTask(fn=b"payload", args=(), kwargs={}, reply_to=caller, task_id="t1")
        nodes = MappingProxyType({
            1: NodeSlots(ref=node_ref, total=2, used=0),
        })
        inflight: MappingProxyType[str, SubmitTask] = MappingProxyType({})

        new_nodes, _ = _dispatch(1, task, nodes, tm_ref, inflight)

        assert new_nodes[1].used == 1

    def test_dispatch_sends_execute_on_node(self):
        node_ref = MagicMock()
        tm_ref = MagicMock()
        caller = MagicMock()
        task = SubmitTask(fn=b"payload", args=(1,), kwargs={"x": 2}, reply_to=caller, task_id="t1", timeout=120.0)
        nodes = MappingProxyType({
            1: NodeSlots(ref=node_ref, total=2, used=0),
        })
        inflight: MappingProxyType[str, SubmitTask] = MappingProxyType({})

        _dispatch(1, task, nodes, tm_ref, inflight)

        node_ref.tell.assert_called_once()
        msg = node_ref.tell.call_args[0][0]
        assert isinstance(msg, ExecuteOnNode)
        assert msg.fn == b"payload"
        assert msg.args == (1,)
        assert msg.kwargs == {"x": 2}
        assert msg.task_id == "t1"
        assert msg.timeout == 120.0
        assert msg.reply_to is tm_ref

    def test_dispatch_sends_task_submitted(self):
        node_ref = MagicMock()
        tm_ref = MagicMock()
        caller = MagicMock()
        task = SubmitTask(fn=b"payload", args=(), kwargs={}, reply_to=caller, task_id="t1")
        nodes = MappingProxyType({
            1: NodeSlots(ref=node_ref, total=2, used=0),
        })
        inflight: MappingProxyType[str, SubmitTask] = MappingProxyType({})

        _dispatch(1, task, nodes, tm_ref, inflight)

        tm_ref.tell.assert_called_once()
        msg = tm_ref.tell.call_args[0][0]
        assert isinstance(msg, TaskSubmitted)
        assert msg.task_id == "t1"
        assert msg.node_id == 1


class TestPatternMatchingOnSubtypes:
    def test_task_succeeded_matches(self):
        msg = TaskSucceeded(value=42, node_id=1, task_id="t1")
        match msg:
            case TaskSucceeded(value=v, node_id=nid, task_id=tid):
                assert v == 42
                assert nid == 1
                assert tid == "t1"
            case _:
                pytest.fail("TaskSucceeded should match")

    def test_task_failed_matches(self):
        err = ValueError("bad input")
        msg = TaskFailed(error=err, node_id=2, task_id="t2")
        match msg:
            case TaskFailed(error=e, node_id=nid, task_id=tid):
                assert e is err
                assert nid == 2
                assert tid == "t2"
            case _:
                pytest.fail("TaskFailed should match")

    def test_task_interrupted_matches(self):
        err = RuntimeError("node lost")
        msg = TaskInterrupted(error=err, node_id=3, task_id="t3")
        match msg:
            case TaskInterrupted(error=e, node_id=nid, task_id=tid):
                assert e is err
                assert nid == 3
                assert tid == "t3"
            case _:
                pytest.fail("TaskInterrupted should match")

    def test_subtypes_are_distinct_in_match(self):
        results = [
            TaskSucceeded(value="ok", node_id=1, task_id="t1"),
            TaskFailed(error=ValueError("x"), node_id=1, task_id="t2"),
            TaskInterrupted(error=RuntimeError("lost"), node_id=1, task_id="t3"),
        ]
        kinds = []
        for r in results:
            match r:
                case TaskSucceeded():
                    kinds.append("success")
                case TaskFailed():
                    kinds.append("user_error")
                case TaskInterrupted():
                    kinds.append("infra_error")
        assert kinds == ["success", "user_error", "infra_error"]
