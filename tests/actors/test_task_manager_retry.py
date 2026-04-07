"""Task manager retry logic for interrupted tasks."""

from types import MappingProxyType
from unittest.mock import MagicMock

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeSlots,
    SubmitTask,
    TaskSubmitted,
)
from skyward.actors.task_manager.state import (
    _dispatch,
)


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

        new_nodes, new_inflight, new_nt = _dispatch(1, task, nodes, tm_ref, inflight, MappingProxyType({}))

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

        new_nodes, _, _ = _dispatch(1, task, nodes, tm_ref, inflight, MappingProxyType({}))

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

        _dispatch(1, task, nodes, tm_ref, inflight, MappingProxyType({}))

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

        _dispatch(1, task, nodes, tm_ref, inflight, MappingProxyType({}))

        tm_ref.tell.assert_called_once()
        msg = tm_ref.tell.call_args[0][0]
        assert isinstance(msg, TaskSubmitted)
        assert msg.task_id == "t1"
        assert msg.node_id == 1


