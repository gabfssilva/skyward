"""Task manager dispatch and retry logic with opaque-payload SubmitTask."""

from types import MappingProxyType
from unittest.mock import MagicMock

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeSlots,
    SubmitTask,
    TaskSubmitted,
)
from skyward.actors.task_manager.state import _dispatch


def _make_task(task_id: str = "t1", timeout: float = 60.0) -> SubmitTask:
    return SubmitTask(
        payload=b"payload",
        reply_to=MagicMock(),
        task_id=task_id,
        timeout=timeout,
        task_key=("mod", "fn"),
    )


class TestDispatchStoresFullTask:
    def test_inflight_contains_submit_task(self):
        node_ref = MagicMock()
        tm_ref = MagicMock()
        task = _make_task()
        nodes = MappingProxyType({
            1: NodeSlots(ref=node_ref, total=2, used=0),
        })
        inflight: MappingProxyType[str, SubmitTask] = MappingProxyType({})

        _new_nodes, new_inflight, _ = _dispatch(1, task, nodes, tm_ref, inflight, MappingProxyType({}))

        assert "t1" in new_inflight
        assert isinstance(new_inflight["t1"], SubmitTask)
        assert new_inflight["t1"].payload == b"payload"

    def test_dispatch_increments_slot_used(self):
        node_ref = MagicMock()
        tm_ref = MagicMock()
        task = _make_task()
        nodes = MappingProxyType({
            1: NodeSlots(ref=node_ref, total=2, used=0),
        })
        inflight: MappingProxyType[str, SubmitTask] = MappingProxyType({})

        new_nodes, _, _ = _dispatch(1, task, nodes, tm_ref, inflight, MappingProxyType({}))

        assert new_nodes[1].used == 1

    def test_dispatch_sends_execute_on_node_with_payload(self):
        node_ref = MagicMock()
        tm_ref = MagicMock()
        task = _make_task(task_id="t1", timeout=120.0)
        nodes = MappingProxyType({
            1: NodeSlots(ref=node_ref, total=2, used=0),
        })
        inflight: MappingProxyType[str, SubmitTask] = MappingProxyType({})

        _dispatch(1, task, nodes, tm_ref, inflight, MappingProxyType({}))

        node_ref.tell.assert_called_once()
        msg = node_ref.tell.call_args[0][0]
        assert isinstance(msg, ExecuteOnNode)
        assert msg.payload == b"payload"
        assert msg.task_id == "t1"
        assert msg.timeout == 120.0
        assert msg.reply_to is tm_ref

    def test_dispatch_sends_task_submitted(self):
        node_ref = MagicMock()
        tm_ref = MagicMock()
        task = _make_task()
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
