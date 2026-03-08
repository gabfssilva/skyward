from dataclasses import dataclass, field
from typing import Any

from casty import ActorRef

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeSlots,
    PressureReport,
    SubmitTask,
    TaskSubmitted,
)

type NodeId = int


@dataclass(slots=True)
class PendingBroadcast:
    caller: ActorRef
    pending: set[NodeId]
    results: dict[NodeId, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _State:
    nodes: dict[NodeId, NodeSlots]
    queue: tuple[SubmitTask, ...]
    round_robin: int
    inflight: dict[str, ActorRef]
    broadcasts: dict[str, PendingBroadcast]
    pressure_observer: ActorRef | None = None


def _pick_with_free_slot(
    nodes: dict[NodeId, NodeSlots],
    round_robin: int,
) -> NodeId | None:
    node_ids = sorted(nodes.keys())
    if not node_ids:
        return None
    for i in range(len(node_ids)):
        nid = node_ids[(round_robin + i) % len(node_ids)]
        slot = nodes[nid]
        if slot.used < slot.total:
            return nid
    return None


def _dispatch(
    nid: NodeId,
    task_id: str,
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    reply_to: ActorRef,
    nodes: dict[NodeId, NodeSlots],
    tm_ref: ActorRef,
    inflight: dict[str, ActorRef],
    timeout: float = 600.0,
) -> dict[NodeId, NodeSlots]:
    slot = nodes[nid]
    tm_ref.tell(TaskSubmitted(task_id=task_id, node_id=nid))
    slot.ref.tell(ExecuteOnNode(
        fn=fn, args=args, kwargs=kwargs,
        reply_to=tm_ref, task_id=task_id, timeout=timeout,
    ))
    inflight[task_id] = reply_to
    return {**nodes, nid: NodeSlots(slot.ref, slot.total, slot.used + 1)}


def _drain_queue(
    queue: tuple[SubmitTask, ...],
    nodes: dict[NodeId, NodeSlots],
    round_robin: int,
    tm_ref: ActorRef,
    inflight: dict[str, ActorRef],
) -> tuple[tuple[SubmitTask, ...], dict[NodeId, NodeSlots], int]:
    remaining: list[SubmitTask] = []
    for task in queue:
        nid = _pick_with_free_slot(nodes, round_robin)
        if nid is None:
            remaining.append(task)
            continue
        nodes = _dispatch(
            nid, task.task_id, task.fn, task.args, task.kwargs,
            task.reply_to, nodes, tm_ref, inflight, timeout=task.timeout,
        )
        round_robin += 1
    return tuple(remaining), nodes, round_robin


def _emit_pressure(s: _State) -> None:
    if s.pressure_observer is None:
        return
    s.pressure_observer.tell(PressureReport(
        queued=len(s.queue),
        inflight=sum(slot.used for slot in s.nodes.values()),
        total_capacity=sum(slot.total for slot in s.nodes.values()),
        node_count=len(s.nodes),
    ))
