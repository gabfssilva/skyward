from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from casty import ActorRef

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeSlots,
    NodeTarget,
    PressureReport,
    SubmitTask,
    TaskSubmitted,
)

type NodeId = int


@dataclass(frozen=True, slots=True)
class PendingBroadcast:
    caller: ActorRef
    pending: frozenset[NodeId]
    results: MappingProxyType[NodeId, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class _State:
    nodes: MappingProxyType[NodeId, NodeSlots]
    queue: tuple[SubmitTask, ...]
    round_robin: int
    inflight: MappingProxyType[str, SubmitTask]
    broadcasts: MappingProxyType[str, PendingBroadcast]
    node_tasks: MappingProxyType[NodeId, frozenset[str]] = field(default_factory=lambda: MappingProxyType({}))
    retry_on_interruption: int = 3
    retries: MappingProxyType[str, int] = field(default_factory=lambda: MappingProxyType({}))
    pressure_observer: ActorRef | None = None


def _pick_with_free_slot(
    nodes: MappingProxyType[NodeId, NodeSlots],
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


def _target_rank(target: NodeTarget) -> NodeId:
    """Resolve a ``NodeTarget`` to a concrete rank (``"head"`` → 0)."""
    return 0 if target == "head" else int(target)


def _pick_target(
    nodes: MappingProxyType[NodeId, NodeSlots],
    target: NodeTarget,
) -> NodeId | None:
    """Return the resolved rank if it is present, else ``None`` (absent rank)."""
    rank = _target_rank(target)
    return rank if rank in nodes else None


def _dispatch(
    nid: NodeId,
    task: SubmitTask,
    nodes: MappingProxyType[NodeId, NodeSlots],
    tm_ref: ActorRef,
    inflight: MappingProxyType[str, SubmitTask],
    node_tasks: MappingProxyType[NodeId, frozenset[str]],
) -> tuple[MappingProxyType[NodeId, NodeSlots], MappingProxyType[str, SubmitTask], MappingProxyType[NodeId, frozenset[str]]]:
    slot = nodes[nid]
    tm_ref.tell(TaskSubmitted(task_id=task.task_id, node_id=nid))
    slot.ref.tell(ExecuteOnNode(
        fn=task.fn, args=task.args, kwargs=task.kwargs,
        reply_to=tm_ref, task_id=task.task_id, timeout=task.timeout,
    ))
    new_nodes = MappingProxyType({**nodes, nid: NodeSlots(slot.ref, slot.total, slot.used + 1)})
    new_inflight = MappingProxyType({**inflight, task.task_id: task})
    existing = node_tasks.get(nid, frozenset())
    new_node_tasks = MappingProxyType({**node_tasks, nid: existing | {task.task_id}})
    return new_nodes, new_inflight, new_node_tasks


def _drain_queue(
    queue: tuple[SubmitTask, ...],
    nodes: MappingProxyType[NodeId, NodeSlots],
    round_robin: int,
    tm_ref: ActorRef,
    inflight: MappingProxyType[str, SubmitTask],
    node_tasks: MappingProxyType[NodeId, frozenset[str]],
) -> tuple[tuple[SubmitTask, ...], MappingProxyType[NodeId, NodeSlots], int, MappingProxyType[str, SubmitTask], MappingProxyType[NodeId, frozenset[str]]]:
    remaining: list[SubmitTask] = []
    for task in queue:
        if task.target is not None:
            rank = _target_rank(task.target)
            slot = nodes.get(rank)
            if slot is not None and slot.used < slot.total:
                nodes, inflight, node_tasks = _dispatch(rank, task, nodes, tm_ref, inflight, node_tasks)
            else:
                remaining.append(task)
            continue
        nid = _pick_with_free_slot(nodes, round_robin)
        if nid is None:
            remaining.append(task)
            continue
        nodes, inflight, node_tasks = _dispatch(nid, task, nodes, tm_ref, inflight, node_tasks)
        round_robin += 1
    return tuple(remaining), nodes, round_robin, inflight, node_tasks


def _emit_pressure(s: _State) -> None:
    if s.pressure_observer is None:
        return
    s.pressure_observer.tell(PressureReport(
        queued=len(s.queue),
        inflight=sum(slot.used for slot in s.nodes.values()),
        total_capacity=sum(slot.total for slot in s.nodes.values()),
        node_count=len(s.nodes),
    ))
