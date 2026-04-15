from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from casty import ActorRef

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeSlots,
    PressureReport,
    SubmitTask,
    TaskSubmitted,
)

if TYPE_CHECKING:
    from skyward.server.host.store import Store

type NodeId = int


@dataclass(frozen=True, slots=True)
class PendingBroadcast:
    caller: ActorRef
    pending: frozenset[NodeId]
    results: MappingProxyType[NodeId, object] = field(
        default_factory=lambda: MappingProxyType({}),
    )


@dataclass(frozen=True, slots=True)
class _State:
    nodes: MappingProxyType[NodeId, NodeSlots]
    queue: tuple[SubmitTask, ...]
    round_robin: int
    inflight: MappingProxyType[str, SubmitTask]
    broadcasts: MappingProxyType[str, PendingBroadcast]
    node_tasks: MappingProxyType[NodeId, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    retry_on_interruption: int = 3
    retries: MappingProxyType[str, int] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    pressure_observer: ActorRef | None = None
    store: "Store | None" = None


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


def _to_node_message(task: SubmitTask, tm_ref: ActorRef) -> ExecuteOnNode:
    return ExecuteOnNode(
        payload=task.payload,
        reply_to=tm_ref,
        task_id=task.task_id,
        timeout=task.timeout,
        input_streams=task.input_streams,
        is_generator=task.is_generator,
    )


def _dispatch(
    nid: NodeId,
    task: SubmitTask,
    nodes: MappingProxyType[NodeId, NodeSlots],
    tm_ref: ActorRef,
    inflight: MappingProxyType[str, SubmitTask],
    node_tasks: MappingProxyType[NodeId, frozenset[str]],
) -> tuple[
    MappingProxyType[NodeId, NodeSlots],
    MappingProxyType[str, SubmitTask],
    MappingProxyType[NodeId, frozenset[str]],
]:
    slot = nodes[nid]
    tid = task.task_id
    tm_ref.tell(TaskSubmitted(task_id=tid, node_id=nid))
    slot.ref.tell(_to_node_message(task, tm_ref))
    new_nodes = MappingProxyType({**nodes, nid: NodeSlots(slot.ref, slot.total, slot.used + 1)})
    new_inflight = MappingProxyType({**inflight, tid: task})
    existing = node_tasks.get(nid, frozenset())
    new_node_tasks = MappingProxyType({**node_tasks, nid: existing | {tid}})
    return new_nodes, new_inflight, new_node_tasks


def _drain_queue(
    queue: tuple[SubmitTask, ...],
    nodes: MappingProxyType[NodeId, NodeSlots],
    round_robin: int,
    tm_ref: ActorRef,
    inflight: MappingProxyType[str, SubmitTask],
    node_tasks: MappingProxyType[NodeId, frozenset[str]],
) -> tuple[
    tuple[SubmitTask, ...],
    MappingProxyType[NodeId, NodeSlots],
    int,
    MappingProxyType[str, SubmitTask],
    MappingProxyType[NodeId, frozenset[str]],
]:
    remaining: list[SubmitTask] = []
    for task in queue:
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
