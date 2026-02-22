from dataclasses import dataclass, field, replace
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeAvailable,
    NodeSlots,
    NodeUnavailable,
    SubmitBroadcast,
    SubmitTask,
    TaskManagerMsg,
    TaskResult,
    TaskSubmitted,
)
from skyward.observability.logger import logger

log = logger.bind(actor="task_manager")

type NodeId = int


@dataclass(slots=True)
class PendingBroadcast:
    caller: ActorRef
    pending: set[NodeId]
    results: dict[NodeId, Any] = field(default_factory=dict)


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


@dataclass(slots=True)
class _State:
    nodes: dict[NodeId, NodeSlots]
    queue: tuple[SubmitTask, ...]
    round_robin: int
    inflight: dict[str, ActorRef]
    broadcasts: dict[str, PendingBroadcast]



def task_manager_actor() -> Behavior[TaskManagerMsg]:

    def active(s: _State) -> Behavior[TaskManagerMsg]:

        async def receive(
            ctx: ActorContext[TaskManagerMsg], msg: TaskManagerMsg,
        ) -> Behavior[TaskManagerMsg]:
            match msg:
                case NodeAvailable(node_id, node_ref, slots):
                    buffered = slots + 1
                    log.info(
                        "Node {nid} available ({slots} slots, buffered={buf})",
                        nid=node_id, slots=slots, buf=buffered,
                    )
                    new_nodes = {
                        **s.nodes,
                        node_id: NodeSlots(ref=node_ref, total=buffered, used=0),
                    }
                    remaining, new_nodes, rr = _drain_queue(
                        s.queue, new_nodes, s.round_robin, ctx.self, s.inflight,
                    )
                    if len(remaining) < len(s.queue):
                        log.debug("Drained {n} queued tasks", n=len(s.queue) - len(remaining))
                    return active(replace(s, nodes=new_nodes, queue=remaining, round_robin=rr))

                case NodeUnavailable(node_id):
                    log.info("Node {nid} unavailable", nid=node_id)
                    new_nodes = {k: v for k, v in s.nodes.items() if k != node_id}
                    for bc in s.broadcasts.values():
                        if node_id in bc.pending:
                            bc.pending.discard(node_id)
                            bc.results[node_id] = RuntimeError(
                                f"Node {node_id} lost during broadcast",
                            )
                    new_s = replace(s, nodes=new_nodes)
                    return _check_broadcasts(new_s)

                case TaskResult(value, node_id, task_id=tid):
                    broadcast_hit = False
                    for bc in s.broadcasts.values():
                        if node_id in bc.pending:
                            bc.pending.discard(node_id)
                            bc.results[node_id] = value
                            broadcast_hit = True
                            break

                    if not broadcast_hit:
                        caller = s.inflight.pop(tid, None)
                        if caller:
                            caller.tell(TaskResult(value=value, node_id=node_id, task_id=tid))

                    if node_id not in s.nodes:
                        return _check_broadcasts(s) if broadcast_hit else Behaviors.same()
                    slot = s.nodes[node_id]
                    new_used = max(0, slot.used - 1)
                    new_nodes = {**s.nodes, node_id: NodeSlots(slot.ref, slot.total, new_used)}
                    remaining, new_nodes, rr = _drain_queue(
                        s.queue, new_nodes, s.round_robin, ctx.self, s.inflight,
                    )
                    new_s = replace(s, nodes=new_nodes, queue=remaining, round_robin=rr)
                    return _check_broadcasts(new_s) if broadcast_hit else active(new_s)

                case SubmitTask() as task:
                    nid = _pick_with_free_slot(s.nodes, s.round_robin)
                    if nid is None:
                        log.debug(
                            "No available nodes, queuing task (queue_size={qs})",
                            qs=len(s.queue) + 1,
                        )
                        return active(replace(s, queue=(*s.queue, task)))
                    log.debug("Dispatching task to node {nid}", nid=nid)
                    new_nodes = _dispatch(
                        nid, task.task_id, task.fn, task.args, task.kwargs,
                        task.reply_to, s.nodes, ctx.self, s.inflight,
                        timeout=task.timeout,
                    )
                    return active(replace(s, nodes=new_nodes, round_robin=s.round_robin + 1))

                case SubmitBroadcast() as bcast:
                    n = len(s.nodes)
                    log.debug("Broadcasting task to {n} nodes", n=n)
                    pending_nodes: set[NodeId] = set()
                    new_nodes = dict(s.nodes)
                    for nid, slot in s.nodes.items():
                        ctx.self.tell(TaskSubmitted(task_id=bcast.task_id, node_id=nid))
                        slot.ref.tell(ExecuteOnNode(
                            fn=bcast.fn, args=bcast.args, kwargs=bcast.kwargs,
                            reply_to=ctx.self, task_id=bcast.task_id,
                            timeout=bcast.timeout,
                        ))
                        new_nodes[nid] = NodeSlots(slot.ref, slot.total, slot.used + 1)
                        pending_nodes.add(nid)
                    s.broadcasts[bcast.task_id] = PendingBroadcast(
                        caller=bcast.reply_to, pending=pending_nodes,
                    )
                    return active(replace(s, nodes=new_nodes))

            return Behaviors.same()
        return Behaviors.receive(receive)

    def _check_broadcasts(s: _State) -> Behavior[TaskManagerMsg]:
        done_ids: list[str] = []
        for bid, bc in s.broadcasts.items():
            if not bc.pending:
                ordered = [bc.results[nid] for nid in sorted(bc.results)]
                bc.caller.tell(ordered)
                done_ids.append(bid)
        for bid in done_ids:
            del s.broadcasts[bid]
        return active(s)

    log.info("Task manager started")
    return active(_State(
        nodes={}, queue=(), round_robin=0, inflight={}, broadcasts={},
    ))
