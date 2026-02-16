from collections import deque

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
)

type NodeId = int


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
    fn_bytes: bytes,
    reply_to: ActorRef,
    nodes: dict[NodeId, NodeSlots],
    tm_ref: ActorRef,
    inflight: dict[NodeId, deque[ActorRef]],
) -> dict[NodeId, NodeSlots]:
    slot = nodes[nid]
    slot.ref.tell(ExecuteOnNode(fn_bytes=fn_bytes, reply_to=tm_ref))
    inflight.setdefault(nid, deque()).append(reply_to)
    return {**nodes, nid: NodeSlots(slot.ref, slot.total, slot.used + 1)}


def _drain_queue(
    queue: tuple[SubmitTask, ...],
    nodes: dict[NodeId, NodeSlots],
    round_robin: int,
    tm_ref: ActorRef,
    inflight: dict[NodeId, deque[ActorRef]],
) -> tuple[tuple[SubmitTask, ...], dict[NodeId, NodeSlots], int]:
    remaining: list[SubmitTask] = []
    for task in queue:
        nid = _pick_with_free_slot(nodes, round_robin)
        if nid is None:
            remaining.append(task)
            continue
        nodes = _dispatch(nid, task.fn_bytes, task.reply_to, nodes, tm_ref, inflight)
        round_robin += 1
    return tuple(remaining), nodes, round_robin


def task_manager_actor() -> Behavior[TaskManagerMsg]:

    def active(
        nodes: dict[NodeId, NodeSlots] | None = None,
        queue: tuple[SubmitTask, ...] = (),
        round_robin: int = 0,
        inflight: dict[NodeId, deque[ActorRef]] | None = None,
    ) -> Behavior[TaskManagerMsg]:
        if nodes is None:
            nodes = {}
        if inflight is None:
            inflight = {}

        async def receive(
            ctx: ActorContext[TaskManagerMsg], msg: TaskManagerMsg,
        ) -> Behavior[TaskManagerMsg]:
            match msg:
                case NodeAvailable(node_id, node_ref, slots):
                    new_nodes = {**nodes, node_id: NodeSlots(ref=node_ref, total=slots, used=0)}
                    remaining, new_nodes, rr = _drain_queue(
                        queue, new_nodes, round_robin, ctx.self, inflight,
                    )
                    return active(new_nodes, remaining, rr, inflight)
                case NodeUnavailable(node_id):
                    new_nodes = {k: v for k, v in nodes.items() if k != node_id}
                    inflight.pop(node_id, None)
                    return active(new_nodes, queue, round_robin, inflight)
                case TaskResult(value, node_id):
                    callers = inflight.get(node_id)
                    if callers:
                        caller = callers.popleft()
                        caller.tell(TaskResult(value=value, node_id=node_id))
                    if node_id not in nodes:
                        return Behaviors.same()
                    slot = nodes[node_id]
                    new_used = max(0, slot.used - 1)
                    new_nodes = {**nodes, node_id: NodeSlots(slot.ref, slot.total, new_used)}
                    remaining, new_nodes, rr = _drain_queue(
                        queue, new_nodes, round_robin, ctx.self, inflight,
                    )
                    return active(new_nodes, remaining, rr, inflight)
                case SubmitTask(fn_bytes, reply_to):
                    nid = _pick_with_free_slot(nodes, round_robin)
                    if nid is None:
                        return active(
                            nodes, (*queue, SubmitTask(fn_bytes, reply_to)),
                            round_robin, inflight,
                        )
                    new_nodes = _dispatch(nid, fn_bytes, reply_to, nodes, ctx.self, inflight)
                    return active(new_nodes, queue, round_robin + 1, inflight)
                case SubmitBroadcast(fn_bytes, reply_to):
                    for _nid, slot in nodes.items():
                        slot.ref.tell(ExecuteOnNode(fn_bytes=fn_bytes, reply_to=reply_to))
                    return Behaviors.same()
            return Behaviors.same()
        return Behaviors.receive(receive)

    return active()
