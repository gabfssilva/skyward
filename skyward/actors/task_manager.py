from casty import ActorContext, Behavior, Behaviors

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeAvailable,
    NodeSlots,
    NodeUnavailable,
    SlotFreed,
    SubmitBroadcast,
    SubmitTask,
    TaskManagerMsg,
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


def _drain_queue(
    queue: tuple[SubmitTask, ...],
    nodes: dict[NodeId, NodeSlots],
    round_robin: int,
) -> tuple[tuple[SubmitTask, ...], dict[NodeId, NodeSlots], int]:
    remaining: list[SubmitTask] = []
    for task in queue:
        nid = _pick_with_free_slot(nodes, round_robin)
        if nid is None:
            remaining.append(task)
            continue
        slot = nodes[nid]
        slot.ref.tell(ExecuteOnNode(fn_bytes=task.fn_bytes, reply_to=task.reply_to))
        nodes = {**nodes, nid: NodeSlots(slot.ref, slot.total, slot.used + 1)}
        round_robin += 1
    return tuple(remaining), nodes, round_robin


def task_manager_actor() -> Behavior[TaskManagerMsg]:

    def active(
        nodes: dict[NodeId, NodeSlots] | None = None,
        queue: tuple[SubmitTask, ...] = (),
        round_robin: int = 0,
    ) -> Behavior[TaskManagerMsg]:
        if nodes is None:
            nodes = {}

        async def receive(ctx: ActorContext[TaskManagerMsg], msg: TaskManagerMsg) -> Behavior[TaskManagerMsg]:
            match msg:
                case NodeAvailable(node_id, node_ref, slots):
                    new_nodes = {**nodes, node_id: NodeSlots(node_ref, total=slots, used=0)}
                    remaining, new_nodes, rr = _drain_queue(queue, new_nodes, round_robin)
                    return active(new_nodes, remaining, rr)
                case NodeUnavailable(node_id):
                    new_nodes = {k: v for k, v in nodes.items() if k != node_id}
                    return active(new_nodes, queue, round_robin)
                case SlotFreed(node_id):
                    if node_id not in nodes:
                        return Behaviors.same()
                    slot = nodes[node_id]
                    new_nodes = {**nodes, node_id: NodeSlots(slot.ref, slot.total, max(0, slot.used - 1))}
                    remaining, new_nodes, rr = _drain_queue(queue, new_nodes, round_robin)
                    return active(new_nodes, remaining, rr)
                case SubmitTask(fn_bytes, reply_to):
                    nid = _pick_with_free_slot(nodes, round_robin)
                    if nid is None:
                        return active(nodes, (*queue, SubmitTask(fn_bytes, reply_to)), round_robin)
                    slot = nodes[nid]
                    slot.ref.tell(ExecuteOnNode(fn_bytes=fn_bytes, reply_to=reply_to))
                    new_nodes = {**nodes, nid: NodeSlots(slot.ref, slot.total, slot.used + 1)}
                    return active(new_nodes, queue, round_robin + 1)
                case SubmitBroadcast(fn_bytes, reply_to):
                    for nid, slot in nodes.items():
                        slot.ref.tell(ExecuteOnNode(fn_bytes=fn_bytes, reply_to=reply_to))
                    return Behaviors.same()
            return Behaviors.same()
        return Behaviors.receive(receive)

    return active()
