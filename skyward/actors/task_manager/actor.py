from dataclasses import replace
from types import MappingProxyType
from typing import Any

from casty import ActorContext, Behavior, Behaviors

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeAvailable,
    NodeSlots,
    NodeUnavailable,
    RegisterPressureObserver,
    SubmitBroadcast,
    SubmitTask,
    TaskFailed,
    TaskInterrupted,
    TaskSubmitted,
    TaskSucceeded,
)
from skyward.observability.logger import logger

from .messages import TaskManagerMsg
from .state import (
    NodeId,
    PendingBroadcast,
    _dispatch,
    _drain_queue,
    _emit_pressure,
    _pick_with_free_slot,
    _State,
)

log = logger.bind(actor="task_manager")


def _handle_broadcast_result(
    broadcasts: MappingProxyType[str, PendingBroadcast],
    node_id: int,
    value: Any,
) -> tuple[bool, MappingProxyType[str, PendingBroadcast]]:
    for bid, bc in broadcasts.items():
        if node_id in bc.pending:
            new_bc = replace(
                bc,
                pending=bc.pending - {node_id},
                results=MappingProxyType({**bc.results, node_id: value}),
            )
            return True, MappingProxyType({**broadcasts, bid: new_bc})
    return False, broadcasts


def _remove_task_from_node(
    node_tasks: MappingProxyType[NodeId, frozenset[str]],
    node_id: int,
    task_id: str,
) -> MappingProxyType[NodeId, frozenset[str]]:
    existing = node_tasks.get(node_id, frozenset())
    updated = existing - {task_id}
    if updated:
        return MappingProxyType({**node_tasks, node_id: updated})
    return MappingProxyType({k: v for k, v in node_tasks.items() if k != node_id})


def _free_slot_and_drain(
    s: _State,
    node_id: int,
    ctx: ActorContext[TaskManagerMsg],
    new_inflight: MappingProxyType[str, SubmitTask],
    new_broadcasts: MappingProxyType[str, PendingBroadcast],
    new_retries: MappingProxyType[str, int],
    new_queue: tuple[SubmitTask, ...] | None = None,
    new_node_tasks: MappingProxyType[NodeId, frozenset[str]] | None = None,
) -> _State:
    queue = new_queue if new_queue is not None else s.queue
    nt = new_node_tasks if new_node_tasks is not None else s.node_tasks
    if node_id not in s.nodes:
        return replace(s, broadcasts=new_broadcasts, inflight=new_inflight, retries=new_retries, queue=queue, node_tasks=nt)
    slot = s.nodes[node_id]
    new_used = max(0, slot.used - 1)
    new_nodes = MappingProxyType({**s.nodes, node_id: NodeSlots(slot.ref, slot.total, new_used)})
    remaining, new_nodes, rr, new_inflight, nt = _drain_queue(
        queue, new_nodes, s.round_robin, ctx.self, new_inflight, nt,
    )
    return replace(
        s, nodes=new_nodes, queue=remaining, round_robin=rr,
        inflight=new_inflight, broadcasts=new_broadcasts, retries=new_retries, node_tasks=nt,
    )


def task_manager_actor(retry_on_interruption: int = 3) -> Behavior[TaskManagerMsg]:

    def active(s: _State) -> Behavior[TaskManagerMsg]:

        async def receive(
            ctx: ActorContext[TaskManagerMsg], msg: TaskManagerMsg,
        ) -> Behavior[TaskManagerMsg]:
            match msg:
                case NodeAvailable(node_id, node_ref, slots):
                    log.info(
                        "Node {nid} available ({slots} slots)",
                        nid=node_id, slots=slots,
                    )
                    new_nodes = MappingProxyType({
                        **s.nodes,
                        node_id: NodeSlots(ref=node_ref, total=slots, used=0),
                    })
                    remaining, new_nodes, rr, new_inflight, new_nt = _drain_queue(
                        s.queue, new_nodes, s.round_robin, ctx.self, s.inflight, s.node_tasks,
                    )
                    if len(remaining) < len(s.queue):
                        log.debug("Drained {n} queued tasks", n=len(s.queue) - len(remaining))
                    new_s = replace(s, nodes=new_nodes, queue=remaining, round_robin=rr, inflight=new_inflight, node_tasks=new_nt)
                    _emit_pressure(new_s)
                    return active(new_s)

                case NodeUnavailable(node_id):
                    orphaned_tids = s.node_tasks.get(node_id, frozenset())
                    if orphaned_tids:
                        log.info(
                            "Node {nid} unavailable, re-enqueuing {n} orphaned tasks",
                            nid=node_id, n=len(orphaned_tids),
                        )
                    else:
                        log.info("Node {nid} unavailable", nid=node_id)
                    new_nodes = MappingProxyType({k: v for k, v in s.nodes.items() if k != node_id})
                    new_broadcasts = MappingProxyType({
                        bid: replace(
                            bc,
                            pending=bc.pending - {node_id},
                            results=MappingProxyType({
                                **bc.results,
                                node_id: RuntimeError(f"Node {node_id} lost during broadcast"),
                            }),
                        ) if node_id in bc.pending else bc
                        for bid, bc in s.broadcasts.items()
                    })
                    requeue = tuple(s.inflight[tid] for tid in orphaned_tids if tid in s.inflight)
                    new_inflight = MappingProxyType({k: v for k, v in s.inflight.items() if k not in orphaned_tids})
                    new_node_tasks = MappingProxyType({k: v for k, v in s.node_tasks.items() if k != node_id})
                    new_s = replace(
                        s, nodes=new_nodes, broadcasts=new_broadcasts,
                        inflight=new_inflight, node_tasks=new_node_tasks,
                        queue=(*s.queue, *requeue),
                    )
                    _emit_pressure(new_s)
                    return _check_broadcasts(new_s)

                case TaskSucceeded(value=value, node_id=node_id, task_id=tid, elapsed=elapsed):
                    broadcast_hit, new_broadcasts = _handle_broadcast_result(s.broadcasts, node_id, value)
                    new_inflight = s.inflight
                    new_retries = MappingProxyType({k: v for k, v in s.retries.items() if k != tid})
                    if not broadcast_hit:
                        task = s.inflight.get(tid)
                        if task:
                            task.reply_to.tell(TaskSucceeded(
                                value=value, node_id=node_id, task_id=tid, elapsed=elapsed,
                            ))
                            new_inflight = MappingProxyType({k: v for k, v in s.inflight.items() if k != tid})
                    new_nt = _remove_task_from_node(s.node_tasks, node_id, tid)
                    new_s = _free_slot_and_drain(s, node_id, ctx, new_inflight, new_broadcasts, new_retries, new_node_tasks=new_nt)
                    _emit_pressure(new_s)
                    return _check_broadcasts(new_s) if broadcast_hit else active(new_s)

                case TaskFailed(error=err, node_id=node_id, task_id=tid):
                    broadcast_hit, new_broadcasts = _handle_broadcast_result(s.broadcasts, node_id, err)
                    new_inflight = s.inflight
                    new_retries = MappingProxyType({k: v for k, v in s.retries.items() if k != tid})
                    if not broadcast_hit:
                        task = s.inflight.get(tid)
                        if task:
                            task.reply_to.tell(TaskFailed(
                                error=err, node_id=node_id, task_id=tid,
                            ))
                            new_inflight = MappingProxyType({k: v for k, v in s.inflight.items() if k != tid})
                    new_nt = _remove_task_from_node(s.node_tasks, node_id, tid)
                    new_s = _free_slot_and_drain(s, node_id, ctx, new_inflight, new_broadcasts, new_retries, new_node_tasks=new_nt)
                    _emit_pressure(new_s)
                    return _check_broadcasts(new_s) if broadcast_hit else active(new_s)

                case TaskInterrupted(error=err, node_id=node_id, task_id=tid):
                    broadcast_hit, new_broadcasts = _handle_broadcast_result(s.broadcasts, node_id, err)
                    new_inflight = s.inflight
                    new_retries = s.retries
                    new_queue = s.queue
                    if not broadcast_hit:
                        task = s.inflight.get(tid)
                        if task:
                            attempt = s.retries.get(tid, 0) + 1
                            new_inflight = MappingProxyType({k: v for k, v in s.inflight.items() if k != tid})
                            if attempt <= s.retry_on_interruption:
                                log.info(
                                    "Task {tid} interrupted on node {nid}, re-enqueuing (attempt {att}/{max})",
                                    tid=tid, nid=node_id, att=attempt, max=s.retry_on_interruption,
                                )
                                new_retries = MappingProxyType({**s.retries, tid: attempt})
                                new_queue = (*s.queue, task)
                            else:
                                log.warning(
                                    "Task {tid} interrupted on node {nid}, retries exhausted ({max})",
                                    tid=tid, nid=node_id, max=s.retry_on_interruption,
                                )
                                new_retries = MappingProxyType({k: v for k, v in s.retries.items() if k != tid})
                                task.reply_to.tell(TaskFailed(
                                    error=err, node_id=node_id, task_id=tid,
                                ))
                    new_nt = _remove_task_from_node(s.node_tasks, node_id, tid)
                    new_s = _free_slot_and_drain(
                        s, node_id, ctx, new_inflight, new_broadcasts, new_retries, new_queue=new_queue, new_node_tasks=new_nt,
                    )
                    _emit_pressure(new_s)
                    return _check_broadcasts(new_s) if broadcast_hit else active(new_s)

                case SubmitTask() as task:
                    nid = _pick_with_free_slot(s.nodes, s.round_robin)
                    if nid is None:
                        log.debug(
                            "No available nodes, queuing task (queue_size={qs})",
                            qs=len(s.queue) + 1,
                        )
                        new_s = replace(s, queue=(*s.queue, task))
                        _emit_pressure(new_s)
                        return active(new_s)
                    log.debug("Dispatching task to node {nid}", nid=nid)
                    new_nodes, new_inflight, new_nt = _dispatch(
                        nid, task, s.nodes, ctx.self, s.inflight, s.node_tasks,
                    )
                    return active(replace(s, nodes=new_nodes, inflight=new_inflight, node_tasks=new_nt, round_robin=s.round_robin + 1))

                case SubmitBroadcast() as bcast:
                    n = len(s.nodes)
                    log.debug("Broadcasting task to {n} nodes", n=n)
                    pending_nodes: frozenset[NodeId] = frozenset()
                    new_nodes = dict(s.nodes)
                    for nid, slot in s.nodes.items():
                        ctx.self.tell(TaskSubmitted(task_id=bcast.task_id, node_id=nid))
                        slot.ref.tell(ExecuteOnNode(
                            fn=bcast.fn, args=bcast.args, kwargs=bcast.kwargs,
                            reply_to=ctx.self, task_id=bcast.task_id,
                            timeout=bcast.timeout,
                        ))
                        new_nodes[nid] = NodeSlots(slot.ref, slot.total, slot.used + 1)
                        pending_nodes = pending_nodes | {nid}
                    new_broadcasts = MappingProxyType({
                        **s.broadcasts,
                        bcast.task_id: PendingBroadcast(
                            caller=bcast.reply_to, pending=pending_nodes,
                        ),
                    })
                    return active(replace(s, nodes=MappingProxyType(new_nodes), broadcasts=new_broadcasts))

                case RegisterPressureObserver(observer=observer):
                    log.info("Pressure observer registered")
                    new_s = replace(s, pressure_observer=observer)
                    _emit_pressure(new_s)
                    return active(new_s)

            return Behaviors.same()
        return Behaviors.receive(receive)

    def _check_broadcasts(s: _State) -> Behavior[TaskManagerMsg]:
        new_broadcasts = dict(s.broadcasts)
        for bid, bc in s.broadcasts.items():
            if not bc.pending:
                ordered = [bc.results[nid] for nid in sorted(bc.results)]
                bc.caller.tell(ordered)
                del new_broadcasts[bid]
        return active(replace(s, broadcasts=MappingProxyType(new_broadcasts)))

    log.info("Task manager started")
    return active(_State(
        nodes=MappingProxyType({}),
        queue=(),
        round_robin=0,
        inflight=MappingProxyType({}),
        broadcasts=MappingProxyType({}),
        retry_on_interruption=retry_on_interruption,
    ))
