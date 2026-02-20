from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.instance import instance_actor
from skyward.actors.messages import (
    Execute,
    ExecuteOnNode,
    HeadAddressKnown,
    InstanceBecameReady,
    InstanceDied,
    NodeBecameReady,
    NodeInstance,
    NodeLost,
    NodeMsg,
    Provision,
    TaskResult,
)
from skyward.observability.logger import logger

log = logger.bind(actor="node")

type NodeId = int


@dataclass(frozen=True, slots=True)
class _PendingTask:
    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[Any]
    timeout: float


@dataclass(frozen=True, slots=True)
class ActiveState:
    cluster: Any
    provider: Any
    instance_ref: ActorRef | None
    pending_tasks: tuple[_PendingTask, ...] = ()
    inflight: dict[str, ActorRef] = field(default_factory=dict)
    task_counter: int = 0
    current_node_instance: NodeInstance | None = None


def node_actor(
    node_id: NodeId,
    pool: ActorRef,
    ssh_timeout: float = 300.0,
    ssh_retry_interval: float = 5.0,
) -> Behavior[NodeMsg]:
    """A node tells this story: idle → waiting → active."""

    def _spawn_instance(
        ctx: ActorContext, instance: Any, provider: Any, cluster: Any,
    ) -> ActorRef:
        behavior = instance_actor(
            instance_id=instance.id,
            provider=provider,
            cluster=cluster,
            spec=cluster.spec,
            node_id=node_id,
            parent=ctx.self,
            ssh_timeout=ssh_timeout,
            ssh_retry_interval=ssh_retry_interval,
        )
        return ctx.spawn(behavior, name=f"instance-{instance.id}")

    def idle() -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case Provision(cluster=cluster, provider=provider, instance=instance):
                    log.info("Node {nid} provisioning instance {iid}", nid=node_id, iid=instance.id)
                    instance_ref = _spawn_instance(ctx, instance, provider, cluster)
                    return waiting(cluster, provider, instance_ref)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def waiting(
        cluster: Any,
        provider: Any,
        instance_ref: ActorRef,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    if node_id == 0:
                        log.debug("Node {nid} forwarding head address to pool", nid=node_id)
                        pool.tell(h)
                    else:
                        log.debug("Node {nid} received head address", nid=node_id)
                        instance_ref.tell(h)
                    return Behaviors.same()
                case InstanceBecameReady(instance_id=iid, ip=ip, node_instance=ni):
                    log.info(
                        "Node {nid} ready (instance={iid}, ip={ip})",
                        nid=node_id, iid=iid, ip=ip,
                    )
                    if ni is None:
                        raise RuntimeError(
                            f"InstanceBecameReady for {iid} has no NodeInstance"
                        )
                    pool.tell(NodeBecameReady(node_id=node_id, instance=ni))
                    return active(ActiveState(
                        cluster=cluster, provider=provider,
                        instance_ref=instance_ref, pending_tasks=pending_tasks,
                        current_node_instance=ni,
                    ))
                case InstanceDied(instance_id=dead_id, reason=reason):
                    log.warning(
                        "Node {nid} instance died: {reason}, replacing",
                        nid=node_id, reason=reason,
                    )
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    ctx.pipe_to_self(
                        _terminate_and_replace(provider, cluster, dead_id),
                        mapper=lambda inst: Provision(
                            cluster=cluster, provider=provider, instance=inst,
                        ),
                    )
                    return replacing(cluster, provider, pending_tasks)
                case ExecuteOnNode() as ex:
                    log.debug("Node {nid} queuing task while waiting", nid=node_id)
                    pt = _PendingTask(ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.timeout)
                    return waiting(
                        cluster, provider, instance_ref,
                        (*pending_tasks, pt),
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def replacing(
        cluster: Any,
        provider: Any,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown():
                    return Behaviors.same()
                case Provision(instance=instance):
                    instance_ref = _spawn_instance(ctx, instance, provider, cluster)
                    return waiting(cluster, provider, instance_ref, pending_tasks)
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.timeout)
                    return replacing(
                        cluster, provider,
                        (*pending_tasks, pt),
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def active(s: ActiveState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    if node_id == 0:
                        pool.tell(h)
                    elif s.instance_ref:
                        s.instance_ref.tell(h)
                    return Behaviors.same()
                case InstanceBecameReady(instance_id=iid, node_instance=ni):
                    log.info(
                        "Node {nid} replacement ready, replaying {n} pending tasks",
                        nid=node_id, n=len(s.pending_tasks),
                    )
                    ni = ni or s.current_node_instance
                    if ni is None:
                        raise RuntimeError(
                            f"InstanceBecameReady for {iid} has no NodeInstance"
                        )
                    pool.tell(NodeBecameReady(node_id=node_id, instance=ni))
                    new_inflight = dict(s.inflight)
                    new_counter = s.task_counter
                    if s.instance_ref:
                        for pt in s.pending_tasks:
                            tid = str(new_counter)
                            s.instance_ref.tell(Execute(
                                fn=pt.fn, args=pt.args, kwargs=pt.kwargs,
                                reply_to=ctx.self, task_id=tid,
                                timeout=pt.timeout,
                            ))
                            new_inflight[tid] = pt.reply_to
                            new_counter += 1
                    return active(replace(
                        s, pending_tasks=(), inflight=new_inflight,
                        task_counter=new_counter, current_node_instance=ni,
                    ))
                case InstanceDied(instance_id=dead_id, reason=reason):
                    log.warning(
                        "Node {nid} instance died while active: {reason}, replacing",
                        nid=node_id, reason=reason,
                    )
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    ctx.pipe_to_self(
                        _terminate_and_replace(s.provider, s.cluster, dead_id),
                        mapper=lambda inst: Provision(
                            cluster=s.cluster, provider=s.provider, instance=inst,
                        ),
                    )
                    return replacing(s.cluster, s.provider, s.pending_tasks)
                case ExecuteOnNode() as ex:
                    local_tid = ex.task_id or str(s.task_counter)
                    log.debug("Node {nid} dispatching task {tid}", nid=node_id, tid=local_tid)
                    if s.instance_ref:
                        s.instance_ref.tell(Execute(
                            fn=ex.fn, args=ex.args, kwargs=ex.kwargs,
                            reply_to=ctx.self, task_id=local_tid,
                            timeout=ex.timeout,
                        ))
                        new_inflight = {**s.inflight, local_tid: ex.reply_to}
                        return active(replace(
                            s,
                            inflight=new_inflight,
                            task_counter=s.task_counter + 1,
                        ))
                    return Behaviors.same()
                case TaskResult(value, _, task_id=tid):
                    log.debug("Node {nid} received task result (tid={tid})", nid=node_id, tid=tid)
                    caller = s.inflight.get(tid)
                    if caller:
                        caller.tell(TaskResult(value=value, node_id=node_id, task_id=tid))
                        new_inflight = {k: v for k, v in s.inflight.items() if k != tid}
                        return active(replace(s, inflight=new_inflight))
            return Behaviors.same()
        return Behaviors.receive(receive)

    return idle()


async def _terminate_and_replace(provider: Any, cluster: Any, dead_id: str) -> Any:
    try:
        await provider.terminate(cluster, (dead_id,))
    except Exception as e:
        log.warning("Failed to terminate dead instance {iid}: {err}", iid=dead_id, err=e)
    _, instances = await provider.provision(cluster, 1)
    if not instances:
        raise RuntimeError("Failed to provision replacement instance")
    return instances[0]
