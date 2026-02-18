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
    InstanceMetadata,
    NodeBecameReady,
    NodeLost,
    NodeMsg,
    Provision,
    TaskResult,
)
from skyward.observability.logger import logger

log = logger.bind(actor="node")

type NodeId = int


@dataclass(frozen=True, slots=True)
class ActiveState:
    cluster: Any
    provider: Any
    instance_ref: ActorRef | None
    pending_tasks: tuple[tuple[bytes, ActorRef], ...] = ()
    inflight: dict[str, ActorRef] = field(default_factory=dict)
    task_counter: int = 0
    current_metadata: InstanceMetadata | None = None


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
        pending_tasks: tuple[tuple[bytes, ActorRef], ...] = (),
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
                case InstanceBecameReady(instance_id=iid, ip=ip, metadata=meta):
                    log.info(
                        "Node {nid} ready (instance={iid}, ip={ip})",
                        nid=node_id, iid=iid, ip=ip,
                    )
                    meta = meta or InstanceMetadata(
                        id=iid, node=node_id, provider=cluster.spec.provider or "aws", ip=ip,
                        ssh_user=cluster.ssh_user, ssh_key_path=cluster.ssh_key_path,
                    )
                    pool.tell(NodeBecameReady(node_id=node_id, instance=meta))
                    return active(ActiveState(
                        cluster=cluster, provider=provider,
                        instance_ref=instance_ref, pending_tasks=pending_tasks,
                        current_metadata=meta,
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
                case ExecuteOnNode(fn_bytes, reply_to):
                    log.debug("Node {nid} queuing task while waiting", nid=node_id)
                    return waiting(
                        cluster, provider, instance_ref,
                        (*pending_tasks, (fn_bytes, reply_to)),
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def replacing(
        cluster: Any,
        provider: Any,
        pending_tasks: tuple[tuple[bytes, ActorRef], ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown():
                    return Behaviors.same()
                case Provision(instance=instance):
                    instance_ref = _spawn_instance(ctx, instance, provider, cluster)
                    return waiting(cluster, provider, instance_ref, pending_tasks)
                case ExecuteOnNode(fn_bytes, reply_to):
                    return replacing(
                        cluster, provider,
                        (*pending_tasks, (fn_bytes, reply_to)),
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
                case InstanceBecameReady(instance_id=iid, ip=ip, metadata=meta):
                    log.info(
                        "Node {nid} replacement ready, replaying {n} pending tasks",
                        nid=node_id, n=len(s.pending_tasks),
                    )
                    meta = meta or s.current_metadata or InstanceMetadata(
                        id=iid, node=node_id, provider="aws", ip=ip,
                    )
                    pool.tell(NodeBecameReady(node_id=node_id, instance=meta))
                    new_inflight = dict(s.inflight)
                    new_counter = s.task_counter
                    if s.instance_ref:
                        for fn_bytes, rto in s.pending_tasks:
                            tid = str(new_counter)
                            s.instance_ref.tell(Execute(
                                fn_bytes=fn_bytes, reply_to=ctx.self, task_id=tid,
                            ))
                            new_inflight[tid] = rto
                            new_counter += 1
                    return active(replace(
                        s, pending_tasks=(), inflight=new_inflight,
                        task_counter=new_counter, current_metadata=meta,
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
                case ExecuteOnNode(fn_bytes, reply_to, task_id=tid):
                    local_tid = tid or str(s.task_counter)
                    log.debug("Node {nid} dispatching task {tid}", nid=node_id, tid=local_tid)
                    if s.instance_ref:
                        s.instance_ref.tell(Execute(
                            fn_bytes=fn_bytes, reply_to=ctx.self, task_id=local_tid,
                        ))
                        new_inflight = {**s.inflight, local_tid: reply_to}
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
        await provider.terminate((dead_id,))
    except Exception as e:
        log.warning("Failed to terminate dead instance {iid}: {err}", iid=dead_id, err=e)
    instances = await provider.provision(cluster, 1)
    if not instances:
        raise RuntimeError("Failed to provision replacement instance")
    return instances[0]
