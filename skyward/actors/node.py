from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.instance import instance_actor
from skyward.actors.messages import (
    Execute,
    ExecuteOnNode,
    InstanceBecameReady,
    InstanceDied,
    InstanceMetadata,
    NodeBecameReady,
    NodeLost,
    NodeMsg,
    Provision,
    SetHeadAddr,
    TaskResult,
)

type NodeId = int


@dataclass(frozen=True, slots=True)
class ActiveState:
    cluster: Any
    provider: Any
    instance_ref: ActorRef | None
    pending_tasks: tuple[tuple[bytes, ActorRef], ...] = ()
    inflight: tuple[tuple[int, ActorRef], ...] = ()
    task_counter: int = 0
    current_metadata: InstanceMetadata | None = None


def node_actor(
    node_id: NodeId,
    pool: ActorRef,
    panel: ActorRef | None = None,
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
        return ctx.spawn(
            Behaviors.spy(behavior, observer=panel) if panel else behavior,
            name=f"instance-{instance.id}",
        )

    def idle() -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case Provision(cluster=cluster, provider=provider, instance=instance):
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
                case SetHeadAddr() as h:
                    instance_ref.tell(h)
                    return Behaviors.same()
                case InstanceBecameReady(instance_id=iid, ip=ip, metadata=meta):
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
                case InstanceDied(reason=reason):
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    ctx.pipe_to_self(
                        _provision_replacement(provider, cluster),
                        mapper=lambda inst: Provision(
                            cluster=cluster, provider=provider, instance=inst,
                        ),
                    )
                    return replacing(cluster, provider, pending_tasks)
                case ExecuteOnNode(fn_bytes, reply_to):
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
                case SetHeadAddr():
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
                case SetHeadAddr() as h:
                    if s.instance_ref:
                        s.instance_ref.tell(h)
                    return Behaviors.same()
                case InstanceBecameReady(instance_id=iid, ip=ip, metadata=meta):
                    meta = meta or s.current_metadata or InstanceMetadata(
                        id=iid, node=node_id, provider="aws", ip=ip,
                    )
                    pool.tell(NodeBecameReady(node_id=node_id, instance=meta))
                    new_inflight = s.inflight
                    new_counter = s.task_counter
                    if s.instance_ref:
                        for fn_bytes, rto in s.pending_tasks:
                            s.instance_ref.tell(Execute(fn_bytes=fn_bytes, reply_to=ctx.self))
                            new_inflight = (*new_inflight, (new_counter, rto))
                            new_counter += 1
                    return active(replace(
                        s, pending_tasks=(), inflight=new_inflight,
                        task_counter=new_counter, current_metadata=meta,
                    ))
                case InstanceDied(reason=reason):
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    ctx.pipe_to_self(
                        _provision_replacement(s.provider, s.cluster),
                        mapper=lambda inst: Provision(
                            cluster=s.cluster, provider=s.provider, instance=inst,
                        ),
                    )
                    return replacing(s.cluster, s.provider, s.pending_tasks)
                case ExecuteOnNode(fn_bytes, reply_to):
                    if s.instance_ref:
                        s.instance_ref.tell(Execute(fn_bytes=fn_bytes, reply_to=ctx.self))
                        return active(replace(
                            s,
                            inflight=(*s.inflight, (s.task_counter, reply_to)),
                            task_counter=s.task_counter + 1,
                        ))
                    return Behaviors.same()
                case TaskResult(value, _):
                    if s.inflight:
                        _, caller = s.inflight[0]
                        caller.tell(TaskResult(value=value, node_id=node_id))
                        return active(replace(s, inflight=s.inflight[1:]))
            return Behaviors.same()
        return Behaviors.receive(receive)

    return idle()


async def _provision_replacement(provider: Any, cluster: Any) -> Any:
    instances = await provider.provision(cluster, 1)
    if not instances:
        raise RuntimeError("Failed to provision replacement instance")
    return instances[0]
