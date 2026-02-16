from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.instance import instance_actor
from skyward.actors.messages import (
    BootstrapRequested,
    Execute,
    ExecuteOnNode,
    InstanceBecameReady,
    InstanceBootstrapped,
    InstanceDied,
    InstanceLaunched,
    InstanceMetadata,
    InstanceProvisioned,
    InstanceRequested,
    InstanceRunning,
    NodeBecameReady,
    NodeLost,
    NodeMsg,
    Provision,
    Running,
    SetWorkerRef,
    TaskResult,
    _to_metadata,
)

if TYPE_CHECKING:
    from casty import ClusterClient

type NodeId = int


@dataclass(frozen=True, slots=True)
class ActiveState:
    cluster_id: str
    provider_ref: ActorRef
    worker_ref: ActorRef | None
    client: ClusterClient | None
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
    """A node tells this story: idle → provisioning → waiting → bootstrapping → active."""

    def _request_instance(provider_ref: ActorRef, cluster_id: str, node_ref: ActorRef) -> None:
        provider_ref.tell(InstanceRequested(
            request_id=str(uuid.uuid4()),
            provider="aws",  # type: ignore[arg-type]
            cluster_id=cluster_id,
            node_id=node_id,
            reply_to=node_ref,
        ))

    def _spawn_instance(
        ctx: ActorContext, instance_id: str, provider_ref: ActorRef,
        worker_ref: ActorRef | None, client: ClusterClient | None = None,
        metadata: InstanceMetadata | None = None,
    ) -> ActorRef:
        behavior = instance_actor(
            instance_id=instance_id,
            provider_ref=provider_ref,
            worker_ref=worker_ref,
            client=client,
            parent=ctx.self,
            metadata=metadata,
            _skip_tunnel=True,
            ssh_timeout=ssh_timeout,
            ssh_retry_interval=ssh_retry_interval,
        )
        return ctx.spawn(
            Behaviors.spy(behavior, observer=panel) if panel else behavior,
            name=f"instance-{instance_id}",
        )

    def idle() -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case Provision(cluster_id, provider_ref):
                    _request_instance(provider_ref, cluster_id, ctx.self)
                    return provisioning(cluster_id, provider_ref)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def provisioning(
        cluster_id: str,
        provider_ref: ActorRef,
        worker_ref: ActorRef | None = None,
        client: ClusterClient | None = None,
        pending_tasks: tuple[tuple[bytes, ActorRef], ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case SetWorkerRef(worker_ref=wref, client=cl):
                    return provisioning(cluster_id, provider_ref, wref, cl or client, pending_tasks)
                case InstanceLaunched(instance_id=iid):
                    instance_ref = _spawn_instance(ctx, iid, provider_ref, worker_ref, client)
                    return waiting_for_running(
                        cluster_id, provider_ref, worker_ref, client, instance_ref, pending_tasks,
                    )
                case InstanceRunning() as event:
                    metadata = _to_metadata(event)
                    pool.tell(InstanceProvisioned(request_id=event.request_id, instance=metadata))
                    instance_ref = _spawn_instance(
                        ctx, metadata.id, provider_ref, worker_ref, client, metadata,
                    )
                    instance_ref.tell(Running(ip=metadata.ip))
                    provider_ref.tell(BootstrapRequested(
                        request_id=str(uuid.uuid4()),
                        instance=metadata,
                        cluster_id=cluster_id,
                        reply_to=ctx.self,
                    ))
                    return bootstrapping(
                        cluster_id, provider_ref, worker_ref, client,
                        instance_ref, metadata, pending_tasks,
                    )
                case ExecuteOnNode(fn_bytes, reply_to):
                    return provisioning(
                        cluster_id, provider_ref, worker_ref, client,
                        (*pending_tasks, (fn_bytes, reply_to)),
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def waiting_for_running(
        cluster_id: str,
        provider_ref: ActorRef,
        worker_ref: ActorRef | None,
        client: ClusterClient | None,
        instance_ref: ActorRef,
        pending_tasks: tuple[tuple[bytes, ActorRef], ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case SetWorkerRef(worker_ref=wref, client=cl):
                    instance_ref.tell(SetWorkerRef(worker_ref=wref, client=cl))
                    return waiting_for_running(
                        cluster_id, provider_ref, wref, cl or client, instance_ref, pending_tasks,
                    )
                case InstanceRunning() as event:
                    metadata = _to_metadata(event)
                    pool.tell(InstanceProvisioned(request_id=event.request_id, instance=metadata))
                    instance_ref.tell(Running(ip=metadata.ip))
                    provider_ref.tell(BootstrapRequested(
                        request_id=str(uuid.uuid4()),
                        instance=metadata,
                        cluster_id=cluster_id,
                        reply_to=ctx.self,
                    ))
                    return bootstrapping(
                        cluster_id, provider_ref, worker_ref, client,
                        instance_ref, metadata, pending_tasks,
                    )
                case InstanceDied():
                    _request_instance(provider_ref, cluster_id, ctx.self)
                    return provisioning(cluster_id, provider_ref, worker_ref, client, pending_tasks)
                case ExecuteOnNode(fn_bytes, reply_to):
                    return waiting_for_running(
                        cluster_id, provider_ref, worker_ref, client, instance_ref,
                        (*pending_tasks, (fn_bytes, reply_to)),
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def bootstrapping(
        cluster_id: str,
        provider_ref: ActorRef,
        worker_ref: ActorRef | None,
        client: ClusterClient | None,
        instance_ref: ActorRef,
        metadata: InstanceMetadata,
        pending_tasks: tuple[tuple[bytes, ActorRef], ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case SetWorkerRef(worker_ref=wref, client=cl):
                    instance_ref.tell(SetWorkerRef(worker_ref=wref, client=cl))
                    return bootstrapping(
                        cluster_id, provider_ref, wref, cl or client,
                        instance_ref, metadata, pending_tasks,
                    )
                case InstanceBecameReady():
                    pool.tell(NodeBecameReady(node_id=node_id, instance=metadata))
                    return active(ActiveState(
                        cluster_id=cluster_id, provider_ref=provider_ref,
                        worker_ref=worker_ref, client=client,
                        instance_ref=instance_ref, pending_tasks=pending_tasks,
                        current_metadata=metadata,
                    ))
                case InstanceBootstrapped(instance=inst):
                    pool.tell(NodeBecameReady(node_id=node_id, instance=inst))
                    return active(ActiveState(
                        cluster_id=cluster_id, provider_ref=provider_ref,
                        worker_ref=worker_ref, client=client,
                        instance_ref=instance_ref, pending_tasks=pending_tasks,
                        current_metadata=inst,
                    ))
                case InstanceDied(reason=reason):
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    _request_instance(provider_ref, cluster_id, ctx.self)
                    return provisioning(cluster_id, provider_ref, worker_ref, client, pending_tasks)
                case ExecuteOnNode(fn_bytes, reply_to):
                    return bootstrapping(
                        cluster_id, provider_ref, worker_ref, client, instance_ref, metadata,
                        (*pending_tasks, (fn_bytes, reply_to)),
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def active(s: ActiveState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case SetWorkerRef(worker_ref=wref, client=cl):
                    if s.instance_ref:
                        s.instance_ref.tell(SetWorkerRef(worker_ref=wref, client=cl))
                    return active(replace(s, worker_ref=wref, client=cl or s.client))
                case InstanceBecameReady(instance_id=iid, ip=ip):
                    meta = s.current_metadata or InstanceMetadata(
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
                case InstanceRunning() as ev:
                    return active(replace(s, current_metadata=_to_metadata(ev)))
                case InstanceDied(reason=reason):
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    _request_instance(s.provider_ref, s.cluster_id, ctx.self)
                    return provisioning(
                        s.cluster_id, s.provider_ref, s.worker_ref, s.client, s.pending_tasks,
                    )
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
