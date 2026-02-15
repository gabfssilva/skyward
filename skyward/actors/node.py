import uuid

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
    SlotFreed,
    TaskResult,
    _to_metadata,
)

type NodeId = int


def node_actor(
    node_id: NodeId,
    pool: ActorRef,
    task_manager: ActorRef | None = None,
    panel: ActorRef | None = None,
) -> Behavior[NodeMsg]:

    def _request_instance(
        provider_ref: ActorRef, cluster_id: str, node_ref: ActorRef, provider_name: str = "aws",
    ) -> None:
        provider_ref.tell(InstanceRequested(
            request_id=str(uuid.uuid4()),
            provider=provider_name,  # type: ignore[arg-type]
            cluster_id=cluster_id,
            node_id=node_id,
            reply_to=node_ref,
        ))

    def idle() -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case Provision(cluster_id, provider_ref, cluster_client):
                    _request_instance(provider_ref, cluster_id, ctx.self)
                    return provisioning(cluster_id, provider_ref, cluster_client)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def provisioning(
        cluster_id: str,
        provider_ref: ActorRef,
        cluster_client: object,
        pending_tasks: tuple[tuple[bytes, ActorRef], ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case InstanceLaunched(instance_id=iid):
                    behavior = instance_actor(
                        instance_id=iid,
                        provider_ref=provider_ref,
                        cluster_client=cluster_client,
                        parent=ctx.self,
                        _skip_tunnel=True,
                    )
                    instance_ref = ctx.spawn(
                        Behaviors.spy(behavior, observer=panel) if panel else behavior,
                        name=f"instance-{iid}",
                    )
                    return waiting_for_running(
                        cluster_id, provider_ref, cluster_client, instance_ref, pending_tasks
                    )
                case InstanceRunning() as event:
                    metadata = _to_metadata(event)
                    pool.tell(InstanceProvisioned(
                        request_id=event.request_id,
                        instance=metadata,
                    ))

                    behavior = instance_actor(
                        instance_id=metadata.id,
                        provider_ref=provider_ref,
                        cluster_client=cluster_client,
                        parent=ctx.self,
                        metadata=metadata,
                        _skip_tunnel=True,
                    )
                    instance_ref = ctx.spawn(
                        Behaviors.spy(behavior, observer=panel) if panel else behavior,
                        name=f"instance-{metadata.id}",
                    )
                    instance_ref.tell(Running(ip=metadata.ip))

                    provider_ref.tell(BootstrapRequested(
                        request_id=str(uuid.uuid4()),
                        instance=metadata,
                        cluster_id=cluster_id,
                        reply_to=ctx.self,
                    ))
                    return bootstrapping(
                        cluster_id, provider_ref, cluster_client, instance_ref, metadata, pending_tasks
                    )
                case ExecuteOnNode(fn_bytes, reply_to):
                    return provisioning(
                        cluster_id,
                        provider_ref,
                        cluster_client,
                        (*pending_tasks, (fn_bytes, reply_to)),
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def waiting_for_running(
        cluster_id: str,
        provider_ref: ActorRef,
        cluster_client: object,
        instance_ref: ActorRef,
        pending_tasks: tuple[tuple[bytes, ActorRef], ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case InstanceRunning() as event:
                    metadata = _to_metadata(event)
                    pool.tell(InstanceProvisioned(
                        request_id=event.request_id,
                        instance=metadata,
                    ))
                    instance_ref.tell(Running(ip=metadata.ip))
                    provider_ref.tell(BootstrapRequested(
                        request_id=str(uuid.uuid4()),
                        instance=metadata,
                        cluster_id=cluster_id,
                        reply_to=ctx.self,
                    ))
                    return bootstrapping(
                        cluster_id, provider_ref, cluster_client,
                        instance_ref, metadata, pending_tasks,
                    )
                case InstanceDied(_, _):
                    _request_instance(provider_ref, cluster_id, ctx.self)
                    return provisioning(cluster_id, provider_ref, cluster_client, pending_tasks)
                case ExecuteOnNode(fn_bytes, reply_to):
                    return waiting_for_running(
                        cluster_id,
                        provider_ref,
                        cluster_client,
                        instance_ref,
                        (*pending_tasks, (fn_bytes, reply_to)),
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def bootstrapping(
        cluster_id: str,
        provider_ref: ActorRef,
        cluster_client: object,
        instance_ref: ActorRef,
        metadata: InstanceMetadata,
        pending_tasks: tuple[tuple[bytes, ActorRef], ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case InstanceBecameReady(instance_id=iid, ip=ip):
                    final_metadata = metadata
                    pool.tell(NodeBecameReady(node_id=node_id, instance=final_metadata))
                    return active(
                        cluster_id,
                        provider_ref,
                        cluster_client,
                        instance_ref=instance_ref,
                        pending_tasks=pending_tasks,
                        inflight={},
                        current_metadata=final_metadata,
                    )
                case InstanceBootstrapped(instance=instance):
                    pool.tell(NodeBecameReady(node_id=node_id, instance=instance))
                    return active(
                        cluster_id,
                        provider_ref,
                        cluster_client,
                        instance_ref=instance_ref,
                        pending_tasks=pending_tasks,
                        inflight={},
                        current_metadata=instance,
                    )
                case InstanceDied(_, reason):
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    _request_instance(provider_ref, cluster_id, ctx.self)
                    return provisioning(cluster_id, provider_ref, cluster_client, pending_tasks)
                case ExecuteOnNode(fn_bytes, reply_to):
                    return bootstrapping(
                        cluster_id,
                        provider_ref,
                        cluster_client,
                        instance_ref,
                        metadata,
                        (*pending_tasks, (fn_bytes, reply_to)),
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def active(
        cluster_id: str,
        provider_ref: ActorRef,
        cluster_client: object,
        instance_ref: ActorRef | None,
        pending_tasks: tuple[tuple[bytes, ActorRef], ...] = (),
        inflight: dict[int, ActorRef] | None = None,
        task_counter: int = 0,
        current_metadata: InstanceMetadata | None = None,
    ) -> Behavior[NodeMsg]:
        if inflight is None:
            inflight = {}

        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case InstanceBecameReady(instance_id, ip):
                    metadata = current_metadata or InstanceMetadata(
                        id=instance_id, node=node_id, provider="aws", ip=ip,
                    )
                    pool.tell(NodeBecameReady(node_id=node_id, instance=metadata))
                    new_counter = task_counter
                    new_inflight = dict(inflight)
                    if instance_ref:
                        for fn_bytes, reply_to in pending_tasks:
                            instance_ref.tell(Execute(fn_bytes=fn_bytes, reply_to=ctx.self))
                            new_inflight[new_counter] = reply_to
                            new_counter += 1
                    return active(
                        cluster_id,
                        provider_ref,
                        cluster_client,
                        instance_ref,
                        pending_tasks=(),
                        inflight=new_inflight,
                        task_counter=new_counter,
                        current_metadata=metadata,
                    )
                case InstanceRunning() as ev:
                    metadata = _to_metadata(ev)
                    return active(
                        cluster_id,
                        provider_ref,
                        cluster_client,
                        instance_ref,
                        pending_tasks,
                        inflight,
                        task_counter,
                        current_metadata=metadata,
                    )
                case InstanceDied(_, reason):
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    _request_instance(provider_ref, cluster_id, ctx.self)
                    return provisioning(cluster_id, provider_ref, cluster_client, pending_tasks)
                case ExecuteOnNode(fn_bytes, reply_to):
                    if instance_ref:
                        instance_ref.tell(Execute(fn_bytes=fn_bytes, reply_to=ctx.self))
                        new_inflight = {**inflight, task_counter: reply_to}
                        return active(
                            cluster_id,
                            provider_ref,
                            cluster_client,
                            instance_ref,
                            pending_tasks,
                            new_inflight,
                            task_counter + 1,
                            current_metadata=current_metadata,
                        )
                    return Behaviors.same()
                case TaskResult(value, _node_id):
                    if inflight:
                        first_key = next(iter(inflight))
                        caller = inflight[first_key]
                        caller.tell(TaskResult(value=value, node_id=node_id))
                        new_inflight = {k: v for k, v in inflight.items() if k != first_key}
                        if task_manager:
                            task_manager.tell(SlotFreed(node_id=node_id))
                        return active(
                            cluster_id,
                            provider_ref,
                            cluster_client,
                            instance_ref,
                            pending_tasks,
                            new_inflight,
                            task_counter,
                            current_metadata=current_metadata,
                        )
            return Behaviors.same()
        return Behaviors.receive(receive)

    return idle()
