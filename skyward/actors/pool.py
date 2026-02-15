from __future__ import annotations

import uuid

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    BroadcastResult,
    BroadcastTask,
    ClusterProvisioned,
    ClusterRequested,
    ExecuteResult,
    ExecuteTask,
    InstanceMetadata,
    NodeAvailable,
    NodeBecameReady,
    NodeLost,
    NodeUnavailable,
    PoolMsg,
    PoolStarted,
    PoolStopped,
    Provision,
    ShutdownCompleted,
    ShutdownRequested,
    StartPool,
    StopPool,
    SubmitBroadcast,
    SubmitTask,
)
from skyward.actors.node import node_actor
from skyward.actors.task_manager import task_manager_actor

type NodeId = int


def pool_actor() -> Behavior[PoolMsg]:

    def idle() -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case StartPool(
                    spec=spec,
                    provider_config=provider_config,
                    provider_ref=provider_ref,
                    reply_to=reply_to,
                ):
                    request_id = str(uuid.uuid4())
                    provider_name = spec.provider or "unknown"
                    provider_ref.tell(ClusterRequested(
                        request_id=request_id,
                        provider=provider_name,
                        spec=spec,
                        reply_to=ctx.self,
                    ))
                    return requesting(spec, provider_config, provider_ref, reply_to, request_id)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def requesting(spec, provider_config, provider_ref, reply_to, request_id) -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case ClusterProvisioned(cluster_id=cluster_id, provider=_):
                    tm_ref = ctx.spawn(task_manager_actor(), "task-manager")

                    node_refs: dict[NodeId, ActorRef] = {}
                    for nid in range(spec.nodes):
                        ref = ctx.spawn(
                            node_actor(node_id=nid, pool=ctx.self, task_manager=tm_ref),
                            f"node-{nid}",
                        )
                        ref.tell(Provision(
                            cluster_id=cluster_id,
                            provider_ref=provider_ref,
                            cluster_client=None,
                        ))
                        node_refs[nid] = ref

                    instances: dict[NodeId, InstanceMetadata] = {}
                    return provisioning(
                        spec,
                        provider_config,
                        provider_ref,
                        reply_to,
                        cluster_id,
                        instances,
                        node_refs,
                        tm_ref,
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def provisioning(
        spec, provider_config, provider_ref, reply_to, cluster_id, instances, node_refs, tm_ref
    ) -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case NodeBecameReady(node_id=nid, instance=instance):
                    new_instances = {**instances, nid: instance}
                    tm_ref.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=node_refs[nid],
                        slots=spec.concurrency if hasattr(spec, "concurrency") else 1,
                    ))
                    if len(new_instances) == spec.nodes:
                        reply_to.tell(PoolStarted(
                            cluster_id=cluster_id,
                            instances=tuple(new_instances[i] for i in range(spec.nodes)),
                        ))
                        return ready(
                            spec,
                            provider_ref,
                            cluster_id,
                            new_instances,
                            reply_to,
                            node_refs,
                            tm_ref,
                        )
                    return provisioning(
                        spec,
                        provider_config,
                        provider_ref,
                        reply_to,
                        cluster_id,
                        new_instances,
                        node_refs,
                        tm_ref,
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def ready(
        spec, provider_ref, cluster_id, instances, reply_to, node_refs, tm_ref
    ) -> Behavior[PoolMsg]:
        ready_nodes: set[int] = set(instances.keys())
        node_cycle = _round_robin(list(range(spec.nodes)))

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            nonlocal ready_nodes
            match msg:
                case SubmitTask(fn_bytes=fn_bytes, reply_to=task_reply):
                    tm_ref.tell(SubmitTask(fn_bytes=fn_bytes, reply_to=task_reply))
                    return Behaviors.same()
                case SubmitBroadcast(fn_bytes=fn_bytes, reply_to=bcast_reply):
                    tm_ref.tell(SubmitBroadcast(fn_bytes=fn_bytes, reply_to=bcast_reply))
                    return Behaviors.same()
                case ExecuteTask(fn=fn, args=args, kwargs=kwargs, node=node, reply_to=exec_reply):
                    target = node if node is not None else _next_ready(node_cycle, ready_nodes)
                    if target is not None:
                        result = fn(*args, **kwargs)
                        exec_reply.tell(ExecuteResult(value=result, node_id=target))
                    return Behaviors.same()
                case BroadcastTask(fn=fn, args=args, kwargs=kwargs, reply_to=bcast_reply):
                    values = tuple(fn(*args, **kwargs) for _ in range(spec.nodes))
                    bcast_reply.tell(BroadcastResult(values=values))
                    return Behaviors.same()
                case NodeBecameReady(node_id=nid, instance=instance):
                    ready_nodes.add(nid)
                    tm_ref.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=node_refs.get(nid, tm_ref),
                        slots=spec.concurrency if hasattr(spec, "concurrency") else 1,
                    ))
                    return Behaviors.same()
                case NodeLost(node_id=nid, reason=_):
                    ready_nodes.discard(nid)
                    tm_ref.tell(NodeUnavailable(node_id=nid))
                    return Behaviors.same()
                case StopPool(reply_to=stop_reply):
                    provider_ref.tell(ShutdownRequested(
                        cluster_id=cluster_id,
                        reply_to=ctx.self,  # type: ignore[arg-type]
                    ))
                    return stopping(stop_reply)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def stopping(stop_reply) -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case ShutdownCompleted():
                    stop_reply.tell(PoolStopped())
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    return idle()


def _round_robin(nodes: list[int]):
    idx = 0
    while True:
        yield nodes[idx % len(nodes)]
        idx += 1


def _next_ready(cycle, ready_nodes: set[int]) -> int | None:
    for _ in range(len(ready_nodes) + 1):
        n = next(cycle)
        if n in ready_nodes:
            return n
    return None
