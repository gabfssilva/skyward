from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from casty import ActorContext, ActorRef, Behavior, Behaviors
from loguru import logger

from skyward.actors.messages import (
    ClusterConnected,
    ClusterProvisioned,
    ClusterRequested,
    InstanceMetadata,
    NodeAvailable,
    NodeBecameReady,
    NodeLost,
    NodeUnavailable,
    PoolMsg,
    PoolStarted,
    PoolStopped,
    Provision,
    SetWorkerRef,
    ShutdownCompleted,
    ShutdownRequested,
    StartPool,
    StopPool,
    SubmitBroadcast,
    SubmitTask,
)
from skyward.actors.node import node_actor
from skyward.actors.task_manager import task_manager_actor

if TYPE_CHECKING:
    from skyward.actors.messages import ClusterId, NodeId
    from skyward.api.spec import PoolSpec


def pool_actor() -> Behavior[PoolMsg]:
    """A pool tells this story: idle → requesting → provisioning → ready → stopping."""

    def idle() -> Behavior[PoolMsg]:
        async def receive(
            ctx: ActorContext[PoolMsg], msg: PoolMsg,
        ) -> Behavior[PoolMsg]:
            match msg:
                case StartPool(
                    spec=spec, provider_config=_, provider_ref=provider_ref, reply_to=reply_to,
                ):
                    request_id = str(uuid.uuid4())
                    provider_name = spec.provider or "aws"
                    provider_ref.tell(ClusterRequested(
                        request_id=request_id,
                        provider=provider_name,
                        spec=spec,
                        reply_to=ctx.self,
                    ))
                    return requesting(spec, provider_ref, reply_to)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def requesting(
        spec: PoolSpec, provider_ref: ActorRef, reply_to: ActorRef,
    ) -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case ClusterProvisioned(cluster_id=cluster_id):
                    tm_ref = ctx.spawn(task_manager_actor(), "task-manager")

                    node_refs: dict[NodeId, ActorRef] = {}
                    for nid in range(spec.nodes):
                        ref = ctx.spawn(
                            node_actor(
                                node_id=nid, pool=ctx.self,
                                ssh_timeout=spec.ssh_timeout,
                                ssh_retry_interval=spec.ssh_retry_interval,
                            ),
                            f"node-{nid}",
                        )
                        ref.tell(Provision(cluster_id=cluster_id, provider_ref=provider_ref))
                        node_refs[nid] = ref

                    return provisioning(
                        spec, provider_ref, reply_to, cluster_id,
                        instances={}, node_refs=node_refs, tm_ref=tm_ref,
                        worker_refs=(),
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def provisioning(
        spec: PoolSpec,
        provider_ref: ActorRef,
        reply_to: ActorRef,
        cluster_id: ClusterId,
        instances: dict[NodeId, InstanceMetadata],
        node_refs: dict[NodeId, ActorRef],
        tm_ref: ActorRef,
        worker_refs: tuple[tuple[NodeId, ActorRef], ...],
    ) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="provisioning")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case ClusterConnected(worker_refs=refs, client=client):
                    ref_map = dict(refs)
                    for nid, node_ref in node_refs.items():
                        node_ref.tell(SetWorkerRef(worker_ref=ref_map.get(nid), client=client))
                    return provisioning(
                        spec, provider_ref, reply_to, cluster_id,
                        instances, node_refs, tm_ref, worker_refs=refs,
                    )
                case NodeBecameReady(node_id=nid, instance=instance):
                    new_instances = {**instances, nid: instance}
                    tm_ref.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=node_refs[nid],
                        slots=spec.concurrency,
                    ))
                    if len(new_instances) == spec.nodes:
                        reply_to.tell(PoolStarted(
                            cluster_id=cluster_id,
                            instances=tuple(new_instances[i] for i in range(spec.nodes)),
                        ))
                        return ready(
                            spec, provider_ref, cluster_id, new_instances,
                            reply_to, node_refs, tm_ref,
                            ready_nodes=frozenset(new_instances.keys()),
                            worker_refs=worker_refs,
                        )
                    return provisioning(
                        spec, provider_ref, reply_to, cluster_id,
                        new_instances, node_refs, tm_ref, worker_refs,
                    )
                case StopPool():
                    log.debug("StopPool received while provisioning")
            return Behaviors.same()
        return Behaviors.receive(receive)

    def ready(
        spec: PoolSpec,
        provider_ref: ActorRef,
        cluster_id: ClusterId,
        instances: dict[NodeId, InstanceMetadata],
        reply_to: ActorRef,
        node_refs: dict[NodeId, ActorRef],
        tm_ref: ActorRef,
        ready_nodes: frozenset[int],
        worker_refs: tuple[tuple[NodeId, ActorRef], ...],
    ) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="ready")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case SubmitTask(fn_bytes=fn_bytes, reply_to=task_reply):
                    tm_ref.tell(SubmitTask(fn_bytes=fn_bytes, reply_to=task_reply))
                    return Behaviors.same()
                case SubmitBroadcast(fn_bytes=fn_bytes, reply_to=bcast_reply):
                    tm_ref.tell(SubmitBroadcast(fn_bytes=fn_bytes, reply_to=bcast_reply))
                    return Behaviors.same()
                case ClusterConnected(worker_refs=refs, client=client):
                    ref_map = dict(refs)
                    for nid, node_ref in node_refs.items():
                        node_ref.tell(SetWorkerRef(worker_ref=ref_map.get(nid), client=client))
                    return ready(
                        spec, provider_ref, cluster_id, instances,
                        reply_to, node_refs, tm_ref, ready_nodes, worker_refs=refs,
                    )
                case NodeBecameReady(node_id=nid):
                    tm_ref.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=node_refs[nid],
                        slots=spec.concurrency,
                    ))
                    return ready(
                        spec, provider_ref, cluster_id, instances,
                        reply_to, node_refs, tm_ref,
                        ready_nodes=ready_nodes | {nid},
                        worker_refs=worker_refs,
                    )
                case NodeLost(node_id=nid):
                    tm_ref.tell(NodeUnavailable(node_id=nid))
                    return ready(
                        spec, provider_ref, cluster_id, instances,
                        reply_to, node_refs, tm_ref,
                        ready_nodes=ready_nodes - {nid},
                        worker_refs=worker_refs,
                    )
                case StopPool(reply_to=stop_reply):
                    log.debug(
                        "StopPool, sending ShutdownRequested for cluster {cid}",
                        cid=cluster_id,
                    )
                    provider_ref.tell(ShutdownRequested(
                        cluster_id=cluster_id,
                        reply_to=ctx.self,  # type: ignore[arg-type]
                    ))
                    return stopping(stop_reply)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def stopping(stop_reply: ActorRef) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="stopping")

        async def receive(_ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            log.debug("received: {msg}", msg=type(msg).__name__)
            match msg:
                case ShutdownCompleted(cluster_id=cid):
                    log.info("Cluster {cid} shutdown confirmed", cid=cid)
                    stop_reply.tell(PoolStopped())
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    return idle()
