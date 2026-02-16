from __future__ import annotations

from typing import TYPE_CHECKING, Any

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    InstanceMetadata,
    NodeAvailable,
    NodeBecameReady,
    NodeLost,
    NodeUnavailable,
    PoolMsg,
    PoolStarted,
    PoolStopped,
    Provision,
    SetHeadAddr,
    StartPool,
    StopPool,
    SubmitBroadcast,
    SubmitTask,
    _ClusterReady,
    _InstancesProvisioned,
    _ShutdownDone,
)
from skyward.actors.node import node_actor
from skyward.actors.task_manager import task_manager_actor
from skyward.observability.logger import logger

if TYPE_CHECKING:
    from skyward.actors.messages import ClusterId, NodeId
    from skyward.api.spec import PoolSpec


def pool_actor() -> Behavior[PoolMsg]:
    """idle → requesting → provisioning → ready → stopping."""

    def idle() -> Behavior[PoolMsg]:
        async def receive(
            ctx: ActorContext[PoolMsg], msg: PoolMsg,
        ) -> Behavior[PoolMsg]:
            match msg:
                case StartPool(
                    spec=spec, provider_config=_, provider=provider, reply_to=reply_to,
                ):
                    ctx.pipe_to_self(
                        provider.prepare(spec),
                        mapper=lambda cluster: _ClusterReady(cluster=cluster),
                    )
                    return requesting(spec, provider, reply_to)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def requesting(
        spec: PoolSpec, provider: Any, reply_to: ActorRef,
    ) -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case _ClusterReady(cluster=cluster):
                    ctx.pipe_to_self(
                        provider.provision(cluster, spec.nodes),
                        mapper=lambda instances: _InstancesProvisioned(instances=instances),
                    )
                    return provisioning_instances(spec, provider, cluster, reply_to)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def provisioning_instances(
        spec: PoolSpec, provider: Any, cluster: Any, reply_to: ActorRef,
    ) -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case _InstancesProvisioned(instances=instances):
                    tm_ref = ctx.spawn(task_manager_actor(), "task-manager")

                    node_refs: dict[NodeId, ActorRef] = {}
                    for nid, instance in enumerate(instances):
                        ref = ctx.spawn(
                            node_actor(
                                node_id=nid, pool=ctx.self,
                                ssh_timeout=spec.ssh_timeout,
                                ssh_retry_interval=spec.ssh_retry_interval,
                            ),
                            f"node-{nid}",
                        )
                        ref.tell(Provision(
                            cluster=cluster,
                            provider=provider,
                            instance=instance,
                        ))
                        node_refs[nid] = ref

                    return provisioning(
                        spec, provider, cluster, reply_to, cluster.id,
                        instances={}, node_refs=node_refs, tm_ref=tm_ref,
                        head_addr_sent=False,
                    )
            return Behaviors.same()
        return Behaviors.receive(receive)

    def provisioning(
        spec: PoolSpec,
        provider: Any,
        cluster: Any,
        reply_to: ActorRef,
        cluster_id: ClusterId,
        instances: dict[NodeId, InstanceMetadata],
        node_refs: dict[NodeId, ActorRef],
        tm_ref: ActorRef,
        head_addr_sent: bool,
    ) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="provisioning")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case NodeBecameReady(node_id=0, instance=meta) if not head_addr_sent:
                    new_instances = {**instances, 0: meta}
                    tm_ref.tell(NodeAvailable(
                        node_id=0,
                        node_ref=node_refs[0],
                        slots=spec.concurrency,
                    ))

                    head_private = meta.private_ip or meta.ip
                    head_msg = SetHeadAddr(
                        head_addr=head_private,
                        casty_port=25520,
                        num_nodes=spec.nodes,
                        concurrency=spec.concurrency,
                    )
                    for nid, node_ref in node_refs.items():
                        if nid != 0:
                            node_ref.tell(head_msg)

                    if len(new_instances) == spec.nodes:
                        reply_to.tell(PoolStarted(
                            cluster_id=cluster_id,
                            instances=tuple(new_instances[i] for i in range(spec.nodes)),
                        ))
                        return ready(
                            spec, provider, cluster, cluster_id, new_instances,
                            reply_to, node_refs, tm_ref,
                            ready_nodes=frozenset(new_instances.keys()),
                            head_addr=head_private,
                        )
                    return provisioning(
                        spec, provider, cluster, reply_to, cluster_id,
                        new_instances, node_refs, tm_ref, head_addr_sent=True,
                    )
                case NodeBecameReady(node_id=nid, instance=instance):
                    new_instances = {**instances, nid: instance}
                    tm_ref.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=node_refs[nid],
                        slots=spec.concurrency,
                    ))
                    if len(new_instances) == spec.nodes:
                        head_meta = new_instances.get(0)
                        head_private = (head_meta.private_ip or head_meta.ip) if head_meta else ""
                        reply_to.tell(PoolStarted(
                            cluster_id=cluster_id,
                            instances=tuple(new_instances[i] for i in range(spec.nodes)),
                        ))
                        return ready(
                            spec, provider, cluster, cluster_id, new_instances,
                            reply_to, node_refs, tm_ref,
                            ready_nodes=frozenset(new_instances.keys()),
                            head_addr=head_private,
                        )
                    return provisioning(
                        spec, provider, cluster, reply_to, cluster_id,
                        new_instances, node_refs, tm_ref, head_addr_sent,
                    )
                case StopPool():
                    log.debug("StopPool received while provisioning")
            return Behaviors.same()
        return Behaviors.receive(receive)

    def ready(
        spec: PoolSpec,
        provider: Any,
        cluster: Any,
        cluster_id: ClusterId,
        instances: dict[NodeId, InstanceMetadata],
        reply_to: ActorRef,
        node_refs: dict[NodeId, ActorRef],
        tm_ref: ActorRef,
        ready_nodes: frozenset[int],
        head_addr: str,
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
                case NodeBecameReady(node_id=nid):
                    tm_ref.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=node_refs[nid],
                        slots=spec.concurrency,
                    ))
                    return ready(
                        spec, provider, cluster, cluster_id, instances,
                        reply_to, node_refs, tm_ref,
                        ready_nodes=ready_nodes | {nid},
                        head_addr=head_addr,
                    )
                case NodeLost(node_id=nid):
                    tm_ref.tell(NodeUnavailable(node_id=nid))
                    if head_addr:
                        head_msg = SetHeadAddr(
                            head_addr=head_addr,
                            casty_port=25520,
                            num_nodes=spec.nodes,
                            concurrency=spec.concurrency,
                        )
                        node_refs[nid].tell(head_msg)
                    return ready(
                        spec, provider, cluster, cluster_id, instances,
                        reply_to, node_refs, tm_ref,
                        ready_nodes=ready_nodes - {nid},
                        head_addr=head_addr,
                    )
                case StopPool(reply_to=stop_reply):
                    log.debug(
                        "StopPool, shutting down cluster {cid}",
                        cid=cluster_id,
                    )
                    instance_ids = tuple(
                        inst.id for inst in instances.values()
                    )

                    async def _shutdown() -> None:
                        await provider.terminate(instance_ids)
                        await provider.teardown(cluster)

                    ctx.pipe_to_self(
                        _shutdown(),
                        mapper=lambda _: _ShutdownDone(),
                    )
                    return stopping(stop_reply, cluster_id)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def stopping(stop_reply: ActorRef, cluster_id: str) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="stopping")

        async def receive(_ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            log.debug("received: {message}", message=type(msg).__name__)
            match msg:
                case _ShutdownDone():
                    log.info("Cluster {cid} shutdown confirmed", cid=cluster_id)
                    stop_reply.tell(PoolStopped())
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    return idle()
