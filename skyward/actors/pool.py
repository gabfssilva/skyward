from __future__ import annotations

from typing import TYPE_CHECKING, Any

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    ClusterReady,
    HeadAddressKnown,
    InstanceMetadata,
    InstancesProvisioned,
    NodeAvailable,
    NodeBecameReady,
    NodeLost,
    NodeUnavailable,
    PoolMsg,
    PoolStarted,
    PoolStopped,
    Provision,
    ProvisionFailed,
    StartPool,
    StopPool,
    SubmitBroadcast,
    SubmitTask,
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
                    logger.bind(actor="pool").info(
                        "StartPool received: {nodes} nodes, accelerator={acc}",
                        nodes=spec.nodes, acc=getattr(spec, "accelerator", None),
                    )
                    ctx.pipe_to_self(
                        provider.prepare(spec),
                        mapper=lambda cluster: ClusterReady(cluster=cluster),
                    )
                    return requesting(spec, provider, reply_to)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def requesting(
        spec: PoolSpec, provider: Any, reply_to: ActorRef,
    ) -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case ClusterReady(cluster=cluster):
                    log = logger.bind(actor="pool")
                    log.info("Cluster ready, provisioning {n} instances", n=spec.nodes)
                    ctx.pipe_to_self(
                        provider.provision(cluster, spec.nodes),
                        mapper=lambda result: InstancesProvisioned(
                            instances=result[1], cluster=result[0],
                        ),
                    )
                    return provisioning_instances(spec, provider, cluster, reply_to)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def provisioning_instances(
        spec: PoolSpec,
        provider: Any,
        cluster: Any,
        reply_to: ActorRef,
        spawned: dict[NodeId, ActorRef] | None = None,
        attempt: int = 1,
        early_ready: tuple[NodeBecameReady, ...] = (),
    ) -> Behavior[PoolMsg]:
        if spawned is None:
            spawned = {}
        log = logger.bind(actor="pool", state="provisioning_instances")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case NodeBecameReady() as nbr:
                    log.debug(
                        "Node {nid} ready early (buffered)",
                        nid=nbr.node_id,
                    )
                    return provisioning_instances(
                        spec, provider, cluster, reply_to,
                        spawned=spawned, attempt=attempt,
                        early_ready=(*early_ready, nbr),
                    )
                case InstancesProvisioned(
                    instances=instances, cluster=updated_cluster,
                ):
                    log.info(
                        "Instances provisioned ({n}), attempt "
                        "{attempt}/{max}",
                        n=len(instances), attempt=attempt,
                        max=spec.max_provision_attempts,
                    )
                    new_spawned = dict(spawned)
                    remaining = spec.nodes - len(new_spawned)
                    to_spawn = instances[:remaining]

                    for instance in to_spawn:
                        nid = len(new_spawned)
                        ref = ctx.spawn(
                            node_actor(
                                node_id=nid, pool=ctx.self,
                                ssh_timeout=spec.ssh_timeout,
                                ssh_retry_interval=spec.ssh_retry_interval,
                            ),
                            f"node-{nid}",
                        )
                        ref.tell(Provision(
                            cluster=updated_cluster,
                            provider=provider,
                            instance=instance,
                        ))
                        new_spawned[nid] = ref

                    if len(new_spawned) >= spec.nodes:
                        log.info(
                            "All {n} instances provisioned, "
                            "spawning task manager",
                            n=spec.nodes,
                        )
                        match spec.max_inflight:
                            case int(n):
                                resolved_inflight = n
                            case None:
                                resolved_inflight = (
                                    spec.nodes * spec.concurrency
                                )
                            case strategy:
                                resolved_inflight = strategy(
                                    spec.nodes, spec.concurrency,
                                )

                        tm_ref = ctx.spawn(
                            task_manager_actor(
                                max_inflight=resolved_inflight,
                            ),
                            "task-manager",
                        )
                        for nbr in early_ready:
                            ctx.self.tell(nbr)
                        return provisioning(
                            spec, provider, updated_cluster, reply_to,
                            updated_cluster.id,
                            instances={}, node_refs=new_spawned,
                            tm_ref=tm_ref,
                        )

                    still_needed = spec.nodes - len(new_spawned)
                    if attempt < spec.max_provision_attempts:
                        log.info(
                            "Got {got}/{need} nodes, retrying in "
                            "{delay}s (attempt {next}/{max})",
                            got=len(new_spawned), need=spec.nodes,
                            delay=spec.provision_retry_delay,
                            next=attempt + 1,
                            max=spec.max_provision_attempts,
                        )

                        async def _retry_provision() -> (
                            tuple[Any, tuple[Any, ...]]
                        ):
                            import asyncio as _asyncio

                            await _asyncio.sleep(
                                spec.provision_retry_delay,
                            )
                            return await provider.provision(
                                updated_cluster, still_needed,
                            )

                        ctx.pipe_to_self(
                            _retry_provision(),
                            mapper=lambda result: InstancesProvisioned(
                                instances=result[1], cluster=result[0],
                            ),
                        )
                        return provisioning_instances(
                            spec, provider, updated_cluster, reply_to,
                            spawned=new_spawned, attempt=attempt + 1,
                            early_ready=early_ready,
                        )

                    log.error(
                        "Exhausted {max} provision attempts, "
                        "only got {got}/{need} nodes",
                        max=spec.max_provision_attempts,
                        got=len(new_spawned), need=spec.nodes,
                    )
                    reply_to.tell(ProvisionFailed(
                        reason=(
                            f"Only provisioned {len(new_spawned)}"
                            f"/{spec.nodes} nodes after "
                            f"{spec.max_provision_attempts} attempts"
                        ),
                    ))
                    return Behaviors.stopped()
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
        head_addr: str | None = None,
    ) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="provisioning")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    log.info("Head address known: {addr}", addr=h.head_addr)
                    for nid, node_ref in node_refs.items():
                        if nid != 0:
                            node_ref.tell(h)
                    return provisioning(
                        spec, provider, cluster, reply_to, cluster_id,
                        instances, node_refs, tm_ref, head_addr=h.head_addr,
                    )
                case NodeBecameReady(node_id=nid, instance=meta):
                    new_instances = {**instances, nid: meta}
                    log.info(
                        "Node {nid} ready ({n}/{total})",
                        nid=nid, n=len(new_instances), total=spec.nodes,
                    )
                    tm_ref.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=node_refs[nid],
                        slots=spec.concurrency,
                    ))
                    if len(new_instances) == spec.nodes:
                        log.info("All {n} nodes ready, pool is operational", n=spec.nodes)
                        ctx.self.tell(PoolStarted(
                            cluster_id=cluster_id,
                            instances=tuple(new_instances[i] for i in range(spec.nodes)),
                        ))
                        return ready(
                            spec, provider, cluster, cluster_id, new_instances,
                            reply_to, node_refs, tm_ref,
                            ready_nodes=frozenset(new_instances.keys()),
                            head_addr=head_addr or "",
                        )
                    return provisioning(
                        spec, provider, cluster, reply_to, cluster_id,
                        new_instances, node_refs, tm_ref, head_addr=head_addr,
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
                case PoolStarted() as started:
                    reply_to.tell(started)
                    return Behaviors.same()
                case SubmitTask() as task:
                    log.debug("Task submitted")
                    tm_ref.tell(task)
                    return Behaviors.same()
                case SubmitBroadcast() as bcast:
                    log.debug("Broadcast submitted to {n} nodes", n=len(ready_nodes))
                    tm_ref.tell(bcast)
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
                    log.warning(
                        "Node {nid} lost, {remaining} nodes remaining",
                        nid=nid, remaining=len(ready_nodes) - 1,
                    )
                    tm_ref.tell(NodeUnavailable(node_id=nid))
                    if head_addr:
                        head_msg = HeadAddressKnown(
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
                        await provider.terminate(cluster, instance_ids)
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
