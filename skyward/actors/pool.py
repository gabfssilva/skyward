from __future__ import annotations

import logging as _logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from casty import ActorContext, ActorRef, Behavior, Behaviors, ClusterClient

from skyward.actors.messages import (
    ClusterReady,
    CurrentNodeCount,
    DrainComplete,
    DrainNode,
    GetCurrentNodes,
    HeadAddressKnown,
    InstancesProvisioned,
    JoinCluster,
    NodeAvailable,
    NodeBecameReady,
    NodeInstance,
    NodeJoined,
    NodeLost,
    NodeUnavailable,
    PoolMsg,
    PoolStarted,
    PoolStopped,
    Provision,
    ProvisionFailed,
    ReconcilerNodeLost,
    RegisterPressureObserver,
    SpawnNodes,
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
    from skyward.api.model import Offer
    from skyward.api.spec import PoolSpec

_logging.getLogger("casty").setLevel(_logging.ERROR)


def _build_pool_info_json(
    node_id: int, spec: Any, cluster: Any, ni: NodeInstance,
    head_addr: str,
) -> str:
    from skyward.providers.pool_info import build_pool_info

    accel = ni.instance.offer.instance_type.accelerator
    accelerator_count = accel.count if accel else 1
    total_accelerators = accelerator_count * spec.nodes

    pool_info = build_pool_info(
        node=node_id,
        total_nodes=spec.nodes,
        accelerator_count=accelerator_count,
        total_accelerators=total_accelerators,
        head_addr=head_addr,
        head_port=29500,
        job_id=cluster.id,
        peers=[],
        accelerator_type=getattr(spec, "accelerator_name", None),
        placement_group=ni.network_interface or None,
        worker=0,
        workers_per_node=1,
    )
    return pool_info.model_dump_json()


def pool_actor() -> Behavior[PoolMsg]:
    """idle → requesting → provisioning → ready → stopping."""

    tunnel_map: dict[tuple[str, int], tuple[str, int]] = {}

    def address_resolver(addr: tuple[str, int]) -> tuple[str, int]:
        return tunnel_map.get(addr, addr)

    def idle() -> Behavior[PoolMsg]:
        async def receive(
            ctx: ActorContext[PoolMsg], msg: PoolMsg,
        ) -> Behavior[PoolMsg]:
            match msg:
                case StartPool(
                    spec=spec, provider_config=_, provider=provider,
                    offers=offers, reply_to=reply_to,
                ):
                    if not offers:
                        reply_to.tell(ProvisionFailed(reason="No offers available"))
                        return Behaviors.stopped()
                    offer, *remaining = offers
                    logger.bind(actor="pool").info(
                        "StartPool received: {nodes} nodes, accelerator={acc}, "
                        "offers={n}",
                        nodes=spec.nodes,
                        acc=getattr(spec, "accelerator", None),
                        n=len(offers),
                    )
                    ctx.pipe_to_self(
                        provider.prepare(spec, offer),
                        mapper=lambda cluster: ClusterReady(cluster=cluster),
                        on_failure=lambda err: ProvisionFailed(reason=str(err)),
                    )
                    return requesting(spec, provider, tuple(remaining), reply_to)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def requesting(
        spec: PoolSpec, provider: Any,
        remaining_offers: tuple[Offer, ...], reply_to: ActorRef,
    ) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="requesting")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case ProvisionFailed() as pf:
                    if remaining_offers:
                        offer, *rest = remaining_offers
                        log.warning(
                            "Prepare failed, trying next offer ({n} remaining)",
                            n=len(remaining_offers),
                        )
                        ctx.pipe_to_self(
                            provider.prepare(spec, offer),
                            mapper=lambda cluster: ClusterReady(cluster=cluster),
                            on_failure=lambda err: ProvisionFailed(reason=str(err)),
                        )
                        return requesting(spec, provider, tuple(rest), reply_to)
                    reply_to.tell(pf)
                    return Behaviors.stopped()
                case ClusterReady(cluster=cluster):
                    log.info("Cluster ready, provisioning {n} instances", n=spec.nodes)

                    if spec.volumes and cluster.mount_endpoint is None:
                        from skyward.providers.provider import Mountable

                        if not isinstance(provider, Mountable):
                            reply_to.tell(ProvisionFailed(
                                reason="Provider does not support volumes. "
                                       "Supported providers: AWS, GCP, RunPod.",
                            ))
                            return Behaviors.stopped()

                        async def _resolve_mount() -> Any:
                            endpoint = await provider.mount_endpoint(cluster)
                            from dataclasses import replace
                            return replace(cluster, mount_endpoint=endpoint)

                        ctx.pipe_to_self(
                            _resolve_mount(),
                            mapper=lambda c: ClusterReady(cluster=c),
                            on_failure=lambda err: ProvisionFailed(
                                reason=f"Volume mount setup failed: {err}",
                            ),
                        )
                        return requesting(spec, provider, remaining_offers, reply_to)

                    def _on_provision_error(err: Exception) -> InstancesProvisioned:
                        log.warning("Provision failed: {err}", err=err)
                        return InstancesProvisioned(instances=(), cluster=cluster)

                    ctx.pipe_to_self(
                        provider.provision(cluster, spec.nodes),
                        mapper=lambda result: InstancesProvisioned(
                            instances=result[1], cluster=result[0],
                        ),
                        on_failure=_on_provision_error,
                    )
                    return provisioning_instances(
                        spec, provider, cluster, reply_to,
                        remaining_offers=remaining_offers,
                    )
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
        remaining_offers: tuple[Offer, ...] = (),
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
                        remaining_offers=remaining_offers,
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
                        tm_ref = ctx.spawn(
                            task_manager_actor(),
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

                        def _on_retry_error(err: Exception) -> InstancesProvisioned:
                            log.warning("Retry provision failed: {err}", err=err)
                            return InstancesProvisioned(
                                instances=(), cluster=updated_cluster,
                            )

                        ctx.pipe_to_self(
                            _retry_provision(),
                            mapper=lambda result: InstancesProvisioned(
                                instances=result[1], cluster=result[0],
                            ),
                            on_failure=_on_retry_error,
                        )
                        return provisioning_instances(
                            spec, provider, updated_cluster, reply_to,
                            spawned=new_spawned, attempt=attempt + 1,
                            early_ready=early_ready,
                            remaining_offers=remaining_offers,
                        )

                    if remaining_offers:
                        offer, *rest = remaining_offers
                        log.warning(
                            "Exhausted {max} attempts, trying next offer "
                            "({n} remaining)",
                            max=spec.max_provision_attempts,
                            n=len(remaining_offers),
                        )
                        ctx.pipe_to_self(
                            provider.prepare(spec, offer),
                            mapper=lambda c: ClusterReady(cluster=c),
                            on_failure=lambda err: ProvisionFailed(
                                reason=str(err),
                            ),
                        )
                        return requesting(
                            spec, provider, tuple(rest), reply_to,
                        )

                    log.error(
                        "Exhausted {max} provision attempts and all offers, "
                        "only got {got}/{need} nodes",
                        max=spec.max_provision_attempts,
                        got=len(new_spawned), need=spec.nodes,
                    )
                    reply_to.tell(ProvisionFailed(
                        reason=(
                            f"Only provisioned {len(new_spawned)}"
                            f"/{spec.nodes} nodes after "
                            f"{spec.max_provision_attempts} attempts "
                            f"across all offers"
                        ),
                    ))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    async def _create_client(
        private_ip: str, casty_port: int, local_port: int,
    ) -> ClusterClient:
        from skyward.infra.worker import skyward_serializer

        tunnel_map[(private_ip, casty_port)] = ("127.0.0.1", local_port)
        client = ClusterClient(
            contact_points=[(private_ip, casty_port)],
            system_name="skyward",
            address_map=address_resolver,
            serializer=skyward_serializer(),
        )
        await client.__aenter__()
        return client

    def _update_tunnel(
        nbr: NodeBecameReady,
    ) -> None:
        tunnel_map[(nbr.private_ip, nbr.casty_port)] = ("127.0.0.1", nbr.local_port)

    def _send_join_cluster(
        client: ClusterClient,
        nbr: NodeBecameReady,
        node_ref: ActorRef,
        spec: Any,
        cluster: Any,
        head_addr: str,
    ) -> None:
        pool_json = _build_pool_info_json(
            nbr.node_id, spec, cluster, nbr.instance, head_addr,
        )
        image_env = dict(spec.image.env) if spec.image and spec.image.env else {}
        node_ref.tell(JoinCluster(
            client=client,
            pool_info_json=pool_json,
            env_vars=image_env,
        ))

    def provisioning(
        spec: PoolSpec,
        provider: Any,
        cluster: Any,
        reply_to: ActorRef,
        cluster_id: ClusterId,
        instances: dict[NodeId, NodeInstance],
        node_refs: dict[NodeId, ActorRef],
        tm_ref: ActorRef,
        head_addr: str | None = None,
        client: ClusterClient | None = None,
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
                        instances, node_refs, tm_ref,
                        head_addr=h.head_addr, client=client,
                    )
                case NodeBecameReady(node_id=nid, instance=meta):
                    new_instances = {**instances, nid: meta}
                    log.info(
                        "Node {nid} ready ({n}/{total})",
                        nid=nid, n=len(new_instances), total=spec.nodes,
                    )
                    effective_head = head_addr or msg.private_ip or ""

                    if client is None:
                        new_client = await _create_client(
                            msg.private_ip, msg.casty_port, msg.local_port,
                        )
                        log.info("ClusterClient created")
                    else:
                        _update_tunnel(msg)
                        new_client = client

                    _send_join_cluster(
                        new_client, msg, node_refs[nid],
                        spec, cluster, effective_head,
                    )
                    tm_ref.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=node_refs[nid],
                        slots=spec.worker.concurrency,
                    ))
                    if len(new_instances) == spec.nodes:
                        log.info("All {n} nodes ready, pool is operational", n=spec.nodes)
                        ctx.self.tell(PoolStarted(
                            cluster_id=cluster_id,
                            instances=tuple(new_instances[i] for i in range(spec.nodes)),
                            cluster=cluster,
                        ))
                        return ready(
                            spec, provider, cluster, cluster_id, new_instances,
                            reply_to, node_refs, tm_ref,
                            client=new_client,
                            ready_nodes=frozenset(new_instances.keys()),
                            head_addr=effective_head,
                        )
                    return provisioning(
                        spec, provider, cluster, reply_to, cluster_id,
                        new_instances, node_refs, tm_ref,
                        head_addr=head_addr, client=new_client,
                    )
                case StopPool():
                    log.debug("StopPool received while provisioning")
            return Behaviors.same()

        behavior: Behavior[PoolMsg] = Behaviors.receive(receive)
        if client is not None:
            async def _close(_ctx: ActorContext[PoolMsg]) -> None:
                with suppress(Exception):
                    await client.__aexit__(None, None, None)
            behavior = Behaviors.with_lifecycle(behavior, post_stop=_close)
        return behavior

    def ready(
        spec: PoolSpec,
        provider: Any,
        cluster: Any,
        cluster_id: ClusterId,
        instances: dict[NodeId, NodeInstance],
        reply_to: ActorRef,
        node_refs: dict[NodeId, ActorRef],
        tm_ref: ActorRef,
        client: ClusterClient,
        ready_nodes: frozenset[int],
        head_addr: str,
        reconciler_ref: ActorRef | None = None,
        instance_map: dict[NodeId, str] | None = None,
    ) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="ready")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case PoolStarted() as started:
                    reply_to.tell(started)
                    from skyward.actors.reconciler import reconciler_actor

                    min_n = spec.min_nodes or spec.nodes
                    max_n = spec.max_nodes or spec.nodes
                    inst_map = {nid: ni.instance.id for nid, ni in instances.items()}
                    rec_ref = ctx.spawn(
                        reconciler_actor(
                            pool=ctx.self,
                            provider=provider,
                            cluster=cluster,
                            min_nodes=min_n,
                            max_nodes=max_n,
                            initial_node_ids=frozenset(ready_nodes),
                            initial_instance_map=inst_map,
                            next_node_id=max(ready_nodes) + 1 if ready_nodes else spec.nodes,
                            tick_interval=spec.reconcile_tick_interval,
                        ),
                        "reconciler",
                    )
                    if spec.auto_scaling:
                        from skyward.actors.autoscaler import autoscaler_actor

                        assert spec.min_nodes is not None
                        assert spec.max_nodes is not None
                        as_ref = ctx.spawn(
                            autoscaler_actor(
                                min_nodes=spec.min_nodes,
                                max_nodes=spec.max_nodes,
                                reconciler=rec_ref,
                                slots_per_node=spec.worker.concurrency + 1,
                                initial_count=spec.nodes,
                                cooldown=spec.autoscale_cooldown,
                                scale_down_idle_seconds=spec.autoscale_idle_timeout,
                            ),
                            "autoscaler",
                        )
                        tm_ref.tell(RegisterPressureObserver(observer=as_ref))
                    return ready(
                        spec, provider, cluster, cluster_id, instances,
                        reply_to, node_refs, tm_ref, client=client,
                        ready_nodes=ready_nodes, head_addr=head_addr,
                        reconciler_ref=rec_ref, instance_map=inst_map,
                    )
                case SubmitTask() as task:
                    log.debug("Task submitted")
                    tm_ref.tell(task)
                    return Behaviors.same()
                case SubmitBroadcast() as bcast:
                    log.debug("Broadcast submitted to {n} nodes", n=len(ready_nodes))
                    tm_ref.tell(bcast)
                    return Behaviors.same()
                case NodeBecameReady(node_id=nid) as nbr:
                    _update_tunnel(nbr)
                    effective_head = head_addr or nbr.private_ip or ""
                    _send_join_cluster(
                        client, nbr, node_refs[nid],
                        spec, cluster, effective_head,
                    )
                    tm_ref.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=node_refs[nid],
                        slots=spec.worker.concurrency,
                    ))
                    if reconciler_ref is not None:
                        reconciler_ref.tell(NodeJoined(node_id=nid))
                    return ready(
                        spec, provider, cluster, cluster_id, instances,
                        reply_to, node_refs, tm_ref,
                        client=client,
                        ready_nodes=ready_nodes | {nid},
                        head_addr=head_addr,
                        reconciler_ref=reconciler_ref,
                        instance_map=instance_map,
                    )
                case NodeLost(node_id=nid):
                    log.warning(
                        "Node {nid} lost, {remaining} nodes remaining",
                        nid=nid, remaining=len(ready_nodes) - 1,
                    )
                    tm_ref.tell(NodeUnavailable(node_id=nid))
                    if reconciler_ref is not None:
                        reconciler_ref.tell(ReconcilerNodeLost(
                            node_id=nid, reason="node lost",
                        ))
                    if head_addr:
                        head_msg = HeadAddressKnown(
                            head_addr=head_addr,
                            casty_port=25520,
                            num_nodes=spec.max_nodes or spec.nodes,
                            worker_concurrency=spec.worker.concurrency,
                            worker_executor=spec.worker.executor,
                        )
                        node_refs[nid].tell(head_msg)
                    return ready(
                        spec, provider, cluster, cluster_id, instances,
                        reply_to, node_refs, tm_ref,
                        client=client,
                        ready_nodes=ready_nodes - {nid},
                        head_addr=head_addr,
                        reconciler_ref=reconciler_ref,
                        instance_map=instance_map,
                    )
                case SpawnNodes(
                    instances=new_instances, cluster=upd_cluster,
                    start_node_id=start_id,
                ):
                    log.info(
                        "Spawning {n} dynamic nodes from id {sid}",
                        n=len(new_instances), sid=start_id,
                    )
                    new_node_refs = dict(node_refs)
                    new_inst_map = dict(instance_map or {})
                    for i, inst in enumerate(new_instances):
                        nid = start_id + i
                        ref = ctx.spawn(
                            node_actor(
                                node_id=nid, pool=ctx.self,
                                ssh_timeout=spec.ssh_timeout,
                                ssh_retry_interval=spec.ssh_retry_interval,
                            ),
                            f"node-{nid}",
                        )
                        ref.tell(Provision(
                            cluster=upd_cluster, provider=provider,
                            instance=inst,
                        ))
                        if head_addr:
                            ref.tell(HeadAddressKnown(
                                head_addr=head_addr, casty_port=25520,
                                num_nodes=spec.max_nodes or spec.nodes,
                                worker_concurrency=spec.worker.concurrency,
                                worker_executor=spec.worker.resolved_executor,
                            ))
                        new_node_refs[nid] = ref
                        new_inst_map[nid] = inst.id
                    return ready(
                        spec, provider, cluster, cluster_id, instances,
                        reply_to, new_node_refs, tm_ref, client=client,
                        ready_nodes=ready_nodes, head_addr=head_addr,
                        reconciler_ref=reconciler_ref,
                        instance_map=new_inst_map,
                    )

                case DrainNode(node_id=nid, reply_to=drain_reply):
                    log.info("Draining node {nid}", nid=nid)
                    tm_ref.tell(NodeUnavailable(node_id=nid))
                    iid = (instance_map or {}).get(nid, "")
                    drain_reply.tell(DrainComplete(node_id=nid, instance_id=iid))
                    new_node_refs = {k: v for k, v in node_refs.items() if k != nid}
                    new_ready = ready_nodes - {nid}
                    return ready(
                        spec, provider, cluster, cluster_id, instances,
                        reply_to, new_node_refs, tm_ref, client=client,
                        ready_nodes=new_ready, head_addr=head_addr,
                        reconciler_ref=reconciler_ref,
                        instance_map=instance_map,
                    )

                case GetCurrentNodes(reply_to=query_reply):
                    query_reply.tell(CurrentNodeCount(
                        count=len(node_refs), ready=len(ready_nodes),
                    ))
                    return Behaviors.same()

                case StopPool(reply_to=stop_reply):
                    log.debug(
                        "StopPool, shutting down cluster {cid}",
                        cid=cluster_id,
                    )
                    instance_ids = tuple(
                        ni.instance.id for ni in instances.values()
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

        async def _close_client(_ctx: ActorContext[PoolMsg]) -> None:
            with suppress(Exception):
                await client.__aexit__(None, None, None)

        return Behaviors.with_lifecycle(
            Behaviors.receive(receive),
            post_stop=_close_client,
        )

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
