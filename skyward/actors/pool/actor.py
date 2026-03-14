from __future__ import annotations

import logging as _logging
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from casty import ActorContext, ActorRef, Behavior, Behaviors, ClusterClient

if TYPE_CHECKING:
    from skyward.actors.session.messages import SessionMsg

from skyward.actors.messages import (
    ClusterReady,
    CurrentNodeCount,
    DrainComplete,
    DrainNode,
    GetCurrentNodes,
    HeadAddressKnown,
    NodeAvailable,
    NodeBecameReady,
    NodeInstance,
    NodeJoined,
    NodeLost,
    NodeUnavailable,
    Provision,
    ReconcilerNodeLost,
    RegisterPressureObserver,
    SpawnNodes,
    SubmitBroadcast,
    SubmitTask,
)
from skyward.actors.node import node_actor
from skyward.actors.node.messages import JoinCluster
from skyward.actors.task_manager import task_manager_actor
from skyward.observability.logger import logger

from .messages import (
    InstancesProvisioned,
    PoolMsg,
    PoolStarted,
    PoolStopped,
    ProvisionFailed,
    StartPool,
    StopPool,
    _ShutdownDone,
)
from .state import PoolState

_logging.getLogger("casty").setLevel(_logging.ERROR)


def _build_pool_info_json(
    node_id: int, spec: Any, cluster: Any, ni: NodeInstance,
    head_addr: str,
) -> str:
    from skyward.providers.pool_info import build_pool_info

    accel = ni.instance.offer.instance_type.accelerator
    accelerator_count = accel.count if accel else 1
    total_accelerators = accelerator_count * spec.nodes.min

    wpn = spec.worker.concurrency if spec.worker.executor == "process" else 1

    pool_info = build_pool_info(
        node=node_id,
        total_nodes=spec.nodes.min,
        accelerator_count=accelerator_count,
        total_accelerators=total_accelerators,
        head_addr=head_addr,
        head_port=29500,
        job_id=cluster.id,
        peers=[],
        accelerator_type=getattr(spec, "accelerator_name", None),
        placement_group=ni.network_interface or None,
        worker=0,
        workers_per_node=wpn,
    )
    return pool_info.model_dump_json()


def pool_actor(
    *,
    session_ref: ActorRef[SessionMsg] | None = None,
    pool_name: str = "",
) -> Behavior[PoolMsg]:
    """idle -> requesting -> provisioning -> ready -> stopping."""

    def _notify_session(phase: str, nodes_ready: int, nodes_total: int) -> None:
        if session_ref is not None:
            from skyward.actors.session.messages import PoolStateChanged

            session_ref.tell(PoolStateChanged(
                name=pool_name, phase=phase,
                nodes_ready=nodes_ready, nodes_total=nodes_total,
            ))

    tunnel_map: dict[tuple[str, int], tuple[str, int]] = {}

    def address_resolver(addr: tuple[str, int]) -> tuple[str, int]:
        return tunnel_map.get(addr, addr)

    async def _create_client(
        private_ip: str, casty_port: int, local_port: int,
        tls: Any | None = None,
    ) -> ClusterClient:
        from skyward.infra.worker import skyward_serializer

        tunnel_map[(private_ip, casty_port)] = ("127.0.0.1", local_port)
        client = ClusterClient(
            contact_points=[(private_ip, casty_port)],
            system_name="skyward",
            address_map=address_resolver,
            serializer=skyward_serializer(),
            tls=tls,
        )
        await client.__aenter__()
        return client

    def _update_tunnel(nbr: NodeBecameReady) -> None:
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
        hooks = tuple(
            (p.name, p.around_app) for p in spec.plugins if p.around_app is not None
        )
        process_hooks = tuple(
            (p.name, p.around_process) for p in spec.plugins if p.around_process is not None
        )
        node_ref.tell(JoinCluster(
            client=client,
            pool_info_json=pool_json,
            env_vars=image_env,
            around_app_hooks=hooks,
            around_process_hooks=process_hooks,
        ))

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
                        nodes=spec.nodes.min,
                        acc=getattr(spec, "accelerator", None),
                        n=len(offers),
                    )

                    from skyward.infra.tls import ensure_ca, issue_client_config

                    ca = ensure_ca()
                    client_tls = issue_client_config(ca)

                    ctx.pipe_to_self(
                        provider.prepare(spec, offer),
                        mapper=lambda cluster: ClusterReady(cluster=cluster),
                        on_failure=lambda err: ProvisionFailed(reason=str(err)),
                    )
                    s = PoolState(
                        spec=spec, provider=provider, reply_to=reply_to,
                        remaining_offers=tuple(remaining),
                        ca=ca, client_tls=client_tls,
                    )
                    return requesting(s)
            return Behaviors.same()
        return Behaviors.receive(receive)

    def requesting(s: PoolState) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="requesting")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case ProvisionFailed() as pf:
                    if s.remaining_offers:
                        offer, *rest = s.remaining_offers
                        log.warning(
                            "Prepare failed, trying next offer ({n} remaining)",
                            n=len(s.remaining_offers),
                        )
                        ctx.pipe_to_self(
                            s.provider.prepare(s.spec, offer),
                            mapper=lambda cluster: ClusterReady(cluster=cluster),
                            on_failure=lambda err: ProvisionFailed(reason=str(err)),
                        )
                        return requesting(replace(s, remaining_offers=tuple(rest)))
                    s.reply_to.tell(pf)
                    return Behaviors.stopped()
                case ClusterReady(cluster=cluster):
                    transformed_image = s.spec.image
                    for plugin in s.spec.plugins:
                        if plugin.transform is not None:
                            transformed_image = plugin.transform(
                                transformed_image, cluster,
                            )
                    effective_spec = replace(s.spec, image=transformed_image)
                    prebaked = cluster.prebaked and transformed_image == s.spec.image
                    cluster = replace(cluster, spec=effective_spec, prebaked=prebaked)

                    log.info(
                        "Cluster ready, provisioning {n} instances",
                        n=effective_spec.nodes.min,
                    )

                    if effective_spec.volumes and cluster.resolved_volumes is None:
                        from skyward.providers.provider import Mountable

                        provider = s.provider

                        async def _resolve_volumes() -> Any:
                            resolved: list[tuple] = []
                            for vol in effective_spec.volumes:
                                if vol.storage is not None:
                                    st = await vol.storage.resolve()
                                    resolved.append((vol, st))
                                elif isinstance(provider, Mountable):
                                    st = await provider.storage(cluster)
                                    resolved.append((vol, st))
                                else:
                                    raise RuntimeError(
                                        f"Volume '{vol.bucket}' has no storage and provider "
                                        "does not support volumes."
                                    )
                            return replace(cluster, resolved_volumes=tuple(resolved))

                        ctx.pipe_to_self(
                            _resolve_volumes(),
                            mapper=lambda c: ClusterReady(cluster=c),
                            on_failure=lambda err: ProvisionFailed(
                                reason=f"Volume storage resolution failed: {err}",
                            ),
                        )
                        return requesting(replace(s, spec=effective_spec))

                    def _on_provision_error(err: Exception) -> InstancesProvisioned:
                        log.warning("Provision failed: {err}", err=err)
                        return InstancesProvisioned(instances=(), cluster=cluster)

                    ctx.pipe_to_self(
                        s.provider.provision(cluster, effective_spec.nodes.min),
                        mapper=lambda result: InstancesProvisioned(
                            instances=result[1], cluster=result[0],
                        ),
                        on_failure=_on_provision_error,
                    )
                    return provisioning_instances(replace(
                        s, spec=effective_spec, cluster=cluster,
                    ))
            return Behaviors.same()
        return Behaviors.receive(receive)

    def provisioning_instances(s: PoolState) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="provisioning_instances")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case NodeBecameReady() as nbr:
                    log.debug(
                        "Node {nid} ready early (buffered)",
                        nid=nbr.node_id,
                    )
                    return provisioning_instances(replace(
                        s, early_ready=(*s.early_ready, nbr),
                    ))
                case InstancesProvisioned(
                    instances=instances, cluster=updated_cluster,
                ):
                    log.info(
                        "Instances provisioned ({n}), attempt "
                        "{attempt}/{max}",
                        n=len(instances), attempt=s.attempt,
                        max=s.spec.max_provision_attempts,
                    )

                    new_spawned = dict(s.node_refs)
                    remaining = s.spec.nodes.min - len(new_spawned)
                    to_spawn = instances[:remaining]

                    updated_cluster = replace(
                        updated_cluster,
                        instances=(*updated_cluster.instances, *to_spawn),
                    )

                    for instance in to_spawn:
                        nid = len(new_spawned)
                        ref = ctx.spawn(
                            node_actor(
                                node_id=nid, pool=ctx.self,
                                ssh_timeout=s.spec.ssh_timeout,
                                ssh_retry_interval=s.spec.ssh_retry_interval,
                                ca=s.ca,
                            ),
                            f"node-{nid}",
                        )
                        ref.tell(Provision(
                            cluster=updated_cluster,
                            provider=s.provider,
                            instance=instance,
                        ))
                        new_spawned[nid] = ref

                    if len(new_spawned) >= s.spec.nodes.min:
                        log.info(
                            "All {n} instances provisioned, "
                            "spawning task manager",
                            n=s.spec.nodes.min,
                        )
                        tm_ref = ctx.spawn(
                            task_manager_actor(),
                            "task-manager",
                        )
                        for nbr in s.early_ready:
                            ctx.self.tell(nbr)
                        _notify_session("provisioning", 0, s.spec.nodes.min)
                        return provisioning(replace(
                            s,
                            cluster=updated_cluster,
                            cluster_id=updated_cluster.id,
                            node_refs=new_spawned,
                            tm_ref=tm_ref,
                            early_ready=(),
                        ))

                    still_needed = s.spec.nodes.min - len(new_spawned)
                    if s.attempt < s.spec.max_provision_attempts:
                        log.info(
                            "Got {got}/{need} nodes, retrying in "
                            "{delay}s (attempt {next}/{max})",
                            got=len(new_spawned), need=s.spec.nodes.min,
                            delay=s.spec.provision_retry_delay,
                            next=s.attempt + 1,
                            max=s.spec.max_provision_attempts,
                        )

                        provision_retry_delay = s.spec.provision_retry_delay
                        provider = s.provider

                        async def _retry_provision() -> (
                            tuple[Any, tuple[Any, ...]]
                        ):
                            import asyncio as _asyncio

                            await _asyncio.sleep(provision_retry_delay)
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
                        return provisioning_instances(replace(
                            s,
                            cluster=updated_cluster,
                            node_refs=new_spawned,
                            attempt=s.attempt + 1,
                        ))

                    if s.remaining_offers:
                        offer, *rest = s.remaining_offers
                        log.warning(
                            "Exhausted {max} attempts, trying next offer "
                            "({n} remaining)",
                            max=s.spec.max_provision_attempts,
                            n=len(s.remaining_offers),
                        )
                        ctx.pipe_to_self(
                            s.provider.prepare(s.spec, offer),
                            mapper=lambda c: ClusterReady(cluster=c),
                            on_failure=lambda err: ProvisionFailed(
                                reason=str(err),
                            ),
                        )
                        return requesting(replace(
                            s, remaining_offers=tuple(rest),
                        ))

                    log.error(
                        "Exhausted {max} provision attempts and all offers, "
                        "only got {got}/{need} nodes",
                        max=s.spec.max_provision_attempts,
                        got=len(new_spawned), need=s.spec.nodes.min,
                    )
                    s.reply_to.tell(ProvisionFailed(
                        reason=(
                            f"Only provisioned {len(new_spawned)}"
                            f"/{s.spec.nodes.min} nodes after "
                            f"{s.spec.max_provision_attempts} attempts "
                            f"across all offers"
                        ),
                    ))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def provisioning(s: PoolState) -> Behavior[PoolMsg]:
        assert s.tm_ref is not None
        tm = s.tm_ref
        log = logger.bind(actor="pool", state="provisioning")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    log.info("Head address known: {addr}", addr=h.head_addr)
                    for nid, node_ref in s.node_refs.items():
                        if nid != 0:
                            node_ref.tell(h)
                    return provisioning(replace(s, head_addr=h.head_addr))
                case NodeBecameReady(node_id=nid, instance=meta):
                    new_instances = {**s.instances, nid: meta}
                    log.info(
                        "Node {nid} ready ({n}/{total})",
                        nid=nid, n=len(new_instances), total=s.spec.nodes.min,
                    )
                    effective_head = s.head_addr or msg.private_ip or ""

                    if s.client is None:
                        new_client = await _create_client(
                            msg.private_ip, msg.casty_port, msg.local_port,
                            tls=s.client_tls,
                        )
                        log.info("ClusterClient created")
                    else:
                        _update_tunnel(msg)
                        new_client = s.client

                    _send_join_cluster(
                        new_client, msg, s.node_refs[nid],
                        s.spec, s.cluster, effective_head,
                    )
                    tm.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=s.node_refs[nid],
                        slots=s.spec.worker.concurrency,
                    ))
                    ready_threshold = s.spec.nodes.desired or s.spec.nodes.min

                    if len(new_instances) >= ready_threshold:
                        if len(new_instances) == s.spec.nodes.min:
                            log.info("All {n} nodes ready, pool is operational", n=s.spec.nodes.min)
                        else:
                            log.info(
                                "{n}/{total} nodes ready (desired met), pool is operational",
                                n=len(new_instances), total=s.spec.nodes.min,
                            )
                        ctx.self.tell(PoolStarted(
                            cluster_id=s.cluster_id,
                            instances=tuple(new_instances.values()),
                            cluster=s.cluster,
                        ))
                        _notify_session("ready", len(new_instances), s.spec.nodes.min)
                        return ready(replace(
                            s,
                            instances=new_instances,
                            client=new_client,
                            ready_nodes=frozenset(new_instances.keys()),
                            head_addr=effective_head,
                        ))
                    return provisioning(replace(
                        s,
                        instances=new_instances,
                        client=new_client,
                    ))
                case StopPool():
                    log.debug("StopPool received while provisioning")
            return Behaviors.same()

        behavior: Behavior[PoolMsg] = Behaviors.receive(receive)
        if (client := s.client) is not None:
            async def _close(_ctx: ActorContext[PoolMsg]) -> None:
                with suppress(Exception):
                    await client.__aexit__(None, None, None)
            behavior = Behaviors.with_lifecycle(behavior, post_stop=_close)
        return behavior

    def ready(s: PoolState) -> Behavior[PoolMsg]:
        assert s.tm_ref is not None
        assert s.client is not None
        tm = s.tm_ref
        client = s.client
        log = logger.bind(actor="pool", state="ready")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case PoolStarted() as started:
                    s.reply_to.tell(started)
                    from skyward.actors.reconciler import reconciler_actor

                    min_n = s.spec.nodes.min
                    max_n = s.spec.nodes.max or s.spec.nodes.min
                    inst_map = {nid: ni.instance.id for nid, ni in s.instances.items()}
                    rec_ref = ctx.spawn(
                        reconciler_actor(
                            pool=ctx.self,
                            provider=s.provider,
                            cluster=s.cluster,
                            min_nodes=min_n,
                            max_nodes=max_n,
                            initial_node_ids=frozenset(s.ready_nodes),
                            initial_instance_map=inst_map,
                            next_node_id=max(s.ready_nodes) + 1 if s.ready_nodes else s.spec.nodes.min,
                            tick_interval=s.spec.reconcile_tick_interval,
                        ),
                        "reconciler",
                    )
                    if s.spec.nodes.auto_scaling:
                        from skyward.actors.autoscaler import autoscaler_actor

                        assert s.spec.nodes.max is not None
                        as_ref = ctx.spawn(
                            autoscaler_actor(
                                min_nodes=s.spec.nodes.min,
                                max_nodes=s.spec.nodes.max,
                                reconciler=rec_ref,
                                slots_per_node=s.spec.worker.concurrency + 1,
                                initial_count=s.spec.nodes.min,
                                cooldown=s.spec.autoscale_cooldown,
                                scale_down_idle_seconds=s.spec.autoscale_idle_timeout,
                            ),
                            "autoscaler",
                        )
                        tm.tell(RegisterPressureObserver(observer=as_ref))
                    return ready(replace(
                        s,
                        reconciler_ref=rec_ref,
                        instance_map=inst_map,
                    ))
                case SubmitTask() as task:
                    log.debug("Task submitted")
                    tm.tell(task)
                    return Behaviors.same()
                case SubmitBroadcast() as bcast:
                    log.debug("Broadcast submitted to {n} nodes", n=len(s.ready_nodes))
                    tm.tell(bcast)
                    return Behaviors.same()
                case NodeBecameReady(node_id=nid) as nbr:
                    _update_tunnel(nbr)
                    effective_head = s.head_addr or nbr.private_ip or ""
                    _send_join_cluster(
                        client, nbr, s.node_refs[nid],
                        s.spec, s.cluster, effective_head,
                    )
                    tm.tell(NodeAvailable(
                        node_id=nid,
                        node_ref=s.node_refs[nid],
                        slots=s.spec.worker.concurrency,
                    ))
                    if s.reconciler_ref is not None:
                        s.reconciler_ref.tell(NodeJoined(node_id=nid))
                    _notify_session("ready", len(s.ready_nodes | {nid}), s.spec.nodes.min)
                    return ready(replace(
                        s, ready_nodes=s.ready_nodes | {nid},
                    ))
                case NodeLost(node_id=nid):
                    log.warning(
                        "Node {nid} lost, {remaining} nodes remaining",
                        nid=nid, remaining=len(s.ready_nodes) - 1,
                    )
                    tm.tell(NodeUnavailable(node_id=nid))
                    if s.reconciler_ref is not None:
                        s.reconciler_ref.tell(ReconcilerNodeLost(
                            node_id=nid, reason="node lost",
                        ))
                    if s.head_addr:
                        head_msg = HeadAddressKnown(
                            head_addr=s.head_addr,
                            casty_port=25520,
                            num_nodes=s.spec.nodes.max or s.spec.nodes.min,
                            worker_concurrency=s.spec.worker.concurrency,
                            worker_executor=s.spec.worker.resolved_executor,
                        )
                        s.node_refs[nid].tell(head_msg)
                    _notify_session("ready", len(s.ready_nodes - {nid}), s.spec.nodes.min)
                    return ready(replace(
                        s, ready_nodes=s.ready_nodes - {nid},
                    ))
                case SpawnNodes(
                    instances=new_instances, cluster=upd_cluster,
                    start_node_id=start_id,
                ):
                    log.info(
                        "Spawning {n} dynamic nodes from id {sid}",
                        n=len(new_instances), sid=start_id,
                    )
                    new_node_refs = dict(s.node_refs)
                    new_inst_map = dict(s.instance_map or {})
                    for i, inst in enumerate(new_instances):
                        nid = start_id + i
                        ref = ctx.spawn(
                            node_actor(
                                node_id=nid, pool=ctx.self,
                                ssh_timeout=s.spec.ssh_timeout,
                                ssh_retry_interval=s.spec.ssh_retry_interval,
                                ca=s.ca,
                            ),
                            f"node-{nid}",
                        )
                        ref.tell(Provision(
                            cluster=upd_cluster, provider=s.provider,
                            instance=inst,
                        ))
                        if s.head_addr:
                            ref.tell(HeadAddressKnown(
                                head_addr=s.head_addr, casty_port=25520,
                                num_nodes=s.spec.nodes.max or s.spec.nodes.min,
                                worker_concurrency=s.spec.worker.concurrency,
                                worker_executor=s.spec.worker.resolved_executor,
                            ))
                        new_node_refs[nid] = ref
                        new_inst_map[nid] = inst.id
                    return ready(replace(
                        s,
                        node_refs=new_node_refs,
                        instance_map=new_inst_map,
                    ))

                case DrainNode(node_id=nid, reply_to=drain_reply):
                    log.info("Draining node {nid}", nid=nid)
                    tm.tell(NodeUnavailable(node_id=nid))
                    iid = (s.instance_map or {}).get(nid, "")
                    drain_reply.tell(DrainComplete(node_id=nid, instance_id=iid))
                    new_node_refs = {k: v for k, v in s.node_refs.items() if k != nid}
                    return ready(replace(
                        s,
                        node_refs=new_node_refs,
                        ready_nodes=s.ready_nodes - {nid},
                    ))

                case GetCurrentNodes(reply_to=query_reply):
                    query_reply.tell(CurrentNodeCount(
                        count=len(s.node_refs), ready=len(s.ready_nodes),
                    ))
                    return Behaviors.same()

                case StopPool(reply_to=stop_reply):
                    log.debug(
                        "StopPool, shutting down cluster {cid}",
                        cid=s.cluster_id,
                    )
                    instance_ids = tuple(
                        ni.instance.id for ni in s.instances.values()
                    )
                    provider = s.provider
                    cluster = s.cluster

                    async def _shutdown() -> None:
                        await provider.terminate(cluster, instance_ids)
                        await provider.teardown(cluster)

                    ctx.pipe_to_self(
                        _shutdown(),
                        mapper=lambda _: _ShutdownDone(),
                    )
                    _notify_session("stopping", 0, s.spec.nodes.min)
                    return stopping(stop_reply, s.cluster_id)
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
