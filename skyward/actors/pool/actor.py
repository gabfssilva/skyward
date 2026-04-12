from __future__ import annotations

import logging as _logging
import time
from contextlib import suppress
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from casty import ActorContext, ActorRef, Behavior, Behaviors, ClusterClient

if TYPE_CHECKING:
    from skyward.actors.session.messages import SessionMsg

from skyward.actors.messages import (
    ClusterReady,
    CurrentNodeCount,
    DrainComplete,
    GetCurrentNodes,
    GetPoolSnapshot,
    HeadAddressKnown,
    NodeActivated,
    NodeAvailable,
    NodeBecameReady,
    NodeBecameUnready,
    NodeConnected,
    NodeExhausted,
    NodeInstance,
    NodeJoined,
    NodeLost,
    NodeUnavailable,
    Provision,
    ReconcilerNodeLost,
    ReconciliationExhausted,
    RegisterPressureObserver,
    RequestScaleDown,
    RequestScaleUp,
    ScaleDownComplete,
    ScaleUpComplete,
    SubmitBroadcast,
    SubmitTask,
)
from skyward.actors.node import node_actor
from skyward.actors.node.messages import JoinCluster
from skyward.actors.snapshot import NodeStatus, PoolPhase, ScalingSnapshot
from skyward.actors.task_manager import task_manager_actor
from skyward.core.model import Cluster
from skyward.observability.logger import logger

from .messages import (
    InstancesProvisioned,
    PoolMsg,
    PoolStarted,
    PoolStopped,
    ProvisionFailed,
    RecoverPool,
    StartPool,
    StopPool,
    _ShutdownDone,
)
from .state import PoolState, build_pool_snapshot

_logging.getLogger("casty").setLevel(_logging.ERROR)


def _build_pool_info_json(
    node_id: int, spec: Any, cluster: Any, ni: NodeInstance,
    head_addr: str,
) -> str:
    from skyward.providers.pool_info import build_pool_info

    accel = ni.instance.offer.instance_type.accelerator
    accelerator_count = accel.count if accel else 1
    total_accelerators = accelerator_count * spec.nodes.desired

    wpn = spec.worker.concurrency if spec.worker.executor == "process" else 1

    pool_info = build_pool_info(
        node=node_id,
        total_nodes=spec.nodes.desired,
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


async def _shutdown_cluster(
    provider: Any, cluster: Any, instance_ids: tuple[str, ...],
) -> None:
    if instance_ids and cluster is not None:
        try:
            await provider.terminate(cluster, instance_ids)
        except Exception:
            logger.bind(actor="pool").error(
                "CRITICAL: FAILED to terminate instances "
                "{ids} — they may still be running!",
                ids=instance_ids,
            )
    if cluster is not None:
        with suppress(Exception):
            await provider.teardown(cluster)


def pool_actor(
    *,
    session_ref: ActorRef[SessionMsg] | None = None,
    pool_name: str = "",
) -> Behavior[PoolMsg]:
    """idle -> requesting -> provisioning_instances -> provisioning -> ready -> stopping."""

    tunnel_map: dict[tuple[str, int], tuple[str, int]] = {}

    def address_resolver(addr: tuple[str, int]) -> tuple[str, int]:
        return tunnel_map.get(addr, addr)

    async def _create_client(
        private_ip: str, casty_port: int, local_port: int,
        tls: Any | None = None,
    ) -> ClusterClient:
        from skyward.infra.worker import skyward_serializer

        tunnel_map[(private_ip, casty_port)] = ("127.0.0.1", local_port)
        try:
            client = ClusterClient(
                contact_points=[(private_ip, casty_port)],
                system_name="skyward",
                address_map=address_resolver,
                serializer=skyward_serializer(),
                tls=tls,
            )
            await client.__aenter__()
        except Exception:
            logger.exception("ClusterClient creation failed")
            raise
        return client

    def _update_tunnel(nbr: NodeBecameReady) -> None:
        tunnel_map[(nbr.private_ip, nbr.casty_port)] = ("127.0.0.1", nbr.local_port)

    async def _create_standalone_client(
        private_ip: str, casty_port: int, local_port: int,
        tls: Any | None = None,
    ) -> ClusterClient:
        from skyward.infra.worker import skyward_serializer

        local_map: dict[tuple[str, int], tuple[str, int]] = {
            (private_ip, casty_port): ("127.0.0.1", local_port),
        }
        try:
            client = ClusterClient(
                contact_points=[(private_ip, casty_port)],
                system_name="skyward",
                address_map=lambda addr, _m=local_map: _m.get(addr, addr),
                serializer=skyward_serializer(),
                tls=tls,
            )
            await client.__aenter__()
        except Exception:
            logger.exception("Standalone ClusterClient creation failed")
            raise
        return client

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

    def _spawn_node(
        ctx: ActorContext[PoolMsg],
        nid: int,
        spec: Any,
        cluster: Any,
        provider: Any,
        instance: Any,
        ca: Any,
        head_addr: str | None,
    ) -> ActorRef:
        ref = ctx.spawn(
            node_actor(
                node_id=nid, pool=ctx.self,
                ssh_timeout=spec.ssh_timeout,
                ssh_retry_interval=spec.ssh_retry_interval,
                poll_timeout=spec.provision_timeout,
                bootstrap_timeout=spec.bootstrap_timeout,
                ca=ca,
            ),
            f"node-{nid}",
        )
        ref.tell(Provision(
            cluster=cluster, provider=provider, instance=instance,
        ))
        if spec.cluster and head_addr:
            ref.tell(HeadAddressKnown(
                head_addr=head_addr, casty_port=25520,
                num_nodes=spec.nodes.max or spec.nodes.desired,
                worker_concurrency=spec.worker.concurrency,
                worker_executor=spec.worker.resolved_executor,
            ))
        return ref

    def _collect_instance_ids(s: PoolState) -> tuple[str, ...]:
        known_iids = {ni.instance.id for ni in s.instances.values()}
        known_iids |= {iid for iid in s.instance_map.values() if iid}
        return tuple(known_iids)

    def _resolve_dead_iid(s: PoolState, nid: int) -> str | None:
        dead_ni = s.instances.get(nid)
        dead_iid = dead_ni.instance.id if dead_ni is not None else None
        if dead_iid is None:
            dead_iid = s.instance_map.get(nid)
        if dead_iid is None and s.cluster is not None and nid < len(s.cluster.instances):
            dead_iid = s.cluster.instances[nid].id
        return dead_iid

    def _prune_cluster_instance(cluster: Cluster | None, iid: str | None) -> Cluster | None:
        if cluster is None or iid is None:
            return cluster
        return replace(
            cluster,
            instances=tuple(i for i in cluster.instances if i.id != iid),
        )

    # ── idle ──────────────────────────────────────────────────────

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
                        nodes=spec.nodes.desired,
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
                        pool_started_at=time.monotonic(),
                    )
                    return requesting(s)

                case RecoverPool(
                    spec=spec, provider=provider, cluster=cluster,
                    instances=instances, reply_to=reply_to,
                ):
                    logger.bind(actor="pool").info(
                        "RecoverPool: recovering {n} instances for cluster {cid}",
                        n=len(instances), cid=cluster.id,
                    )
                    if not instances:
                        reply_to.tell(ProvisionFailed(reason="No alive instances to recover"))
                        return Behaviors.stopped()

                    from skyward.infra.tls import ensure_ca, issue_client_config

                    ca = ensure_ca()
                    client_tls = issue_client_config(ca)

                    new_refs: MappingProxyType[int, ActorRef] = MappingProxyType({})
                    for instance in instances:
                        nid = len(new_refs)
                        ref = _spawn_node(
                            ctx, nid, spec, cluster, provider, instance, ca, None,
                        )
                        new_refs = MappingProxyType({**new_refs, nid: ref})

                    tm_ref = ctx.spawn(task_manager_actor(), "task-manager")

                    return provisioning(PoolState(
                        spec=spec, provider=provider, reply_to=reply_to,
                        ca=ca, client_tls=client_tls,
                        pool_started_at=time.monotonic(),
                        cluster=cluster, cluster_id=cluster.id,
                        node_refs=new_refs, tm_ref=tm_ref,
                    ))

                case StopPool(reply_to=stop_reply):
                    stop_reply.tell(PoolStopped())
                    return Behaviors.stopped()

            return Behaviors.same()
        return Behaviors.receive(receive)

    # ── requesting ────────────────────────────────────────────────

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
                    spec_with_provider_plugins = replace(
                        s.spec, plugins=cluster.spec.plugins,
                    ) if cluster.spec.plugins != s.spec.plugins else s.spec

                    transformed_image = spec_with_provider_plugins.image
                    for plugin in spec_with_provider_plugins.plugins:
                        if plugin.transform is not None:
                            transformed_image = plugin.transform(
                                transformed_image, cluster,
                            )
                    effective_spec = replace(spec_with_provider_plugins, image=transformed_image)
                    prebaked = cluster.prebaked and transformed_image == s.spec.image
                    cluster = replace(cluster, spec=effective_spec, prebaked=prebaked)

                    log.info(
                        "Cluster ready, provisioning {n} instances",
                        n=effective_spec.nodes.desired,
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

                    remaining = effective_spec.nodes.desired - len(s.node_refs)
                    ctx.pipe_to_self(
                        s.provider.provision(cluster, remaining),
                        mapper=lambda result: InstancesProvisioned(
                            instances=result[1], cluster=result[0],
                        ),
                        on_failure=_on_provision_error,
                    )
                    return provisioning_instances(replace(
                        s, spec=effective_spec, cluster=cluster,
                    ))
                case StopPool(reply_to=stop_reply):
                    log.debug("StopPool during requesting")
                    s.reply_to.tell(ProvisionFailed(reason="Interrupted"))
                    _iids = (
                        tuple(inst.id for inst in s.cluster.instances)
                        if s.cluster is not None else ()
                    )
                    ctx.pipe_to_self(
                        _shutdown_cluster(s.provider, s.cluster, _iids),
                        mapper=lambda _: _ShutdownDone(),
                    )
                    return stopping(stop_reply, s.cluster_id or "")

                case GetPoolSnapshot(reply_to=snap_reply):
                    snap_reply.tell(build_pool_snapshot(s, pool_name))
                    return Behaviors.same()
            return Behaviors.same()
        return Behaviors.receive(receive)

    # ── provisioning_instances ────────────────────────────────────

    def provisioning_instances(s: PoolState) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="provisioning_instances")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case NodeConnected(instance=meta):
                    iid = meta.instance.id
                    new_statuses = MappingProxyType({**s.node_statuses, iid: NodeStatus.SSH})
                    return provisioning_instances(replace(s, node_statuses=new_statuses))
                case NodeBecameReady():
                    log.debug(
                        "Node {nid} ready early (ignored during provisioning_instances)",
                        nid=msg.node_id,
                    )
                    return Behaviors.same()
                case InstancesProvisioned(
                    instances=instances, cluster=updated_cluster,
                ):
                    log.info(
                        "Instances provisioned ({n}), attempt "
                        "{attempt}/{max}",
                        n=len(instances), attempt=s.attempt,
                        max=s.spec.max_provision_attempts,
                    )

                    new_spawned = s.node_refs
                    remaining = s.spec.nodes.desired - len(new_spawned)
                    to_spawn = instances[:remaining]

                    updated_cluster = replace(
                        updated_cluster,
                        instances=(*updated_cluster.instances, *to_spawn),
                    )

                    for instance in to_spawn:
                        nid = len(new_spawned)
                        ref = _spawn_node(
                            ctx, nid, s.spec, updated_cluster,
                            s.provider, instance, s.ca, None,
                        )
                        new_spawned = MappingProxyType({**new_spawned, nid: ref})

                    min_needed = s.spec.nodes.min or s.spec.nodes.desired

                    if len(new_spawned) >= min_needed:
                        log.info(
                            "Provisioned {got}/{desired} instances "
                            "(min={min}), spawning task manager",
                            got=len(new_spawned),
                            desired=s.spec.nodes.desired,
                            min=min_needed,
                        )
                        tm_ref = ctx.spawn(
                            task_manager_actor(
                                retry_on_interruption=s.spec.retry_on_interruption,
                            ),
                            "task-manager",
                        )

                        from skyward.actors.reconciler import reconciler_actor

                        min_n = s.spec.nodes.min or s.spec.nodes.desired
                        max_n = s.spec.nodes.max or s.spec.nodes.desired
                        rec_ref = ctx.spawn(
                            reconciler_actor(
                                pool=ctx.self,
                                min_nodes=min_n,
                                max_nodes=max_n,
                                initial_node_ids=frozenset(new_spawned.keys()),
                                tick_interval=s.spec.reconcile_tick_interval,
                                max_provision_retries=s.spec.max_provision_attempts,
                            ),
                            "reconciler",
                        )

                        return provisioning(replace(
                            s,
                            cluster=updated_cluster,
                            cluster_id=updated_cluster.id,
                            node_refs=new_spawned,
                            tm_ref=tm_ref,
                            reconciler_ref=rec_ref,
                        ))

                    still_needed = s.spec.nodes.desired - len(new_spawned)
                    got_partial = len(instances) > 0

                    if s.attempt < s.spec.max_provision_attempts:
                        log.info(
                            "Got {got}/{need} nodes, retrying in "
                            "{delay}s (attempt {next}/{max})",
                            got=len(new_spawned), need=s.spec.nodes.desired,
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
                        if got_partial:
                            log.info(
                                "Got {got}/{need} nodes from this offer, "
                                "trying next offer for remaining "
                                "({n} offers left)",
                                got=len(new_spawned),
                                need=s.spec.nodes.desired,
                                n=len(s.remaining_offers),
                            )
                        else:
                            log.warning(
                                "Exhausted {max} attempts, trying next "
                                "offer ({n} remaining)",
                                max=s.spec.max_provision_attempts,
                                n=len(s.remaining_offers),
                            )

                        stale_iids = tuple(
                            inst.id for inst in updated_cluster.instances
                        )
                        _prov_fb = s.provider
                        _cl_fb = updated_cluster

                        async def _cleanup_then_prepare() -> Any:
                            if stale_iids:
                                try:
                                    await _prov_fb.terminate(_cl_fb, stale_iids)
                                except Exception:
                                    log.error(
                                        "CRITICAL: FAILED to terminate stale "
                                        "instances {ids} from previous offer "
                                        "— they may still be running!",
                                        ids=stale_iids,
                                    )
                            with suppress(Exception):
                                await _prov_fb.teardown(_cl_fb)
                            return await s.provider.prepare(s.spec, offer)

                        ctx.pipe_to_self(
                            _cleanup_then_prepare(),
                            mapper=lambda c: ClusterReady(cluster=c),
                            on_failure=lambda err: ProvisionFailed(
                                reason=str(err),
                            ),
                        )
                        return requesting(replace(
                            s,
                            remaining_offers=tuple(rest),
                            attempt=1,
                            node_refs=MappingProxyType({}),
                        ))

                    log.error(
                        "Exhausted {attempts} provision attempts and all "
                        "offers, only got {got}/{need} nodes — cleaning up",
                        attempts=s.attempt,
                        got=len(new_spawned), need=s.spec.nodes.desired,
                    )

                    instance_ids = tuple(
                        inst.id for inst in updated_cluster.instances
                    )
                    reply_to = s.reply_to
                    reason = (
                        f"Only provisioned {len(new_spawned)}"
                        f"/{s.spec.nodes.desired} nodes after "
                        f"{s.attempt} attempts across all offers"
                    )
                    ctx.pipe_to_self(
                        _shutdown_cluster(s.provider, updated_cluster, instance_ids),
                        mapper=lambda _: ProvisionFailed(reason=reason),
                    )
                    return _cleanup_awaiting(reply_to)

                case StopPool(reply_to=stop_reply):
                    log.debug("StopPool during provisioning_instances")
                    for node_ref in s.node_refs.values():
                        ctx.stop(node_ref)
                    s.reply_to.tell(ProvisionFailed(reason="Interrupted"))
                    _iids = (
                        tuple(inst.id for inst in s.cluster.instances)
                        if s.cluster is not None else ()
                    )
                    ctx.pipe_to_self(
                        _shutdown_cluster(s.provider, s.cluster, _iids),
                        mapper=lambda _: _ShutdownDone(),
                    )
                    return stopping(stop_reply, s.cluster_id or "")

                case GetPoolSnapshot(reply_to=snap_reply):
                    snap_reply.tell(build_pool_snapshot(s, pool_name))
                    return Behaviors.same()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def _cleanup_awaiting(reply_to: ActorRef) -> Behavior[PoolMsg]:
        """Wait for cleanup to finish, then report failure and stop."""
        clog = logger.bind(actor="pool", state="cleanup")

        async def receive(_ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case ProvisionFailed() as pf:
                    clog.info("Cleanup complete, reporting failure")
                    reply_to.tell(pf)
                    return Behaviors.stopped()
                case StopPool(reply_to=stop_reply):
                    stop_reply.tell(PoolStopped())
                    return Behaviors.same()
            return Behaviors.same()
        return Behaviors.receive(receive)

    # ── provisioning ──────────────────────────────────────────────

    def provisioning(s: PoolState) -> Behavior[PoolMsg]:
        assert s.tm_ref is not None
        tm = s.tm_ref
        log = logger.bind(actor="pool", state="provisioning")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    if not s.spec.cluster:
                        return Behaviors.same()
                    log.info("Head address known: {addr}", addr=h.head_addr)
                    for nid, node_ref in s.node_refs.items():
                        if nid != 0:
                            node_ref.tell(h)
                    return provisioning(replace(s, head_addr=h.head_addr))
                case NodeConnected(node_id=nid, instance=meta):
                    iid = meta.instance.id
                    new_statuses = MappingProxyType({**s.node_statuses, iid: NodeStatus.SSH})
                    log.debug("Node {nid} SSH connected", nid=nid)
                    return provisioning(replace(s, node_statuses=new_statuses))
                case NodeBecameReady(node_id=nid, instance=meta):
                    new_instances = MappingProxyType({**s.instances, nid: meta})
                    log.info(
                        "Node {nid} ready ({n}/{total})",
                        nid=nid, n=len(new_instances), total=s.spec.nodes.desired,
                    )
                    iid = meta.instance.id
                    new_statuses = MappingProxyType({**s.node_statuses, iid: NodeStatus.BOOTSTRAPPING})
                    effective_head = s.head_addr or msg.private_ip or ""

                    if s.spec.cluster:
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
                        return provisioning(replace(
                            s,
                            instances=new_instances,
                            client=new_client,
                            head_addr=effective_head,
                            node_statuses=new_statuses,
                        ))
                    else:
                        node_client = await _create_standalone_client(
                            msg.private_ip, msg.casty_port, msg.local_port,
                            tls=s.client_tls,
                        )
                        log.info("Standalone client created for node {nid}", nid=nid)
                        _send_join_cluster(
                            node_client, msg, s.node_refs[nid],
                            s.spec, s.cluster, effective_head,
                        )
                        return provisioning(replace(
                            s,
                            instances=new_instances,
                            clients=MappingProxyType({**s.clients, nid: node_client}),
                            head_addr=effective_head,
                            node_statuses=new_statuses,
                        ))
                case NodeActivated(node_id=nid, node_ref=nref, slots=slots):
                    log.info("Node {nid} activated", nid=nid)
                    tm.tell(NodeAvailable(node_id=nid, node_ref=nref, slots=slots + s.spec.worker.buffer))
                    new_ready = s.ready_nodes | {nid}
                    ni = s.instances.get(nid)
                    new_statuses = s.node_statuses
                    if ni:
                        iid = ni.instance.id
                        new_statuses = MappingProxyType({**s.node_statuses, iid: NodeStatus.READY})
                    ready_threshold = s.spec.nodes.min or s.spec.nodes.desired

                    new_scaling = replace(
                        s.scaling,
                        pending_nodes=max(0, s.scaling.pending_nodes - 1),
                    )

                    if len(new_ready) >= ready_threshold:
                        if len(new_ready) == s.spec.nodes.desired:
                            log.info("All {n} nodes active, pool is operational", n=s.spec.nodes.desired)
                        else:
                            log.info(
                                "{n}/{total} nodes active (min threshold met), pool is operational",
                                n=len(new_ready), total=s.spec.nodes.desired,
                            )
                        ctx.self.tell(PoolStarted(
                            cluster_id=s.cluster_id,
                            instances=tuple(s.instances.values()),
                            cluster=s.cluster,
                        ))
                        return ready(replace(
                            s, ready_nodes=new_ready, node_statuses=new_statuses,
                            phase=PoolPhase.WORKERS, scaling=new_scaling,
                        ))
                    return provisioning(replace(
                        s, ready_nodes=new_ready, node_statuses=new_statuses,
                        scaling=new_scaling,
                    ))
                case NodeBecameUnready(node_id=nid, reason=reason):
                    log.warning(
                        "Node {nid} became unready during provisioning: {reason}",
                        nid=nid, reason=reason,
                    )
                    tm.tell(NodeUnavailable(node_id=nid))
                    return provisioning(replace(
                        s, ready_nodes=s.ready_nodes - {nid},
                    ))
                case NodeLost(node_id=nid, reason=reason):
                    log.warning(
                        "Node {nid} lost during provisioning: {reason}",
                        nid=nid, reason=reason,
                    )
                    tm.tell(NodeUnavailable(node_id=nid))
                    return provisioning(replace(
                        s, ready_nodes=s.ready_nodes - {nid},
                    ))
                case NodeExhausted(node_id=nid, reason=reason, instance_id=ev_iid):
                    tm.tell(NodeUnavailable(node_id=nid))
                    new_dead = s.dead_nodes | {nid}

                    dead_iid = ev_iid or _resolve_dead_iid(s, nid)
                    if dead_iid is not None and s.cluster is not None:
                        ctx.pipe_to_self(
                            s.provider.terminate(s.cluster, (dead_iid,)),
                            mapper=lambda _: _ShutdownDone(),
                            on_failure=lambda _err: _ShutdownDone(),
                        )

                    if s.reconciler_ref is not None:
                        s.reconciler_ref.tell(ReconcilerNodeLost(
                            node_id=nid, reason=reason,
                        ))

                    new_node_refs = MappingProxyType(
                        {k: v for k, v in s.node_refs.items() if k != nid},
                    )
                    return provisioning(replace(
                        s,
                        ready_nodes=s.ready_nodes - {nid},
                        dead_nodes=new_dead,
                        node_refs=new_node_refs,
                        cluster=_prune_cluster_instance(s.cluster, dead_iid),
                    ))

                case RequestScaleUp(count=count):
                    log.info("Reconciler requested scale-up of {n} nodes", n=count)
                    _prov = s.provider
                    _cl = s.cluster

                    async def _do_provision_prov() -> InstancesProvisioned:
                        result = await _prov.provision(_cl, count)
                        return InstancesProvisioned(instances=tuple(result[1]), cluster=result[0])

                    ctx.pipe_to_self(
                        _do_provision_prov(),
                        on_failure=lambda _err: InstancesProvisioned(instances=(), cluster=_cl),
                    )
                    return provisioning(s)

                case InstancesProvisioned(instances=new_instances, cluster=upd_cluster):
                    if not new_instances:
                        if s.reconciler_ref is not None:
                            s.reconciler_ref.tell(ScaleUpComplete(provisioned=0))
                        return provisioning(s)

                    start_id = max((*s.node_refs.keys(), *s.dead_nodes), default=-1) + 1
                    log.info(
                        "Scale-up: spawning {n} nodes from id {sid}",
                        n=len(new_instances), sid=start_id,
                    )
                    new_node_refs = s.node_refs
                    new_inst_map = s.instance_map
                    for i, inst in enumerate(new_instances):
                        nid = start_id + i
                        ref = _spawn_node(
                            ctx, nid, s.spec, upd_cluster,
                            s.provider, inst, s.ca, s.head_addr,
                        )
                        new_node_refs = MappingProxyType({**new_node_refs, nid: ref})
                        new_inst_map = MappingProxyType({**new_inst_map, nid: inst.id})

                    if s.reconciler_ref is not None:
                        s.reconciler_ref.tell(ScaleUpComplete(provisioned=len(new_instances)))

                    return provisioning(replace(
                        s,
                        cluster=upd_cluster,
                        node_refs=new_node_refs,
                        instance_map=new_inst_map,
                        scaling=replace(s.scaling, pending_nodes=s.scaling.pending_nodes + len(new_instances)),
                    ))

                case _ShutdownDone():
                    log.debug("Dead instance terminated")
                    return Behaviors.same()

                case StopPool(reply_to=stop_reply):
                    log.debug("StopPool during provisioning")
                    for node_ref in s.node_refs.values():
                        ctx.stop(node_ref)
                    if s.reconciler_ref is not None:
                        ctx.stop(s.reconciler_ref)
                    s.reply_to.tell(ProvisionFailed(reason="Interrupted"))
                    _iids = (
                        tuple(inst.id for inst in s.cluster.instances)
                        if s.cluster is not None else ()
                    )
                    ctx.pipe_to_self(
                        _shutdown_cluster(s.provider, s.cluster, _iids),
                        mapper=lambda _: _ShutdownDone(),
                    )
                    return stopping(
                        stop_reply, s.cluster_id or "",
                        replace(s, phase=PoolPhase.STOPPED), pool_name,
                    )

                case GetPoolSnapshot(reply_to=snap_reply):
                    snap_reply.tell(build_pool_snapshot(s, pool_name))
                    return Behaviors.same()
            return Behaviors.same()

        behavior: Behavior[PoolMsg] = Behaviors.receive(receive)
        if (client := s.client) is not None:
            async def _close(_ctx: ActorContext[PoolMsg]) -> None:
                with suppress(Exception):
                    await client.__aexit__(None, None, None)
            behavior = Behaviors.with_lifecycle(behavior, post_stop=_close)
        return behavior

    # ── ready ─────────────────────────────────────────────────────

    def ready(s: PoolState) -> Behavior[PoolMsg]:
        assert s.tm_ref is not None
        assert s.client is not None or s.clients
        tm = s.tm_ref
        client = s.client
        log = logger.bind(actor="pool", state="ready")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case PoolStarted() as started:
                    s.reply_to.tell(started)

                    is_elastic = s.spec.nodes.auto_scaling
                    if is_elastic:
                        from skyward.actors.autoscaler import autoscaler_actor

                        assert s.spec.nodes.max is not None
                        assert s.reconciler_ref is not None
                        as_ref = ctx.spawn(
                            autoscaler_actor(
                                min_nodes=s.spec.nodes.desired,
                                max_nodes=s.spec.nodes.max,
                                reconciler=s.reconciler_ref,
                                slots_per_node=s.spec.worker.concurrency + s.spec.worker.buffer,
                                initial_count=s.spec.nodes.desired,
                                cooldown=s.spec.autoscale_cooldown,
                                scale_down_idle_seconds=s.spec.autoscale_idle_timeout,
                            ),
                            "autoscaler",
                        )
                        tm.tell(RegisterPressureObserver(observer=as_ref))

                    min_n = s.spec.nodes.desired
                    inst_map = MappingProxyType({nid: ni.instance.id for nid, ni in s.instances.items()})
                    new_scaling = ScalingSnapshot(
                        desired_nodes=min_n,
                        is_elastic=is_elastic,
                        min_nodes=min_n if is_elastic else None,
                        max_nodes=s.spec.nodes.max if is_elastic else None,
                    )
                    return ready(replace(
                        s,
                        instance_map=inst_map,
                        phase=PoolPhase.READY,
                        scaling=new_scaling,
                    ))
                case SubmitTask() as task:
                    log.debug("Task submitted")
                    tm.tell(task)
                    new_counters = replace(s.task_counters, queued=s.task_counters.queued + 1)
                    return ready(replace(s, task_counters=new_counters))
                case SubmitBroadcast() as bcast:
                    log.debug("Broadcast submitted to {n} nodes", n=len(s.ready_nodes))
                    tm.tell(bcast)
                    new_counters = replace(s.task_counters, queued=s.task_counters.queued + 1)
                    return ready(replace(s, task_counters=new_counters))
                case NodeConnected(node_id=nid, instance=meta):
                    iid = meta.instance.id
                    new_statuses = MappingProxyType({**s.node_statuses, iid: NodeStatus.SSH})
                    return ready(replace(s, node_statuses=new_statuses))
                case NodeBecameReady(node_id=nid, instance=meta) as nbr:
                    iid = meta.instance.id
                    new_statuses = MappingProxyType({**s.node_statuses, iid: NodeStatus.BOOTSTRAPPING})
                    new_instances = MappingProxyType({**s.instances, nid: meta})
                    if s.spec.cluster:
                        assert client is not None
                        _update_tunnel(nbr)
                        effective_head = s.head_addr or nbr.private_ip or ""
                        _send_join_cluster(
                            client, nbr, s.node_refs[nid],
                            s.spec, s.cluster, effective_head,
                        )
                        if s.reconciler_ref is not None:
                            s.reconciler_ref.tell(NodeJoined(node_id=nid))
                        return ready(replace(
                            s,
                            node_statuses=new_statuses,
                            instances=new_instances,
                            instance_map=MappingProxyType({**s.instance_map, nid: meta.instance.id}),
                        ))
                    else:
                        node_client = await _create_standalone_client(
                            nbr.private_ip, nbr.casty_port, nbr.local_port,
                            tls=s.client_tls,
                        )
                        effective_head = nbr.private_ip or ""
                        _send_join_cluster(
                            node_client, nbr, s.node_refs[nid],
                            s.spec, s.cluster, effective_head,
                        )
                        old = s.clients.get(nid)
                        if old:
                            with suppress(Exception):
                                await old.__aexit__(None, None, None)
                        if s.reconciler_ref is not None:
                            s.reconciler_ref.tell(NodeJoined(node_id=nid))
                        return ready(replace(
                            s,
                            clients=MappingProxyType({**s.clients, nid: node_client}),
                            node_statuses=new_statuses,
                            instances=new_instances,
                            instance_map=MappingProxyType({**s.instance_map, nid: meta.instance.id}),
                        ))
                case NodeActivated(node_id=nid, node_ref=nref, slots=slots):
                    log.info("Node {nid} activated", nid=nid)
                    tm.tell(NodeAvailable(node_id=nid, node_ref=nref, slots=slots + s.spec.worker.buffer))
                    ni = s.instances.get(nid)
                    new_statuses = s.node_statuses
                    if ni:
                        iid = ni.instance.id
                        new_statuses = MappingProxyType({**s.node_statuses, iid: NodeStatus.READY})
                    new_scaling = replace(
                        s.scaling,
                        pending_nodes=max(0, s.scaling.pending_nodes - 1),
                    )
                    return ready(replace(
                        s, ready_nodes=s.ready_nodes | {nid},
                        node_statuses=new_statuses, scaling=new_scaling,
                    ))
                case NodeBecameUnready(node_id=nid, reason=reason):
                    log.warning(
                        "Node {nid} became unready: {reason}",
                        nid=nid, reason=reason,
                    )
                    tm.tell(NodeUnavailable(node_id=nid))
                    return ready(replace(s, ready_nodes=s.ready_nodes - {nid}))
                case NodeLost(node_id=nid, reason=reason):
                    log.warning(
                        "Node {nid} lost: {reason}, {remaining} nodes remaining",
                        nid=nid, reason=reason,
                        remaining=len(s.ready_nodes) - 1,
                    )
                    tm.tell(NodeUnavailable(node_id=nid))
                    if s.spec.cluster and s.head_addr:
                        head_msg = HeadAddressKnown(
                            head_addr=s.head_addr,
                            casty_port=25520,
                            num_nodes=s.spec.nodes.max or s.spec.nodes.desired,
                            worker_concurrency=s.spec.worker.concurrency,
                            worker_executor=s.spec.worker.resolved_executor,
                        )
                        s.node_refs[nid].tell(head_msg)
                    elif not s.spec.cluster:
                        old_client = s.clients.get(nid)
                        if old_client:
                            with suppress(Exception):
                                await old_client.__aexit__(None, None, None)
                        return ready(replace(
                            s,
                            ready_nodes=s.ready_nodes - {nid},
                            clients=MappingProxyType({k: v for k, v in s.clients.items() if k != nid}),
                        ))
                    return ready(replace(
                        s, ready_nodes=s.ready_nodes - {nid},
                    ))
                case NodeExhausted(node_id=nid, reason=reason, instance_id=ev_iid):
                    if nid not in s.node_refs:
                        return Behaviors.same()
                    log.error(
                        "Node {nid} permanently lost: {reason}",
                        nid=nid, reason=reason,
                    )
                    tm.tell(NodeUnavailable(node_id=nid))
                    new_dead = s.dead_nodes | {nid}

                    dead_iid = ev_iid or _resolve_dead_iid(s, nid)

                    if dead_iid is not None and s.cluster is not None:
                        ctx.pipe_to_self(
                            s.provider.terminate(s.cluster, (dead_iid,)),
                            mapper=lambda _: _ShutdownDone(),
                            on_failure=lambda _err: _ShutdownDone(),
                        )

                    if s.reconciler_ref is not None:
                        s.reconciler_ref.tell(ReconcilerNodeLost(
                            node_id=nid, reason=reason,
                        ))

                    new_node_refs = MappingProxyType(
                        {k: v for k, v in s.node_refs.items() if k != nid}
                    )
                    new_instances = MappingProxyType(
                        {k: v for k, v in s.instances.items() if k != nid}
                    )
                    new_inst_map = MappingProxyType(
                        {k: v for k, v in s.instance_map.items() if k != nid}
                    )
                    return ready(replace(
                        s,
                        ready_nodes=s.ready_nodes - {nid},
                        dead_nodes=new_dead,
                        node_refs=new_node_refs,
                        instances=new_instances,
                        instance_map=new_inst_map,
                        cluster=_prune_cluster_instance(s.cluster, dead_iid),
                    ))
                case ReconciliationExhausted(reason=reason):
                    if not s.ready_nodes:
                        log.error(
                            "Reconciler gave up with 0 ready nodes, shutting down: {reason}",
                            reason=reason,
                        )
                        for node_ref in s.node_refs.values():
                            ctx.stop(node_ref)
                        if s.reconciler_ref is not None:
                            ctx.stop(s.reconciler_ref)
                        instance_ids = _collect_instance_ids(s)
                        ctx.pipe_to_self(
                            _shutdown_cluster(s.provider, s.cluster, instance_ids),
                            mapper=lambda _: _ShutdownDone(),
                        )
                        return stopping(
                            None, s.cluster_id,
                            replace(s, phase=PoolPhase.STOPPED), pool_name,
                        )
                    log.warning(
                        "Reconciler gave up but {n} nodes still active",
                        n=len(s.ready_nodes),
                    )
                    return Behaviors.same()

                case RequestScaleUp(count=count):
                    log.info("Reconciler requested scale-up of {n} nodes", n=count)
                    _prov = s.provider
                    _cl = s.cluster

                    async def _do_provision_ready() -> InstancesProvisioned:
                        result = await _prov.provision(_cl, count)
                        return InstancesProvisioned(instances=tuple(result[1]), cluster=result[0])

                    ctx.pipe_to_self(
                        _do_provision_ready(),
                        on_failure=lambda _err: InstancesProvisioned(instances=(), cluster=_cl),
                    )
                    return ready(s)

                case InstancesProvisioned(instances=new_instances, cluster=upd_cluster):
                    if not new_instances:
                        if s.reconciler_ref is not None:
                            s.reconciler_ref.tell(ScaleUpComplete(provisioned=0))
                        return ready(s)

                    start_id = max((*s.node_refs.keys(), *s.dead_nodes), default=-1) + 1
                    log.info(
                        "Scale-up: spawning {n} nodes from id {sid}",
                        n=len(new_instances), sid=start_id,
                    )
                    new_node_refs = s.node_refs
                    new_inst_map = s.instance_map
                    for i, inst in enumerate(new_instances):
                        nid = start_id + i
                        ref = _spawn_node(
                            ctx, nid, s.spec, upd_cluster,
                            s.provider, inst, s.ca, s.head_addr,
                        )
                        new_node_refs = MappingProxyType({**new_node_refs, nid: ref})
                        new_inst_map = MappingProxyType({**new_inst_map, nid: inst.id})

                    if s.reconciler_ref is not None:
                        s.reconciler_ref.tell(ScaleUpComplete(provisioned=len(new_instances)))

                    return ready(replace(
                        s,
                        cluster=upd_cluster,
                        node_refs=new_node_refs,
                        instance_map=new_inst_map,
                        scaling=replace(s.scaling, pending_nodes=s.scaling.pending_nodes + len(new_instances)),
                    ))

                case RequestScaleDown(count=count):
                    log.info("Reconciler requested scale-down of {n} nodes", n=count)
                    victims = sorted(s.node_refs.keys(), reverse=True)[:count]
                    victims = [nid for nid in victims if nid != 0][:count]
                    if not victims:
                        if s.reconciler_ref is not None:
                            s.reconciler_ref.tell(ScaleDownComplete(drained=0))
                        return ready(s)

                    new_node_refs = s.node_refs
                    for nid in victims:
                        tm.tell(NodeUnavailable(node_id=nid))
                        iid = s.instance_map.get(nid, "")
                        if s.reconciler_ref is not None:
                            s.reconciler_ref.tell(DrainComplete(node_id=nid, instance_id=iid))
                        node_ref = s.node_refs.get(nid)
                        if node_ref:
                            ctx.stop(node_ref)
                        new_node_refs = MappingProxyType({k: v for k, v in new_node_refs.items() if k != nid})
                        if iid and s.cluster is not None:
                            ctx.pipe_to_self(
                                s.provider.terminate(s.cluster, (iid,)),
                                mapper=lambda _: _ShutdownDone(),
                                on_failure=lambda _err: _ShutdownDone(),
                            )

                    if s.reconciler_ref is not None:
                        s.reconciler_ref.tell(ScaleDownComplete(drained=len(victims)))

                    return ready(replace(
                        s,
                        node_refs=new_node_refs,
                        ready_nodes=s.ready_nodes - frozenset(victims),
                        scaling=replace(s.scaling, draining_nodes=s.scaling.draining_nodes + len(victims)),
                    ))

                case _ShutdownDone():
                    return Behaviors.same()

                case GetCurrentNodes(reply_to=query_reply):
                    query_reply.tell(CurrentNodeCount(
                        count=len(s.node_refs), ready=len(s.ready_nodes),
                    ))
                    return Behaviors.same()

                case GetPoolSnapshot(reply_to=snap_reply):
                    snap_reply.tell(build_pool_snapshot(s, pool_name))
                    return Behaviors.same()

                case StopPool(reply_to=stop_reply):
                    log.debug(
                        "StopPool, shutting down cluster {cid}",
                        cid=s.cluster_id,
                    )
                    for node_ref in s.node_refs.values():
                        ctx.stop(node_ref)
                    if s.reconciler_ref is not None:
                        ctx.stop(s.reconciler_ref)
                    instance_ids = _collect_instance_ids(s)
                    ctx.pipe_to_self(
                        _shutdown_cluster(s.provider, s.cluster, instance_ids),
                        mapper=lambda _: _ShutdownDone(),
                    )
                    return stopping(
                        stop_reply, s.cluster_id,
                        replace(s, phase=PoolPhase.STOPPED), pool_name,
                    )
            return Behaviors.same()

        async def _close_client(_ctx: ActorContext[PoolMsg]) -> None:
            if client is not None:
                with suppress(Exception):
                    await client.__aexit__(None, None, None)
            for c in s.clients.values():
                with suppress(Exception):
                    await c.__aexit__(None, None, None)

        return Behaviors.with_lifecycle(
            Behaviors.receive(receive),
            post_stop=_close_client,
        )

    # ── stopping ──────────────────────────────────────────────────

    def stopping(
        stop_reply: ActorRef | None, cluster_id: str,
        s: PoolState | None = None, name: str = "",
    ) -> Behavior[PoolMsg]:
        log = logger.bind(actor="pool", state="stopping")

        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            log.debug("received: {message}", message=type(msg).__name__)
            match msg:
                case _ShutdownDone():
                    log.info("Cluster {cid} shutdown confirmed", cid=cluster_id)
                    if stop_reply is not None:
                        stop_reply.tell(PoolStopped())
                    return Behaviors.stopped()
                case InstancesProvisioned(
                    instances=late_instances,
                ) if s is not None and late_instances:
                    late_iids = tuple(inst.id for inst in late_instances)
                    log.warning(
                        "InstancesProvisioned arrived during shutdown — "
                        "terminating orphaned instances {iids}",
                        iids=late_iids,
                    )
                    ctx.pipe_to_self(
                        s.provider.terminate(s.cluster, late_iids),
                        mapper=lambda _: _ShutdownDone(),
                        on_failure=lambda _err: _ShutdownDone(),
                    )
                    return Behaviors.same()
                case InstancesProvisioned():
                    return Behaviors.same()
                case GetPoolSnapshot(reply_to=snap_reply) if s is not None:
                    snap_reply.tell(build_pool_snapshot(s, name))
                    return Behaviors.same()
            return Behaviors.same()
        return Behaviors.receive(receive)

    return idle()
