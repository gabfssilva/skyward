"""Merged node actor — one actor per compute node.

Tells this story:
  idle → polling → connecting → bootstrapping → post_bootstrap
       → starting_worker → ready → active
       → replacing → polling → … (loop back, if connection permanently fails)

The transport actor (child) fully encapsulates SSH connection, port
forwarding, event streaming, and auto-reconnection. The node actor
communicates with it via messages, never touching asyncssh directly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors, Terminated

from skyward.actors.messages import (
    BootstrapDone,
    ExecuteOnNode,
    HeadAddressKnown,
    NodeActivated,
    NodeBecameReady,
    NodeLost,
    Preempted,
    Provision,
    TaskResult,
    _bind_to_node,
)
from skyward.actors.streaming import instance_monitor
from skyward.infra.ssh_actor import (
    ConnectionFailed,
    ConnectionLost,
    ConnectionRestored,
    ForwardPort,
    PortReForwarded,
    StopTransport,
    ssh_transport,
    transport_ask,
)
from skyward.infra.tls import CertificateAuthority
from skyward.observability.logger import logger
from skyward.providers.provider import WarmableProvider

from .helpers import (
    discover_own_worker,
    do_start_worker,
    execute_with_streaming,
    install_local_skyward,
    run_bootstrap,
    setup_worker_env,
    sync_user_code,
    terminate_and_replace,
)
from .messages import (
    JoinCluster,
    NodeMsg,
    _BootstrapUploaded,
    _BootstrapUploadFailed,
    _Connected,
    _ConnectionFailed,
    _EnvSetupDone,
    _EnvSetupFailed,
    _LocalInstallDone,
    _PollResult,
    _PostBootstrapFailed,
    _RemoteTaskDone,
    _SnapshotFailed,
    _SnapshotSaved,
    _UserCodeSyncDone,
    _WorkerDiscovered,
    _WorkerDiscoveryFailed,
    _WorkerFailed,
    _WorkerStarted,
)
from .state import NodeId, NodeState, PendingTask


def node_actor(
    node_id: NodeId,
    pool: ActorRef,
    ssh_timeout: float = 300.0,
    ssh_retry_interval: float = 5.0,
    poll_interval: float = 5.0,
    poll_timeout: float = 300.0,
    _skip_monitor: bool = False,
    ca: CertificateAuthority | None = None,
) -> Behavior[NodeMsg]:
    """A merged node tells this story: idle → polling → … → active."""

    log = logger.bind(actor="node", node_id=node_id)

    # ── common handlers ──────────────────────────────────────────────

    def _stop_transport(ref: ActorRef | None) -> None:
        if ref:
            ref.tell(StopTransport())

    def _fail_and_replace(ctx: ActorContext[NodeMsg], s: NodeState, reason: str) -> Behavior[NodeMsg]:
        _stop_transport(s.transport_ref)
        pool.tell(NodeLost(node_id=node_id, reason=reason))
        return _start_replacing(ctx, s)

    def _enqueue(s: NodeState, ex: ExecuteOnNode) -> NodeState:
        pt = PendingTask(ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout)
        return replace(s, pending_tasks=(*s.pending_tasks, pt))

    def _common(
        ctx: ActorContext[NodeMsg],
        msg: NodeMsg,
        s: NodeState,
        reenter: Callable[[NodeState], Behavior[NodeMsg]],
    ) -> Behavior[NodeMsg]:
        match msg:
            case HeadAddressKnown() as h:
                return reenter(replace(s, head_info=h))
            case ExecuteOnNode() as ex:
                return reenter(_enqueue(s, ex))
            case Preempted(reason=reason):
                return _fail_and_replace(ctx, s, reason)
            case Terminated():
                return _fail_and_replace(ctx, s, "child stopped")
            case _SnapshotSaved():
                log.info("Snapshot saved")
            case _SnapshotFailed(error=error):
                log.warning("Snapshot failed: {error}", error=error)
        return Behaviors.same()

    # ── idle ──────────────────────────────────────────────────────────

    def idle() -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case Provision(cluster=cluster, provider=provider, instance=instance):
                    log.info("Node {nid} provisioning instance {iid}", nid=node_id, iid=instance.id)
                    s = NodeState(cluster=cluster, provider=provider)
                    return _start_polling(ctx, s, instance)
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── polling ───────────────────────────────────────────────────────

    def _start_polling(
        ctx: ActorContext[NodeMsg],
        s: NodeState,
        instance: Any,
    ) -> Behavior[NodeMsg]:
        instance_id = instance.id
        start_time = asyncio.get_event_loop().time()

        async def _do_poll() -> _PollResult:
            updated_cluster, inst = await s.provider.get_instance(s.cluster, instance_id)
            return _PollResult(instance=inst, cluster=updated_cluster)

        ctx.pipe_to_self(
            _do_poll(),
            on_failure=lambda e: _PollResult(instance=None),
        )
        return polling(s, instance_id, start_time)

    def polling(
        s: NodeState,
        instance_id: str,
        start_time: float,
    ) -> Behavior[NodeMsg]:
        provider_name = s.cluster.spec.provider or "aws"

        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case _PollResult(instance=inst, cluster=updated) if (
                    inst and inst.status == "provisioned" and inst.ip
                ):
                    c = updated or s.cluster
                    ni = _bind_to_node(inst, node_id, provider_name, c)
                    log.info("Instance ready at {ip}", ip=inst.ip)
                    return _start_connecting(ctx, replace(s, cluster=c, ni=ni))
                case _PollResult(cluster=updated):
                    c = updated or s.cluster
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > poll_timeout:
                        log.error("Instance not ready within {t}s", t=poll_timeout)
                        pool.tell(
                            NodeLost(node_id=node_id, reason=f"Instance not ready within {poll_timeout}s")
                        )
                        return _start_replacing(ctx, replace(s, cluster=c))

                    ns = replace(s, cluster=c)

                    async def _poll_after_delay() -> _PollResult:
                        await asyncio.sleep(poll_interval)
                        updated_cluster, inst = await ns.provider.get_instance(ns.cluster, instance_id)
                        return _PollResult(instance=inst, cluster=updated_cluster)

                    ctx.pipe_to_self(
                        _poll_after_delay(),
                        on_failure=lambda e: _PollResult(instance=None),
                    )
                    return polling(ns, instance_id, start_time)
                case Preempted():
                    log.warning("Preempted during polling")
                    pool.tell(NodeLost(node_id=node_id, reason="preempted"))
                    return _start_replacing(ctx, s)
                case HeadAddressKnown() as h:
                    return polling(replace(s, head_info=h), instance_id, start_time)
                case ExecuteOnNode() as ex:
                    return polling(_enqueue(s, ex), instance_id, start_time)
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── connecting ────────────────────────────────────────────────────

    def _start_connecting(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        ni = s.ni
        assert ni is not None
        log.info("Spawning transport actor for {ip}", ip=ni.instance.ip)

        transport_ref = ctx.spawn(
            ssh_transport(
                host=ni.instance.ip or "",
                user=ni.ssh_user or s.cluster.ssh_user,
                key_path=ni.ssh_key_path or s.cluster.ssh_key_path,
                port=ni.instance.ssh_port,
                retry_max_attempts=int(ssh_timeout / ssh_retry_interval) + 1,
                retry_delay=ssh_retry_interval,
                connect_timeout=min(ssh_retry_interval * 2, 10.0),
                reconnect_max_attempts=5,
                parent=ctx.self,
            ),
            f"transport-{ni.instance.id}",
        )
        ctx.watch(transport_ref)

        async def _await_forward() -> _Connected:
            result = await transport_ask(
                transport_ref,
                lambda rt: ForwardPort(remote_host="127.0.0.1", remote_port=25520, reply_to=rt),
                timeout=ssh_timeout,
            )
            return _Connected(transport_ref=transport_ref, local_port=result.local_port, instance=ni)

        ctx.pipe_to_self(
            _await_forward(),
            on_failure=lambda e: _ConnectionFailed(error=str(e)),
        )
        return connecting(replace(s, transport_ref=transport_ref))

    def connecting(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case _Connected(transport_ref=tref, local_port=lp):
                    log.info("SSH tunnel established (port={port})", port=lp)
                    return _start_bootstrapping(ctx, replace(s, transport_ref=tref, local_port=lp))
                case _ConnectionFailed(error=error):
                    log.error("SSH connection failed: {error}", error=error)
                    return _fail_and_replace(ctx, s, error)
                case ConnectionFailed(error=error):
                    log.error("Transport permanently failed: {error}", error=error)
                    return _fail_and_replace(ctx, s, error)
            return _common(ctx, msg, s, connecting)

        return Behaviors.receive(receive)

    # ── bootstrapping ─────────────────────────────────────────────────

    def _start_bootstrapping(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        ni = s.ni
        tref = s.transport_ref
        assert ni is not None
        assert tref is not None
        log.info("Starting bootstrap")

        if not _skip_monitor:
            monitor_ref = ctx.spawn(
                instance_monitor(
                    info=ni,
                    transport=tref,
                    event_listener=pool,
                    reply_to=ctx.self,
                ),
                f"monitor-{ni.instance.id}",
            )
            ctx.watch(monitor_ref)

        if s.cluster.prebaked:
            log.info("Prebaked image detected, skipping bootstrap")
            return _enter_post_bootstrap(ctx, s)

        ctx.pipe_to_self(
            run_bootstrap(tref, ni, s.cluster, s.cluster.spec),
            mapper=lambda _: _BootstrapUploaded(),
            on_failure=lambda e: _BootstrapUploadFailed(error=str(e)),
        )
        return bootstrapping(s)

    def bootstrapping(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case BootstrapDone(success=True, instance=done_info):
                    log.info("Bootstrap completed successfully")
                    final_ni = done_info or s.ni
                    match s.provider:
                        case WarmableProvider() if node_id == 0:
                            log.info("Snapshotting provider image...")
                            ctx.pipe_to_self(
                                s.provider.save(s.cluster),
                                mapper=lambda _: _SnapshotSaved(),
                                on_failure=lambda e: _SnapshotFailed(error=str(e)),
                            )
                    return _enter_post_bootstrap(ctx, replace(s, ni=final_ni))
                case BootstrapDone(success=False, error=error):
                    log.error("Bootstrap failed: {error}", error=error)
                    return _fail_and_replace(ctx, s, error or "bootstrap failed")
                case _BootstrapUploaded():
                    log.info("Bootstrap script uploaded and started")
                    return Behaviors.same()
                case _BootstrapUploadFailed(error=error):
                    log.error("Bootstrap upload failed: {error}", error=error)
                    return _fail_and_replace(ctx, s, error)
            return _common(ctx, msg, s, bootstrapping)

        return Behaviors.receive(receive)

    # ── post_bootstrap ────────────────────────────────────────────────

    def _enter_post_bootstrap(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        ni = s.ni
        tref = s.transport_ref
        assert ni is not None
        assert tref is not None
        spec = s.cluster.spec

        if spec.image and getattr(spec.image, "skyward_source", None) == "local":
            ctx.pipe_to_self(
                install_local_skyward(tref, ni, s.cluster),
                mapper=lambda _, m=ni: _LocalInstallDone(instance=m),
                on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
            )
            return post_bootstrap(s)

        if spec.image and getattr(spec.image, "includes", None):
            ctx.pipe_to_self(
                sync_user_code(tref, ni, spec, s.cluster),
                mapper=lambda _, m=ni: _UserCodeSyncDone(instance=m),
                on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
            )
            return post_bootstrap(s)

        return _enter_ready(ctx, s)

    def post_bootstrap(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            tref = s.transport_ref
            assert tref is not None
            match msg:
                case _LocalInstallDone(instance=info):
                    spec = s.cluster.spec
                    if spec.image and getattr(spec.image, "includes", None):
                        ctx.pipe_to_self(
                            sync_user_code(tref, info, spec, s.cluster),
                            mapper=lambda _, m=info: _UserCodeSyncDone(instance=m),
                            on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
                        )
                        return Behaviors.same()
                    return _enter_ready(ctx, s)
                case _UserCodeSyncDone():
                    return _enter_ready(ctx, s)
                case _PostBootstrapFailed(error=err):
                    return _fail_and_replace(ctx, s, err)
            return _common(ctx, msg, s, post_bootstrap)

        return Behaviors.receive(receive)

    # ── ready (worker started, waiting for JoinCluster) ───────────────

    def _enter_ready(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        ni = s.ni
        assert ni is not None
        spec = s.cluster.spec
        is_head = node_id == 0

        if is_head:
            head_private = ni.instance.private_ip or ni.instance.ip or ""
            pool.tell(
                HeadAddressKnown(
                    head_addr=head_private,
                    casty_port=25520,
                    num_nodes=spec.nodes.min,
                    worker_concurrency=spec.worker.concurrency,
                    worker_executor=spec.worker.resolved_executor,
                )
            )
            log.info("Starting worker (role=head)")
            return _start_worker_process(ctx, s)

        if s.head_info is not None:
            log.info("Starting worker (role=worker)")
            return _start_worker_process(ctx, s)

        log.info("Waiting for head address before starting worker")
        return waiting_for_head(s)

    def waiting_for_head(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    log.info("Head address received, starting worker")
                    return _start_worker_process(ctx, replace(s, head_info=h))
            return _common(ctx, msg, s, waiting_for_head)

        return Behaviors.receive(receive)

    # ── starting worker ───────────────────────────────────────────────

    def _start_worker_process(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        ni = s.ni
        tref = s.transport_ref
        assert ni is not None
        assert tref is not None
        ctx.pipe_to_self(
            do_start_worker(tref, s.local_port, ni, s.head_info, node_id, s.cluster, s.cluster.spec, ca=ca),
            mapper=lambda result: _WorkerStarted(local_port=result[0], private_ip=result[1]),
            on_failure=lambda e: _WorkerFailed(error=str(e)),
        )
        return starting_worker(s)

    def starting_worker(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            ni = s.ni
            assert ni is not None
            match msg:
                case _WorkerStarted(local_port=lp, private_ip=pip):
                    log.info("Worker started, tunnel port={port}", port=lp)
                    pool.tell(
                        NodeBecameReady(
                            node_id=node_id,
                            instance=ni,
                            local_port=lp,
                            private_ip=pip,
                        )
                    )
                    return ready(s)
                case _WorkerFailed(error=error):
                    log.error("Worker failed to start: {error}", error=error)
                    return _fail_and_replace(ctx, s, error)
            return _common(ctx, msg, s, starting_worker)

        return Behaviors.receive(receive)

    # ── ready (worker running, waiting for JoinCluster from pool) ─────

    def ready(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            ni = s.ni
            assert ni is not None
            match msg:
                case JoinCluster(
                    client=client, pool_info_json=pij,
                    env_vars=ev, around_app_hooks=hooks,
                    around_process_hooks=phooks,
                ):
                    log.info("JoinCluster received, discovering worker")
                    ctx.pipe_to_self(
                        discover_own_worker(client, ni),
                        mapper=lambda ref: _WorkerDiscovered(worker_ref=ref),
                        on_failure=lambda e: _WorkerDiscoveryFailed(error=str(e)),
                    )
                    return joining(replace(
                        s, client=client, pool_info_json=pij,
                        env_vars=ev, around_app_hooks=hooks,
                        around_process_hooks=phooks,
                    ))
                case PortReForwarded(old_port=_, new_port=new_port):
                    log.info("Port re-forwarded to {port} while ready", port=new_port)
                    pool.tell(
                        NodeBecameReady(
                            node_id=node_id,
                            instance=ni,
                            local_port=new_port,
                            private_ip=ni.instance.private_ip or ni.instance.ip or "",
                        )
                    )
                    return ready(replace(s, local_port=new_port))
                case ConnectionLost():
                    log.warning("Connection lost while ready, transport reconnecting")
                    return Behaviors.same()
                case ConnectionRestored(local_port=new_port):
                    effective_port = new_port or s.local_port
                    log.info("Connection restored while ready (port={port})", port=effective_port)
                    pool.tell(
                        NodeBecameReady(
                            node_id=node_id,
                            instance=ni,
                            local_port=effective_port,
                            private_ip=ni.instance.private_ip or ni.instance.ip or "",
                        )
                    )
                    return ready(replace(s, local_port=effective_port))
                case ConnectionFailed(error=error):
                    log.error("Transport permanently failed while ready: {err}", err=error)
                    pool.tell(NodeLost(node_id=node_id, reason="connection lost"))
                    return _start_replacing(ctx, s)
            return _common(ctx, msg, s, ready)

        return Behaviors.receive(receive)

    # ── joining ───────────────────────────────────────────────────────

    def joining(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case _WorkerDiscovered(worker_ref=wref):
                    log.info("Worker discovered, setting up env")
                    ctx.pipe_to_self(
                        setup_worker_env(
                            s.client, wref, s.pool_info_json,
                            s.env_vars, s.around_app_hooks, s.around_process_hooks,
                        ),
                        mapper=lambda _: _EnvSetupDone(),
                        on_failure=lambda e: _EnvSetupFailed(error=str(e)),
                    )
                    return joining_env_setup(replace(s, worker_ref=wref))
                case _WorkerDiscoveryFailed(error=error):
                    log.error("Worker discovery failed: {error}", error=error)
                    return _fail_and_replace(ctx, s, error)
            return _common(ctx, msg, s, joining)

        return Behaviors.receive(receive)

    # ── joining_env_setup ─────────────────────────────────────────────

    def joining_env_setup(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case _EnvSetupDone():
                    log.info("Env setup done, entering active")
                    return _enter_active(ctx, s)
                case _EnvSetupFailed(error=error):
                    log.error("Env setup failed: {error}", error=error)
                    return _fail_and_replace(ctx, s, error)
            return _common(ctx, msg, s, joining_env_setup)

        return Behaviors.receive(receive)

    # ── active ────────────────────────────────────────────────────────

    def _enter_active(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        new_inflight: dict[str, ActorRef] = {}
        counter = 0
        for pt in s.pending_tasks:
            tid = pt.task_id or str(counter)
            _dispatch_task(ctx, s, tid, pt.fn, pt.args, pt.kwargs, pt.timeout)
            new_inflight[tid] = pt.reply_to
            counter += 1

        pool.tell(NodeActivated(
            node_id=node_id,
            node_ref=ctx.self,
            slots=s.cluster.spec.worker.concurrency,
        ))
        return active(replace(s, inflight=new_inflight, task_counter=counter, pending_tasks=()))

    def _dispatch_task(
        ctx: ActorContext[NodeMsg],
        s: NodeState,
        tid: str,
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout: float,
    ) -> None:
        ctx.pipe_to_self(
            execute_with_streaming(s.client, s.worker_ref, fn, args, kwargs, timeout),
            mapper=lambda result, _tid=tid: _RemoteTaskDone(  # type: ignore[return-value]
                task_id=_tid,
                value=(result.result if hasattr(result, "result") else RuntimeError(result.error)),
                node_id=node_id,
                reply_to=ctx.self,
                error=not hasattr(result, "result"),
            ),
            on_failure=lambda e, _tid=tid: _RemoteTaskDone(  # type: ignore[return-value]
                task_id=_tid,
                value=RuntimeError(str(e)),
                node_id=node_id,
                reply_to=ctx.self,
                error=True,
                connection_error=True,
            ),
        )

    def active(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return active(replace(s, head_info=h))
                case JoinCluster(client=client):
                    log.info("Re-joining cluster on node {nid}", nid=node_id)
                    if s.client and s.client is not client:
                        with suppress(Exception):
                            await s.client.__aexit__(None, None, None)
                    ni = s.ni
                    if ni is None:
                        log.error("Cannot re-join: no node instance")
                        return Behaviors.same()
                    ctx.pipe_to_self(
                        discover_own_worker(client, ni),
                        mapper=lambda ref: _WorkerDiscovered(worker_ref=ref),
                        on_failure=lambda e: _WorkerDiscoveryFailed(error=str(e)),
                    )
                    return active(replace(s, client=client))
                case _WorkerDiscovered(worker_ref=wref):
                    log.info("Worker re-discovered on node {nid}", nid=node_id)
                    ctx.pipe_to_self(
                        setup_worker_env(
                            s.client, wref, s.pool_info_json,
                            s.env_vars, s.around_app_hooks, s.around_process_hooks,
                        ),
                        mapper=lambda _: _EnvSetupDone(),
                        on_failure=lambda e: _EnvSetupFailed(error=str(e)),
                    )
                    return active(replace(s, worker_ref=wref))
                case _WorkerDiscoveryFailed(error=error):
                    log.warning(
                        "Worker re-discovery failed on node {nid}: {error}",
                        nid=node_id, error=error,
                    )
                    return Behaviors.same()
                case _EnvSetupDone():
                    log.info("Worker env re-setup complete on node {nid}", nid=node_id)
                    return Behaviors.same()
                case _EnvSetupFailed(error=error):
                    log.warning(
                        "Worker env re-setup failed on node {nid}: {error}",
                        nid=node_id, error=error,
                    )
                    return Behaviors.same()
                case ExecuteOnNode() as ex:
                    local_tid = ex.task_id or str(s.task_counter)
                    log.debug("Node {nid} dispatching task {tid}", nid=node_id, tid=local_tid)
                    _dispatch_task(ctx, s, local_tid, ex.fn, ex.args, ex.kwargs, ex.timeout)
                    new_inflight = {**s.inflight, local_tid: ex.reply_to}
                    return active(replace(s, inflight=new_inflight, task_counter=s.task_counter + 1))
                case _RemoteTaskDone(task_id=tid, value=value, error=is_err, connection_error=conn_err):
                    log.debug("Node {nid} received task result (tid={tid})", nid=node_id, tid=tid)
                    caller = s.inflight.get(tid)
                    if caller:
                        caller.tell(
                            TaskResult(value=value, node_id=node_id, task_id=tid, error=is_err)
                        )
                    if conn_err:
                        log.warning(
                            "Connection error on node {nid}, waiting for transport reconnect",
                            nid=node_id,
                        )
                        conn_error = RuntimeError(f"Node {node_id} connection lost")
                        for other_tid, other_caller in s.inflight.items():
                            if other_tid != tid:
                                other_caller.tell(
                                    TaskResult(
                                        value=conn_error, node_id=node_id, task_id=other_tid, error=True,
                                    )
                                )
                        return active(replace(s, inflight={}, pending_tasks=()))
                    if caller:
                        new_inflight = {k: v for k, v in s.inflight.items() if k != tid}
                        return active(replace(s, inflight=new_inflight))
                    return Behaviors.same()
                case ConnectionLost(error=error):
                    log.warning(
                        "Connection lost on node {nid}: {err}, transport reconnecting",
                        nid=node_id, err=error,
                    )
                    return Behaviors.same()
                case ConnectionRestored(local_port=new_port):
                    effective_port = new_port or s.local_port
                    log.info(
                        "Transport reconnected on node {nid} (port={port})",
                        nid=node_id, port=effective_port,
                    )
                    ni = s.ni
                    if ni:
                        pool.tell(
                            NodeBecameReady(
                                node_id=node_id,
                                instance=ni,
                                local_port=effective_port,
                                private_ip=ni.instance.private_ip or ni.instance.ip or "",
                            )
                        )
                    return active(replace(s, local_port=effective_port))
                case ConnectionFailed(error=error):
                    log.error(
                        "Transport permanently failed on node {nid}: {err}",
                        nid=node_id, err=error,
                    )
                    pool.tell(NodeLost(node_id=node_id, reason="connection lost"))
                    return _start_replacing(ctx, s)
                case PortReForwarded(old_port=_, new_port=new_port):
                    log.info("Port re-forwarded to {port}", port=new_port)
                    ni = s.ni
                    if ni:
                        pool.tell(
                            NodeBecameReady(
                                node_id=node_id,
                                instance=ni,
                                local_port=new_port,
                                private_ip=ni.instance.private_ip or ni.instance.ip or "",
                            )
                        )
                    return active(replace(s, local_port=new_port))
                case Preempted(reason=reason):
                    log.warning("Preempted while active: {reason}", reason=reason)
                    _stop_transport(s.transport_ref)
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    return _start_replacing(ctx, s)
                case Terminated():
                    log.warning("Child died while active, marking node lost")
                    for tid, caller in s.inflight.items():
                        caller.tell(
                            TaskResult(
                                value=RuntimeError(f"Node {node_id} child stopped"),
                                node_id=node_id, task_id=tid, error=True,
                            )
                        )
                    pool.tell(NodeLost(node_id=node_id, reason="child stopped"))
                    return _start_replacing(ctx, s)
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── replacing ─────────────────────────────────────────────────────

    def _start_replacing(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        instance_id = s.ni.instance.id if s.ni else ""
        ctx.pipe_to_self(
            terminate_and_replace(s.provider, s.cluster, instance_id),
            mapper=lambda inst: Provision(cluster=s.cluster, provider=s.provider, instance=inst),
        )
        return replacing(s)

    def replacing(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return replacing(replace(s, head_info=h))
                case Provision(instance=instance):
                    log.info("Replacement provisioned, re-entering polling")
                    return _start_polling(ctx, s, instance)
                case ExecuteOnNode() as ex:
                    return replacing(_enqueue(s, ex))
            return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()
