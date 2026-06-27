"""Merged node actor — one actor per compute node.

Tells this story:
  idle → polling → connecting → bootstrapping → post_bootstrap
       → starting_worker → ready → active

On failure the node sends NodeExhausted and stops; the reconciler
handles termination and replacement provisioning.

The transport actor (child) fully encapsulates SSH connection, port
forwarding, event streaming, and auto-reconnection. The node actor
communicates with it via messages, never touching asyncssh directly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from types import MappingProxyType
from typing import Any, Literal

from casty import ActorContext, ActorRef, Behavior, Behaviors, Terminated

from skyward.actors.messages import (
    BootstrapDone,
    ExecuteOnNode,
    HeadAddressKnown,
    NodeActivated,
    NodeBecameBusy,
    NodeBecameIdle,
    NodeBecameReady,
    NodeConnected,
    NodeExhausted,
    NodeFileOp,
    NodeFileResult,
    Preempted,
    Provision,
    TaskFailed,
    TaskInterrupted,
    TaskSucceeded,
    _bind_to_node,
)
from skyward.actors.streaming import instance_monitor
from skyward.api.health import HealthChecker, hc_loop
from skyward.api.spec import (
    DEFAULT_BOOTSTRAP_TIMEOUT,
    DEFAULT_PROVISION_TIMEOUT,
    DEFAULT_SSH_TIMEOUT,
)
from skyward.infra.ssh_actor import (
    ConnectionFailed,
    ConnectionLost,
    ConnectionRestored,
    ForwardPort,
    StopTransport,
    ssh_transport,
    transport_ask,
)
from skyward.infra.tls import CertificateAuthority
from skyward.observability.logger import logger
from skyward.providers.provider import WarmableProvider

from .helpers import (
    _resolve_output_stream,
    _run_file_op,
    discover_own_worker,
    do_start_worker,
    execute_with_streaming,
    install_local_skyward,
    run_bootstrap,
    setup_worker_env,
    sync_user_code,
)
from .messages import (
    Adopt,
    JoinCluster,
    NodeMsg,
    _BootstrapUploaded,
    _BootstrapUploadFailed,
    _Connected,
    _ConnectionFailed,
    _EnvSetupDone,
    _EnvSetupFailed,
    _FileOpDone,
    _HealthCheckResult,
    _HealthStreamEnded,
    _IdleTick,
    _LocalInstallDone,
    _PollResult,
    _PostBootstrapFailed,
    _RemoteTaskDone,
    _ResultReconciled,
    _SnapshotFailed,
    _SnapshotSaved,
    _UserCodeSyncDone,
    _WorkerDiscovered,
    _WorkerDiscoveryFailed,
    _WorkerFailed,
    _WorkerStarted,
)
from .state import NodeId, NodeState, PendingTask


def _should_announce_idle(
    inflight_count: int,
    last_task_at: float,
    now: float,
    threshold: float,
    announced: bool,
) -> bool:
    """Decide whether to emit NodeBecameIdle on this idle tick.

    Parameters
    ----------
    inflight_count : int
        Number of tasks currently inflight on the node.
    last_task_at : float
        Monotonic timestamp of the most recent task activity.
    now : float
        Current monotonic time.
    threshold : float
        Idle window in seconds required before announcing.
    announced : bool
        Whether idle has already been announced in the current idle window.

    Returns
    -------
    bool
        True iff the node is empty, has been quiet longer than ``threshold``,
        and has not yet announced this idle window.
    """
    return (
        inflight_count == 0
        and now - last_task_at > threshold
        and not announced
    )


def _evaluate_health(
    healthy: bool,
    reason: str | None,
    current: int,
    threshold: int,
) -> tuple[int, str | None]:
    """Decide the next failure count and Preempted reason for a health tick.

    Parameters
    ----------
    healthy : bool
        Outcome of the most recent check.
    reason : str | None
        User-provided failure reason (ignored when ``healthy``).
    current : int
        Current consecutive failure count.
    threshold : int
        Failures required to trigger replacement.

    Returns
    -------
    tuple[int, str | None]
        ``(new_count, preempt_reason)``. ``preempt_reason`` is non-None
        iff the actor must send Preempted; ``new_count`` is the updated
        counter to persist in NodeState.
    """
    if healthy:
        return 0, None
    new_count = current + 1
    if new_count >= threshold:
        return new_count, f"health check failed {new_count}x: {reason or 'unspecified'}"
    return new_count, None


type ReconcileDecision = Literal["succeeded", "failed", "wait", "interrupted"]


def _classify_reconcile_result(
    reply: Any,
    error: str | None,
) -> ReconcileDecision:
    """Map a ``GetResult`` reply to a reconciliation decision.

    Parameters
    ----------
    reply : ResultPending | ResultDone | ResultUnknown | None
        Worker's response to ``GetResult``. ``None`` means the
        reconciliation ask itself raised (network failure, timeout).
    error : str | None
        Error string when ``reply`` is ``None``.

    Returns
    -------
    ReconcileDecision
        - ``"succeeded"`` / ``"failed"``: surface to caller and remove from inflight.
        - ``"wait"``: task still running on worker; original ask will resolve.
        - ``"interrupted"``: reply lost or worker has no record; surface
          ``TaskInterrupted`` so the task manager retries on another node.
    """
    from skyward.infra.worker import (
        ResultDone,
        ResultPending,
        ResultUnknown,
    )
    from skyward.infra.worker import (
        TaskFailed as WorkerTaskFailed,
    )
    from skyward.infra.worker import (
        TaskSucceeded as WorkerTaskSucceeded,
    )

    match reply:
        case ResultPending():
            return "wait"
        case ResultDone(result=WorkerTaskSucceeded()):
            return "succeeded"
        case ResultDone(result=WorkerTaskFailed()):
            return "failed"
        case ResultUnknown():
            return "interrupted"
        case _:
            return "interrupted"


def node_actor(
    node_id: NodeId,
    pool: ActorRef,
    ssh_timeout: float = float(DEFAULT_SSH_TIMEOUT),
    ssh_retry_interval: float = 5.0,
    poll_interval: float = 5.0,
    poll_timeout: float = float(DEFAULT_PROVISION_TIMEOUT),
    bootstrap_timeout: float = float(DEFAULT_BOOTSTRAP_TIMEOUT),
    _skip_monitor: bool = False,
    ca: CertificateAuthority | None = None,
    autoscaler: ActorRef | None = None,
    scale_down_idle_seconds: float = 60.0,
    idle_tick_interval: float = 15.0,
) -> Behavior[NodeMsg]:
    """A merged node tells this story: idle → polling → … → active."""

    log = logger.bind(actor="node", node_id=node_id)

    # ── common handlers ──────────────────────────────────────────────

    def _stop_transport(ref: ActorRef | None) -> None:
        if ref:
            ref.tell(StopTransport())

    def _fail_and_stop(s: NodeState, reason: str) -> Behavior[NodeMsg]:
        _stop_transport(s.transport_ref)
        iid = s.ni.instance.id if s.ni else None
        pool.tell(NodeExhausted(node_id=node_id, reason=reason, instance_id=iid))
        return Behaviors.stopped()

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
                return _fail_and_stop(s, reason)
            case Terminated():
                return _fail_and_stop(s, "child stopped")
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
                case Adopt(cluster=cluster, provider=provider, instance=instance):
                    log.info("Node {nid} adopting instance {iid}", nid=node_id, iid=instance.id)
                    s = NodeState(cluster=cluster, provider=provider)
                    return _start_polling(ctx, s, instance, reattach=True)
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── polling ───────────────────────────────────────────────────────

    def _start_polling(
        ctx: ActorContext[NodeMsg],
        s: NodeState,
        instance: Any,
        reattach: bool = False,
    ) -> Behavior[NodeMsg]:
        instance_id = instance.id
        provider_name = s.cluster.spec.provider or "aws"
        ni = _bind_to_node(instance, node_id, provider_name, s.cluster)
        s = replace(s, ni=ni)
        start_time = asyncio.get_event_loop().time()

        async def _do_poll() -> _PollResult:
            updated_cluster, inst = await s.provider.get_instance(s.cluster, instance_id)
            return _PollResult(instance=inst, cluster=updated_cluster)

        ctx.pipe_to_self(
            _do_poll(),
            on_failure=lambda e: _PollResult(instance=None),
        )
        return polling(s, instance_id, start_time, reattach)

    def polling(
        s: NodeState,
        instance_id: str,
        start_time: float,
        reattach: bool = False,
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
                    return _start_connecting(ctx, replace(s, cluster=c, ni=ni), reattach)
                case _PollResult(instance=inst, cluster=updated) if (
                    inst and inst.status == "exited"
                ):
                    c = updated or s.cluster
                    log.warning(
                        "Instance {iid} exited",
                        iid=instance_id,
                    )
                    return _fail_and_stop(replace(s, cluster=c), f"Instance {instance_id} exited")
                case _PollResult(cluster=updated):
                    c = updated or s.cluster
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > poll_timeout:
                        log.error("Instance not ready within {t}s", t=poll_timeout)
                        return _fail_and_stop(replace(s, cluster=c), f"Instance not ready within {poll_timeout}s")

                    ns = replace(s, cluster=c)

                    async def _poll_after_delay() -> _PollResult:
                        await asyncio.sleep(poll_interval)
                        updated_cluster, inst = await ns.provider.get_instance(ns.cluster, instance_id)
                        return _PollResult(instance=inst, cluster=updated_cluster)

                    ctx.pipe_to_self(
                        _poll_after_delay(),
                        on_failure=lambda e: _PollResult(instance=None),
                    )
                    return polling(ns, instance_id, start_time, reattach)
            return _common(ctx, msg, s, lambda ns: polling(ns, instance_id, start_time, reattach))

        return Behaviors.receive(receive)

    # ── connecting ────────────────────────────────────────────────────

    def _start_connecting(
        ctx: ActorContext[NodeMsg], s: NodeState, reattach: bool = False,
    ) -> Behavior[NodeMsg]:
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
                password=ni.ssh_password,
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
        return connecting(replace(s, transport_ref=transport_ref), reattach)

    def connecting(s: NodeState, reattach: bool = False) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case _Connected(transport_ref=tref, local_port=lp, instance=ni):
                    log.info("SSH tunnel established (port={port})", port=lp)
                    if ni is not None:
                        pool.tell(NodeConnected(node_id=node_id, instance=ni))
                    ns = replace(s, transport_ref=tref, local_port=lp)
                    if reattach:
                        return _enter_reattach_ready(ctx, ns)
                    return _start_bootstrapping(ctx, ns)
                case _ConnectionFailed(error=error):
                    log.error("SSH connection failed: {error}", error=error)
                    return _fail_and_stop(s, error)
                case ConnectionFailed(error=error):
                    log.error("Transport permanently failed: {error}", error=error)
                    return _fail_and_stop(s, error)
            return _common(ctx, msg, s, lambda ns: connecting(ns, reattach))

        return Behaviors.receive(receive)

    # ── reattach ready (skip bootstrap + worker launch) ───────────────

    def _enter_reattach_ready(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        ni = s.ni
        tref = s.transport_ref
        assert ni is not None
        assert tref is not None
        if not _skip_monitor:
            monitor_ref = ctx.spawn(
                instance_monitor(
                    info=ni, transport=tref, event_listener=pool, reply_to=ctx.self,
                ),
                f"monitor-{ni.instance.id}",
            )
            ctx.watch(monitor_ref)
        private_ip = ni.instance.private_ip or ni.instance.ip or ""
        log.info("Node {nid} reattached, entering ready", nid=node_id)
        pool.tell(NodeBecameReady(
            node_id=node_id, instance=ni, local_port=s.local_port, private_ip=private_ip,
        ))
        return ready(s)

    # ── bootstrapping ─────────────────────────────────────────────────

    def _start_bootstrapping(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        ni = s.ni
        tref = s.transport_ref
        assert ni is not None
        assert tref is not None
        log.info("Starting bootstrap")
        bs_start = asyncio.get_event_loop().time()

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
            return _enter_post_bootstrap(ctx, s, bs_start)

        ctx.pipe_to_self(
            run_bootstrap(tref, ni, s.cluster, s.cluster.spec),
            mapper=lambda _: _BootstrapUploaded(),
            on_failure=lambda e: _BootstrapUploadFailed(error=str(e)),
        )
        return bootstrapping(s, bs_start)

    def _check_bootstrap_timeout(
        s: NodeState, bs_start: float,
    ) -> Behavior[NodeMsg] | None:
        elapsed = asyncio.get_event_loop().time() - bs_start
        if elapsed > bootstrap_timeout:
            log.error("Bootstrap timed out after {t:.0f}s", t=elapsed)
            return _fail_and_stop(s, f"Bootstrap timed out after {elapsed:.0f}s")
        return None

    def bootstrapping(s: NodeState, bs_start: float) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            if timeout_behavior := _check_bootstrap_timeout(s, bs_start):
                return timeout_behavior
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
                    return _enter_post_bootstrap(ctx, replace(s, ni=final_ni), bs_start)
                case BootstrapDone(success=False, error=error):
                    log.error("Bootstrap failed: {error}", error=error)
                    return _fail_and_stop(s, error or "bootstrap failed")
                case _BootstrapUploaded():
                    log.info("Bootstrap script uploaded and started")
                    return Behaviors.same()
                case _BootstrapUploadFailed(error=error):
                    log.error("Bootstrap upload failed: {error}", error=error)
                    return _fail_and_stop(s, error)
            return _common(ctx, msg, s, lambda ns: bootstrapping(ns, bs_start))

        return Behaviors.receive(receive)

    # ── post_bootstrap ────────────────────────────────────────────────

    def _enter_post_bootstrap(
        ctx: ActorContext[NodeMsg], s: NodeState, bs_start: float,
    ) -> Behavior[NodeMsg]:
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
            return post_bootstrap(s, bs_start)

        if spec.image and getattr(spec.image, "includes", None):
            ctx.pipe_to_self(
                sync_user_code(tref, ni, spec, s.cluster),
                mapper=lambda _, m=ni: _UserCodeSyncDone(instance=m),
                on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
            )
            return post_bootstrap(s, bs_start)

        return _enter_ready(ctx, s, bs_start)

    def post_bootstrap(s: NodeState, bs_start: float) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            if timeout_behavior := _check_bootstrap_timeout(s, bs_start):
                return timeout_behavior
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
                    return _enter_ready(ctx, s, bs_start)
                case _UserCodeSyncDone():
                    return _enter_ready(ctx, s, bs_start)
                case _PostBootstrapFailed(error=err):
                    return _fail_and_stop(s, err)
            return _common(ctx, msg, s, lambda ns: post_bootstrap(ns, bs_start))

        return Behaviors.receive(receive)

    # ── ready (worker started, waiting for JoinCluster) ───────────────

    def _enter_ready(
        ctx: ActorContext[NodeMsg], s: NodeState, bs_start: float,
    ) -> Behavior[NodeMsg]:
        ni = s.ni
        assert ni is not None
        spec = s.cluster.spec

        if not spec.cluster:
            log.info("Starting worker (standalone mode)")
            return _start_worker_process(ctx, s, bs_start)

        is_head = node_id == 0

        if is_head:
            head_private = ni.instance.private_ip or ni.instance.ip or ""
            pool.tell(
                HeadAddressKnown(
                    head_addr=head_private,
                    casty_port=25520,
                    num_nodes=spec.nodes.desired,
                    worker_concurrency=spec.worker.concurrency,
                    worker_executor=spec.worker.resolved_executor,
                    worker_reuse_processes=spec.worker.reuse_processes,
                )
            )
            log.info("Starting worker (role=head)")
            return _start_worker_process(ctx, s, bs_start)

        if s.head_info is not None:
            log.info("Starting worker (role=worker)")
            return _start_worker_process(ctx, s, bs_start)

        log.info("Waiting for head address before starting worker")
        return waiting_for_head(s, bs_start)

    def waiting_for_head(s: NodeState, bs_start: float) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            if timeout_behavior := _check_bootstrap_timeout(s, bs_start):
                return timeout_behavior
            match msg:
                case HeadAddressKnown() as h:
                    log.info("Head address received, starting worker")
                    return _start_worker_process(ctx, replace(s, head_info=h), bs_start)
            return _common(ctx, msg, s, lambda ns: waiting_for_head(ns, bs_start))

        return Behaviors.receive(receive)

    # ── starting worker ───────────────────────────────────────────────

    def _start_worker_process(
        ctx: ActorContext[NodeMsg], s: NodeState, bs_start: float,
    ) -> Behavior[NodeMsg]:
        ni = s.ni
        tref = s.transport_ref
        assert ni is not None
        assert tref is not None
        ctx.pipe_to_self(
            do_start_worker(tref, s.local_port, ni, s.head_info, node_id, s.cluster, s.cluster.spec, ca=ca),
            mapper=lambda result: _WorkerStarted(local_port=result[0], private_ip=result[1]),
            on_failure=lambda e: _WorkerFailed(error=str(e)),
        )
        return starting_worker(s, bs_start)

    def starting_worker(s: NodeState, bs_start: float) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            if timeout_behavior := _check_bootstrap_timeout(s, bs_start):
                return timeout_behavior
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
                    return _fail_and_stop(s, error)
            return _common(ctx, msg, s, lambda ns: starting_worker(ns, bs_start))

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
                        discover_own_worker(client, ni, standalone=not s.cluster.spec.cluster),
                        mapper=lambda ref: _WorkerDiscovered(worker_ref=ref),
                        on_failure=lambda e: _WorkerDiscoveryFailed(error=str(e)),
                    )
                    return joining(replace(
                        s, client=client, pool_info_json=pij,
                        env_vars=MappingProxyType(ev), around_app_hooks=hooks,
                        around_process_hooks=phooks,
                    ))
                case ConnectionLost():
                    log.warning("Connection lost while ready, transport reconnecting")
                    return Behaviors.same()
                case ConnectionRestored():
                    log.info("Connection restored while ready")
                    return Behaviors.same()
                case ConnectionFailed(error=error):
                    log.error("Transport permanently failed while ready: {err}", err=error)
                    return _fail_and_stop(s, "connection lost")
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
                    return _fail_and_stop(s, error)
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
                    return _fail_and_stop(s, error)
            return _common(ctx, msg, s, joining_env_setup)

        return Behaviors.receive(receive)

    # ── active ────────────────────────────────────────────────────────

    def _schedule_idle_tick(ctx: ActorContext[NodeMsg]) -> None:
        async def _tick() -> _IdleTick:
            await asyncio.sleep(idle_tick_interval)
            return _IdleTick()

        ctx.pipe_to_self(
            _tick(),
            mapper=lambda r: r,
            on_failure=lambda _: _IdleTick(),
        )

    def _enter_active(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        if (hc := s.cluster.spec.health_checker) is not None:
            log.info(
                "Node {nid} warming up, awaiting first health check",
                nid=node_id,
            )
            _dispatch_health_check(ctx, s, hc)
            return warming(replace(s, health_failures=0))
        return _activate_now(ctx, s)

    def _activate_now(ctx: ActorContext[NodeMsg], s: NodeState) -> Behavior[NodeMsg]:
        """Flush queued tasks, announce ready, schedule idle tick."""
        new_inflight: MappingProxyType[str, ActorRef] = MappingProxyType({})
        counter = 0
        for pt in s.pending_tasks:
            tid = pt.task_id or str(counter)
            _dispatch_task(ctx, s, tid, pt.fn, pt.args, pt.kwargs, pt.timeout)
            new_inflight = MappingProxyType({**new_inflight, tid: pt.reply_to})
            counter += 1

        pool.tell(NodeActivated(
            node_id=node_id,
            node_ref=ctx.self,
            slots=s.cluster.spec.worker.concurrency,
        ))
        now = time.monotonic()
        _schedule_idle_tick(ctx)
        return active(replace(
            s,
            inflight=new_inflight,
            task_counter=counter,
            pending_tasks=(),
            last_task_at=now,
            idle_announced=False,
            health_failures=0,
        ))

    def _dispatch_task(
        ctx: ActorContext[NodeMsg],
        s: NodeState,
        tid: str,
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout: float,
    ) -> None:
        t0 = time.monotonic()
        ctx.pipe_to_self(
            execute_with_streaming(
                s.client, s.worker_ref, fn, args, kwargs, timeout, task_id=tid,
            ),
            mapper=lambda result, _tid=tid, _t0=t0: _RemoteTaskDone(  # type: ignore[return-value]
                task_id=_tid,
                value=(result.result if hasattr(result, "result") else RuntimeError(result.error)),
                node_id=node_id,
                reply_to=ctx.self,
                error=not hasattr(result, "result"),
                elapsed=time.monotonic() - _t0,
            ),
            on_failure=lambda e, _tid=tid, _t0=t0: _RemoteTaskDone(  # type: ignore[return-value]
                task_id=_tid,
                value=RuntimeError(str(e)),
                node_id=node_id,
                reply_to=ctx.self,
                error=True,
                elapsed=time.monotonic() - _t0,
            ),
        )

    def _dispatch_health_check(
        ctx: ActorContext[NodeMsg],
        s: NodeState,
        hc: HealthChecker,
    ) -> None:
        from skyward.infra.streaming import _StreamHandle
        from skyward.infra.worker import ExecuteTask
        from skyward.infra.worker import TaskSucceeded as WorkerTaskSucceeded

        fn = partial(hc_loop, hc.fn, hc.interval, hc.timeout, hc.initial_delay)
        self_ref = ctx.self
        client = s.client
        worker_ref = s.worker_ref

        async def _run() -> _HealthStreamEnded:
            try:
                result = await client.ask(
                    worker_ref,
                    lambda rto: ExecuteTask(
                        fn=fn, args=(), kwargs={},
                        reply_to=rto, input_streams=(),
                    ),
                    timeout=60.0,
                )
                match result:
                    case WorkerTaskSucceeded(result=_StreamHandle(producer_ref=pref)):
                        source = await _resolve_output_stream(client, pref)
                        async for elem in source:
                            match elem:
                                case ("ok", _):
                                    self_ref.tell(_HealthCheckResult(healthy=True))
                                case ("fail", reason):
                                    self_ref.tell(_HealthCheckResult(
                                        healthy=False, reason=reason,
                                    ))
                                case _:
                                    self_ref.tell(_HealthCheckResult(
                                        healthy=False,
                                        reason=f"malformed yield: {elem!r}",
                                    ))
                        return _HealthStreamEnded(reason="stream closed")
                    case _:
                        err = getattr(result, "error", repr(result))
                        return _HealthStreamEnded(reason=f"failed to start: {err}")
            except Exception as e:
                return _HealthStreamEnded(reason=f"drain failed: {e!r}")

        ctx.pipe_to_self(
            _run(),
            mapper=lambda r: r,
            on_failure=lambda e: _HealthStreamEnded(reason=f"dispatch failed: {e!r}"),
        )

    def warming(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case _HealthCheckResult(healthy=True):
                    log.info(
                        "Node {nid} first health check passed, activating",
                        nid=node_id,
                    )
                    return _activate_now(ctx, s)
                case _HealthCheckResult(healthy=False, reason=reason):
                    hc = s.cluster.spec.health_checker
                    if hc is None:
                        return Behaviors.same()
                    new_count, preempt_reason = _evaluate_health(
                        healthy=False, reason=reason,
                        current=s.health_failures,
                        threshold=hc.consecutive_failures,
                    )
                    log.warning(
                        "Node {nid} warming health check failed ({n}/{max}): {reason}",
                        nid=node_id, n=new_count,
                        max=hc.consecutive_failures,
                        reason=reason or "(no reason)",
                    )
                    if preempt_reason is not None:
                        ctx.self.tell(Preempted(reason=preempt_reason))
                    return warming(replace(s, health_failures=new_count))
                case _HealthStreamEnded(reason=reason):
                    log.error(
                        "Node {nid} health stream ended during warming: {reason}",
                        nid=node_id, reason=reason,
                    )
                    ctx.self.tell(Preempted(reason=f"health stream ended: {reason}"))
                    return Behaviors.same()
                case ConnectionFailed(error=error):
                    log.error(
                        "Transport permanently failed during warming on node {nid}: {err}",
                        nid=node_id, err=error,
                    )
                    return _fail_and_stop(s, f"connection failed: {error}")
            return _common(ctx, msg, s, warming)

        return Behaviors.receive(receive)

    def active(s: NodeState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return active(replace(s, head_info=h))
                case NodeFileOp(op=op, path=path, content=content, timeout=fo_timeout, reply_to=fo_rt):
                    if s.transport_ref is None:
                        fo_rt.tell(NodeFileResult(node_id=node_id, success=False, error="transport not connected"))
                        return Behaviors.same()
                    ctx.pipe_to_self(
                        _run_file_op(s.transport_ref, node_id, op, path, content, fo_timeout),
                        mapper=lambda result, _rt=fo_rt: _FileOpDone(result=result, reply_to=_rt),
                        on_failure=lambda e, _rt=fo_rt: _FileOpDone(
                            result=NodeFileResult(node_id=node_id, success=False, error=str(e)),
                            reply_to=_rt,
                        ),
                    )
                    return Behaviors.same()
                case _FileOpDone(result=result, reply_to=fo_rt):
                    fo_rt.tell(result)
                    return Behaviors.same()
                case ExecuteOnNode() as ex:
                    if not s.transport_connected:
                        pt = PendingTask(
                            ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout,
                        )
                        log.debug(
                            "Node {nid} transport down, queuing task {tid} (queue={n})",
                            nid=node_id, tid=ex.task_id, n=len(s.pending_tasks) + 1,
                        )
                        return active(replace(s, pending_tasks=(*s.pending_tasks, pt)))
                    local_tid = ex.task_id or str(s.task_counter)
                    log.debug("Node {nid} dispatching task {tid}", nid=node_id, tid=local_tid)
                    _dispatch_task(ctx, s, local_tid, ex.fn, ex.args, ex.kwargs, ex.timeout)
                    new_inflight = MappingProxyType({**s.inflight, local_tid: ex.reply_to})
                    new_s = replace(
                        s,
                        inflight=new_inflight,
                        task_counter=s.task_counter + 1,
                        last_task_at=time.monotonic(),
                    )
                    if s.idle_announced and autoscaler is not None:
                        autoscaler.tell(NodeBecameBusy(node_id=node_id))
                        new_s = replace(new_s, idle_announced=False)
                    return active(new_s)
                case _RemoteTaskDone(task_id=tid, value=value, error=is_err, elapsed=elapsed):
                    log.debug("Node {nid} received task result (tid={tid})", nid=node_id, tid=tid)
                    caller = s.inflight.get(tid)
                    if caller is None:
                        # already reconciled via GetResult — discard duplicate
                        return Behaviors.same()
                    result = (
                        TaskFailed(error=value, node_id=node_id, task_id=tid)
                        if is_err
                        else TaskSucceeded(value=value, node_id=node_id, task_id=tid, elapsed=elapsed)
                    )
                    caller.tell(result)
                    new_inflight = MappingProxyType({k: v for k, v in s.inflight.items() if k != tid})
                    return active(replace(s, inflight=new_inflight, last_task_at=time.monotonic()))
                case _ResultReconciled(task_id=tid, reply=reply, error=err):
                    caller = s.inflight.get(tid)
                    if caller is None:
                        return Behaviors.same()  # original ask resolved first
                    from skyward.infra.worker import (
                        ResultDone,
                    )
                    from skyward.infra.worker import (
                        TaskFailed as WorkerTaskFailed,
                    )
                    from skyward.infra.worker import (
                        TaskSucceeded as WorkerTaskSucceeded,
                    )

                    match reply:
                        case ResultDone(result=WorkerTaskSucceeded() as ws):
                            log.info(
                                "Reconcile {tid} on node {nid}: recovered TaskSucceeded",
                                tid=tid, nid=node_id,
                            )
                            caller.tell(TaskSucceeded(
                                value=ws.result,
                                node_id=node_id, task_id=tid, elapsed=0.0,
                            ))
                        case ResultDone(result=WorkerTaskFailed() as wf):
                            log.info(
                                "Reconcile {tid} on node {nid}: recovered TaskFailed",
                                tid=tid, nid=node_id,
                            )
                            caller.tell(TaskFailed(
                                error=RuntimeError(f"{wf.error}\n{wf.traceback}"),
                                node_id=node_id, task_id=tid,
                            ))
                        case _ if _classify_reconcile_result(reply, err) == "wait":
                            log.debug(
                                "Reconcile {tid} on node {nid}: still running",
                                tid=tid, nid=node_id,
                            )
                            return Behaviors.same()
                        case _:
                            log.warning(
                                "Reconcile {tid} on node {nid} unrecoverable: {err}",
                                tid=tid, nid=node_id,
                                err=err or "worker has no record",
                            )
                            caller.tell(TaskInterrupted(
                                error=RuntimeError(f"Result lost on node {node_id}"),
                                node_id=node_id, task_id=tid,
                            ))
                    new_inflight = MappingProxyType({k: v for k, v in s.inflight.items() if k != tid})
                    return active(replace(s, inflight=new_inflight, last_task_at=time.monotonic()))
                case ConnectionLost(error=error):
                    log.warning(
                        "Connection lost on node {nid}: {err}, transport reconnecting; pausing dispatch",
                        nid=node_id, err=error,
                    )
                    return active(replace(s, transport_connected=False))
                case ConnectionRestored():
                    log.info(
                        "Transport reconnected on node {nid}: reconciling {n_inflight} inflight, draining {n_pending} pending",
                        nid=node_id,
                        n_inflight=len(s.inflight),
                        n_pending=len(s.pending_tasks),
                    )
                    # fan out a GetResult ask per inflight tid to recover lost replies
                    from skyward.infra.worker import GetResult

                    for inflight_tid in list(s.inflight):
                        ctx.pipe_to_self(
                            s.client.ask(
                                s.worker_ref,
                                lambda rto, _tid=inflight_tid: GetResult(
                                    task_id=_tid, reply_to=rto,
                                ),
                                timeout=10.0,
                            ),
                            mapper=lambda reply, _tid=inflight_tid: _ResultReconciled(
                                task_id=_tid, reply=reply,
                            ),
                            on_failure=lambda e, _tid=inflight_tid: _ResultReconciled(
                                task_id=_tid, reply=None, error=repr(e),
                            ),
                        )
                    # drain pending normally
                    new_inflight = s.inflight
                    new_counter = s.task_counter
                    for pt in s.pending_tasks:
                        tid = pt.task_id or str(new_counter)
                        _dispatch_task(ctx, s, tid, pt.fn, pt.args, pt.kwargs, pt.timeout)
                        new_inflight = MappingProxyType({**new_inflight, tid: pt.reply_to})
                        new_counter += 1
                    return active(replace(
                        s,
                        transport_connected=True,
                        pending_tasks=(),
                        inflight=new_inflight,
                        task_counter=new_counter,
                        last_task_at=time.monotonic() if s.pending_tasks else s.last_task_at,
                    ))
                case ConnectionFailed(error=error):
                    log.error(
                        "Transport permanently failed on node {nid}: {err}",
                        nid=node_id, err=error,
                    )
                    for tid, caller in s.inflight.items():
                        caller.tell(
                            TaskInterrupted(
                                error=RuntimeError(f"Node {node_id} connection failed: {error}"),
                                node_id=node_id, task_id=tid,
                            )
                        )
                    _stop_transport(s.transport_ref)
                    iid = s.ni.instance.id if s.ni else None
                    pool.tell(NodeExhausted(node_id=node_id, reason=f"connection failed: {error}", instance_id=iid))
                    return Behaviors.stopped()
                case Preempted(reason=reason):
                    log.warning("Preempted while active: {reason}", reason=reason)
                    _stop_transport(s.transport_ref)
                    for tid, caller in s.inflight.items():
                        caller.tell(
                            TaskInterrupted(
                                error=RuntimeError(f"Node {node_id} preempted: {reason}"),
                                node_id=node_id, task_id=tid,
                            )
                        )
                    iid = s.ni.instance.id if s.ni else None
                    pool.tell(NodeExhausted(node_id=node_id, reason=reason, instance_id=iid))
                    return Behaviors.stopped()
                case Terminated():
                    log.warning("Child died while active, marking node lost")
                    for tid, caller in s.inflight.items():
                        caller.tell(
                            TaskInterrupted(
                                error=RuntimeError(f"Node {node_id} child stopped"),
                                node_id=node_id, task_id=tid,
                            )
                        )
                    _stop_transport(s.transport_ref)
                    iid = s.ni.instance.id if s.ni else None
                    pool.tell(NodeExhausted(node_id=node_id, reason="child stopped", instance_id=iid))
                    return Behaviors.stopped()
                case _IdleTick():
                    now = time.monotonic()
                    new_s = s
                    if _should_announce_idle(
                        inflight_count=len(s.inflight),
                        last_task_at=s.last_task_at,
                        now=now,
                        threshold=scale_down_idle_seconds,
                        announced=s.idle_announced,
                    ) and autoscaler is not None:
                        autoscaler.tell(NodeBecameIdle(node_id=node_id))
                        new_s = replace(s, idle_announced=True)
                    _schedule_idle_tick(ctx)
                    return active(new_s)
                case _HealthCheckResult(healthy=healthy, reason=reason):
                    hc = s.cluster.spec.health_checker
                    if hc is None:
                        return Behaviors.same()
                    new_count, preempt_reason = _evaluate_health(
                        healthy=healthy, reason=reason,
                        current=s.health_failures,
                        threshold=hc.consecutive_failures,
                    )
                    if healthy:
                        log.debug("Node {nid} health check ok", nid=node_id)
                    else:
                        log.warning(
                            "Node {nid} health check failed ({n}/{max}): {reason}",
                            nid=node_id, n=new_count,
                            max=hc.consecutive_failures,
                            reason=reason or "(no reason)",
                        )
                    if preempt_reason is not None:
                        ctx.self.tell(Preempted(reason=preempt_reason))
                    if new_count == s.health_failures:
                        return Behaviors.same()
                    return active(replace(s, health_failures=new_count))
                case _HealthStreamEnded(reason=reason):
                    log.error(
                        "Node {nid} health stream ended: {reason}",
                        nid=node_id, reason=reason,
                    )
                    ctx.self.tell(Preempted(reason=f"health stream ended: {reason}"))
                    return Behaviors.same()
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
            return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()
