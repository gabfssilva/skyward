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
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, Final

from casty import ActorContext, ActorRef, Behavior, Behaviors, Terminated

from skyward.actors.messages import (
    BootstrapDone,
    ExecuteOnNode,
    HeadAddressKnown,
    JoinCluster,
    NodeBecameReady,
    NodeInstance,
    NodeLost,
    NodeMsg,
    Preempted,
    Provision,
    TaskResult,
    _bind_to_node,
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
    transport_run,
    transport_write_bytes,
    transport_write_file,
)
from skyward.observability.logger import logger
from skyward.providers.provider import WarmableProvider

_PYTHON_VERSION: Final = f"{sys.version_info.major}.{sys.version_info.minor}"

type NodeId = int


class PythonVersionMismatchError(RuntimeError):
    def __init__(self, local: str, remote: str) -> None:
        self.local = local
        self.remote = remote
        super().__init__(
            f"Python version mismatch: local={local}, remote={remote}. "
            f"Cloudpickle cannot safely serialize bytecode across versions. "
            f"Set Image(python='{local}') or use Image(python='auto')."
        )


def _check_python_version(remote_version: str) -> None:
    if remote_version != _PYTHON_VERSION:
        raise PythonVersionMismatchError(local=_PYTHON_VERSION, remote=remote_version)


@dataclass(frozen=True, slots=True)
class _PendingTask:
    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[Any]
    task_id: str
    timeout: float


@dataclass(frozen=True, slots=True)
class NodeState:
    cluster: Any
    provider: Any
    ni: NodeInstance | None = None
    transport_ref: ActorRef | None = None
    local_port: int = 0
    head_info: HeadAddressKnown | None = None
    pending_tasks: tuple[_PendingTask, ...] = ()
    client: Any = None
    worker_ref: ActorRef | None = None
    pool_info_json: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    around_app_hooks: tuple[tuple[str, Any], ...] = ()
    around_process_hooks: tuple[tuple[str, Any], ...] = ()
    inflight: dict[str, ActorRef] = field(default_factory=dict)
    task_counter: int = 0


def node_actor(
    node_id: NodeId,
    pool: ActorRef,
    ssh_timeout: float = 300.0,
    ssh_retry_interval: float = 5.0,
    poll_interval: float = 5.0,
    poll_timeout: float = 300.0,
    _skip_monitor: bool = False,
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
        pt = _PendingTask(ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout)
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
            _run_bootstrap(tref, ni, s.cluster, s.cluster.spec),
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
                _install_local_skyward(tref, ni, s.cluster),
                mapper=lambda _, m=ni: _LocalInstallDone(instance=m),
                on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
            )
            return post_bootstrap(s)

        if spec.image and getattr(spec.image, "includes", None):
            ctx.pipe_to_self(
                _sync_user_code(tref, ni, spec, s.cluster),
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
                            _sync_user_code(tref, info, spec, s.cluster),
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
                    num_nodes=spec.nodes,
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
            _do_start_worker(tref, s.local_port, ni, s.head_info, node_id, s.cluster, s.cluster.spec),
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
                        _discover_own_worker(client, ni),
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
                        _setup_worker_env(
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
            _execute_with_streaming(s.client, s.worker_ref, fn, args, kwargs, timeout),
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
                        _discover_own_worker(client, ni),
                        mapper=lambda ref: _WorkerDiscovered(worker_ref=ref),
                        on_failure=lambda e: _WorkerDiscoveryFailed(error=str(e)),
                    )
                    return active(replace(s, client=client))
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
            _terminate_and_replace(s.provider, s.cluster, instance_id),
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


# ─── Helpers (module-level) ───────────────────────────────────────────


async def _do_start_worker(
    transport_ref: ActorRef,
    local_port: int,
    ni: NodeInstance,
    head_info: HeadAddressKnown | None,
    node_id: int,
    cluster: Any,
    spec: Any,
) -> tuple[int, str]:
    import logging as _logging

    from skyward.providers.bootstrap.compose import EMIT_SH_PATH, SKYWARD_DIR

    _logging.getLogger("casty").setLevel(_logging.ERROR)

    log = logger.bind(actor="node", instance_id=ni.instance.id)

    private_ip = ni.instance.private_ip or ni.instance.ip or ""
    casty_port = head_info.casty_port if head_info else 25520
    num_nodes = head_info.num_nodes if head_info else spec.nodes
    concurrency = head_info.worker_concurrency if head_info else spec.worker.concurrency
    executor = head_info.worker_executor if head_info else spec.worker.resolved_executor

    seeds = f"{head_info.head_addr}:{casty_port}" if head_info and node_id != 0 else ""

    venv_dir = f"{SKYWARD_DIR}/.venv"
    python_bin = f"{venv_dir}/bin/python"
    ssh_user = ni.ssh_user or cluster.ssh_user
    use_sudo = ssh_user != "root"

    host = private_ip if node_id != 0 else (head_info.head_addr if head_info else private_ip)

    seeds_arg = f"--seeds {seeds} " if seeds else ""
    casty_cmd = (
        f'nohup {python_bin} -c "from skyward.infra.worker import cli; cli()" '
        f"--node-id {node_id} --port {casty_port} "
        f"--num-nodes {num_nodes} --host {host} "
        f"--workers-per-node {concurrency} "
        f"--worker-executor {executor} "
        f"{seeds_arg}"
        f"> /var/log/casty.log 2>&1 & echo $!"
    )
    tail_inner = (
        f"source {EMIT_SH_PATH} && "
        f"tail -f /var/log/casty.log 2>/dev/null | while IFS= read -r line; do "
        f'emit_console "$line"; done'
    )
    tail_cmd = f"nohup bash -c '{tail_inner}' </dev/null >/dev/null 2>&1 &"

    if use_sudo:
        casty_cmd = f"sudo bash -c '{casty_cmd}'"
        tail_cmd = f"sudo {tail_cmd}"

    exit_code, stdout, stderr = await transport_run(transport_ref, casty_cmd, timeout=60.0)
    if exit_code != 0:
        raise RuntimeError(f"Failed to start Casty node {node_id}: {stderr}")
    await transport_run(transport_ref, tail_cmd, timeout=10.0)
    log.debug("Casty worker started, PID: {pid}", pid=stdout.strip())

    if spec.image:
        _check_python_version(spec.image.python)

    return local_port, private_ip


async def _install_local_skyward(transport_ref: ActorRef, ni: NodeInstance, cluster: Any) -> None:
    from skyward.providers.common import _build_wheel_install_script, build_wheel

    log = logger.bind(component="bootstrap_ssh")
    log.info("Building local skyward wheel")
    wheel_path = await asyncio.to_thread(build_wheel)

    install_script = _build_wheel_install_script(wheel_name=wheel_path.name)

    def _read_wheel() -> tuple[int, bytes]:
        return wheel_path.stat().st_size, wheel_path.read_bytes()

    wheel_size, wheel_data = await asyncio.to_thread(_read_wheel)
    log.debug("Wheel size: {size:.1f} KB", size=wheel_size / 1024)
    log.info("Uploading wheel {name}", name=wheel_path.name)
    await transport_write_bytes(transport_ref, f"/tmp/{wheel_path.name}", wheel_data)

    await transport_write_file(transport_ref, "/tmp/.install-wheel.sh", install_script)

    ssh_user = ni.ssh_user or cluster.ssh_user
    use_sudo = ssh_user != "root"
    sudo = "sudo " if use_sudo else ""
    log.info("Running wheel install script on {iid}", iid=ni.instance.id)
    exit_code, stdout, stderr = await transport_run(
        transport_ref, f"{sudo}bash /tmp/.install-wheel.sh", timeout=180.0,
    )
    log.debug("Install script output:\n{out}", out=stdout)

    if exit_code != 0:
        raise RuntimeError(f"Wheel install failed (exit {exit_code}): {stderr or stdout}")

    log.info("Local skyward wheel installed on {iid}", iid=ni.instance.id)


async def _sync_user_code(transport_ref: ActorRef, ni: NodeInstance, spec: Any, cluster: Any) -> None:
    from skyward.providers.bootstrap.compose import SKYWARD_DIR
    from skyward.providers.common import build_user_code_tarball

    image = spec.image
    includes = getattr(image, "includes", ())
    if not includes:
        return

    excludes = getattr(image, "excludes", ())
    tarball = build_user_code_tarball(includes=includes, excludes=excludes)

    ssh_user = ni.ssh_user or cluster.ssh_user
    use_sudo = ssh_user != "root"
    sudo = "sudo " if use_sudo else ""
    remote_tar = "/tmp/_user_code.tar.gz"
    site_packages = f"{SKYWARD_DIR}/.venv/lib/python*/site-packages"

    log = logger.bind(component="bootstrap_ssh")
    log.info("Uploading user code ({size:.1f} KB)", size=len(tarball) / 1024)
    await transport_write_bytes(transport_ref, remote_tar, tarball)

    exit_code, stdout, stderr = await transport_run(
        transport_ref,
        f"{sudo}bash -c 'SP=$(echo {site_packages}); "
        f"tar xzf {remote_tar} -C $SP && rm -f {remote_tar}'",
        timeout=60.0,
    )

    if exit_code != 0:
        raise RuntimeError(f"User code extraction failed (exit {exit_code}): {stderr or stdout}")

    log.info("User code uploaded and extracted to site-packages")


async def _run_bootstrap(
    transport_ref: ActorRef,
    ni: NodeInstance,
    cluster: Any,
    spec: Any,
) -> None:
    import base64

    image = spec.image
    if not image:
        return

    ssh_user = ni.ssh_user or cluster.ssh_user
    use_sudo = ssh_user != "root"

    postamble_ops: list = []
    if cluster.resolved_volumes:
        from skyward.providers.bootstrap import mount_volumes, phase

        postamble_ops.append(phase("volumes", mount_volumes(cluster.resolved_volumes)))
    for plugin in spec.plugins:
        if plugin.bootstrap is not None:
            postamble_ops.extend(plugin.bootstrap(cluster))

    postamble = postamble_ops if postamble_ops else None
    ttl = spec.ttl or 0
    shutdown_command = cluster.shutdown_command.format(instance_id=ni.instance.id)
    bootstrap_script = image.generate_bootstrap(
        ttl=ttl,
        shutdown_command=shutdown_command,
        postamble=postamble,
    )

    log = logger.bind(component="bootstrap_ssh")
    log.info(
        "Uploading bootstrap script to {iid} ({size:.1f} KB)",
        iid=ni.instance.id, size=len(bootstrap_script) / 1024,
    )

    encoded = base64.b64encode(bootstrap_script.encode()).decode()
    sudo = "sudo " if use_sudo else ""
    exit_code, _, stderr = await transport_run(
        transport_ref,
        f"{sudo}bash -c \""
        f"mkdir -p /opt/skyward && "
        f"echo '{encoded}' | base64 -d > /opt/skyward/bootstrap.sh && "
        f"chmod +x /opt/skyward/bootstrap.sh\"",
    )
    if exit_code != 0:
        raise RuntimeError(f"Bootstrap upload failed: {stderr}")

    log.info("Running bootstrap on {iid}", iid=ni.instance.id)
    await transport_run(
        transport_ref,
        f"{sudo}bash -c 'nohup /opt/skyward/bootstrap.sh > /opt/skyward/bootstrap.log 2>&1 &'",
    )

    log.info("Bootstrap started on {iid}", iid=ni.instance.id)


async def _terminate_and_replace(provider: Any, cluster: Any, dead_id: str) -> Any:
    log = logger.bind(actor="node")
    try:
        await provider.terminate(cluster, (dead_id,))
    except Exception as e:
        log.warning("Failed to terminate dead instance {iid}: {err}", iid=dead_id, err=e)
    _, instances = await provider.provision(cluster, 1)
    if not instances:
        raise RuntimeError("Failed to provision replacement instance")
    return instances[0]


async def _discover_own_worker(client: Any, ni: NodeInstance | None) -> Any:
    from skyward.infra.worker import WORKER_KEY

    private_ip = ni.instance.private_ip or ni.instance.ip or "" if ni else ""

    deadline = asyncio.get_event_loop().time() + 120.0
    while asyncio.get_event_loop().time() < deadline:
        listing = client.lookup(WORKER_KEY)
        for instance in listing.instances:
            if instance.node.host == private_ip:
                return instance.ref
        await asyncio.sleep(1.0)

    raise RuntimeError(f"Worker for {private_ip} not discovered within 120s")


async def _setup_worker_env(
    client: Any,
    worker_ref: Any,
    pool_info_json: str,
    env_vars: dict[str, str],
    around_app_hooks: tuple[tuple[str, Any], ...] = (),
    around_process_hooks: tuple[tuple[str, Any], ...] = (),
) -> None:
    from skyward.infra.worker import EnterContext, SetProcessHooks

    _frozen_env = dict(env_vars)

    def make_env_cm(
        info_json: str = pool_info_json,
        extra: dict[str, str] = _frozen_env,
    ) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def lifecycle() -> Any:
            import os

            os.environ["COMPUTE_POOL"] = info_json
            for k, v in extra.items():
                os.environ[k] = v
            yield

        return lifecycle()

    await client.ask(
        worker_ref,
        lambda rto: EnterContext(factory=make_env_cm, reply_to=rto),
        timeout=60.0,
    )

    if around_app_hooks:

        def make_hooks_cm(
            hooks: tuple[tuple[str, Any], ...] = around_app_hooks,
        ) -> Any:
            from contextlib import contextmanager

            @contextmanager
            def lifecycle() -> Any:
                from skyward.api.runtime import instance_info
                from skyward.plugins.state import cleanup, ensure_around_app

                info = instance_info()
                for name, factory in hooks:
                    ensure_around_app(name, factory, info)
                try:
                    yield
                finally:
                    cleanup()

            return lifecycle()

        await client.ask(
            worker_ref,
            lambda rto: EnterContext(factory=make_hooks_cm, reply_to=rto),
            timeout=60.0,
        )

    if around_process_hooks:
        await client.ask(
            worker_ref,
            lambda rto: SetProcessHooks(hooks=around_process_hooks, reply_to=rto),
            timeout=60.0,
        )


async def _execute_with_streaming(
    client: Any,
    worker_ref: Any,
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    timeout: float,
) -> Any:
    from skyward.infra.streaming import _stream_param_indices, _StreamHandle, _SyncSource
    from skyward.infra.worker import ExecuteTask
    from skyward.infra.worker import TaskSucceeded as WorkerTaskSucceeded

    indices = _stream_param_indices(fn)
    pump_tasks: list[asyncio.Task[None]] = []
    resolved_args = args
    stream_refs: tuple[tuple[int, Any], ...] = ()

    if indices:
        resolved_args, pump_tasks, stream_refs = await _setup_input_streams(
            client,
            args,
            indices,
        )

    result = await asyncio.wait_for(
        client.ask(
            worker_ref,
            lambda rto: ExecuteTask(
                fn=fn,
                args=resolved_args,
                kwargs=kwargs,
                reply_to=rto,
                input_streams=stream_refs,
            ),
            timeout=timeout,
        ),
        timeout=timeout,
    )

    for t in pump_tasks:
        await t

    match result:
        case WorkerTaskSucceeded(result=_StreamHandle(producer_ref=pref)):
            source = await _resolve_output_stream(client, pref)
            loop = asyncio.get_running_loop()
            return WorkerTaskSucceeded(result=_SyncSource(source, loop), node_id=result.node_id)
        case _:
            return result


async def _setup_input_streams(
    client: Any,
    args: tuple[Any, ...],
    indices: tuple[int, ...],
) -> tuple[tuple[Any, ...], list[asyncio.Task[None]], tuple[tuple[int, Any], ...]]:
    from uuid import uuid4

    from casty import GetSink, stream_producer

    resolved = list(args)
    pump_tasks: list[asyncio.Task[None]] = []
    stream_refs: list[tuple[int, Any]] = []
    loop = asyncio.get_running_loop()

    for i in indices:
        if i >= len(resolved):
            continue
        iterator = resolved[i]

        producer_ref = client.spawn(
            stream_producer(buffer_size=256),
            f"input-{uuid4().hex[:8]}",
        )
        sink = await client.ask(producer_ref, lambda r: GetSink(reply_to=r), timeout=10.0)
        resolved[i] = None
        stream_refs.append((i, producer_ref))

        async def _pump(_sink: Any = sink, _it: Any = iterator) -> None:
            def _drain() -> None:
                try:
                    for elem in _it:
                        fut = asyncio.run_coroutine_threadsafe(_sink.put(elem), loop)
                        fut.result(timeout=300.0)
                finally:
                    fut = asyncio.run_coroutine_threadsafe(_sink.complete(), loop)
                    fut.result(timeout=60.0)

            await asyncio.to_thread(_drain)

        pump_tasks.append(asyncio.create_task(_pump()))

    return tuple(resolved), pump_tasks, tuple(stream_refs)


async def _resolve_output_stream(client: Any, producer_ref: Any) -> Any:
    from uuid import uuid4

    from casty import GetSource, stream_consumer

    name = f"out-consumer-{uuid4().hex[:8]}"
    consumer_ref = client.spawn(
        stream_consumer(producer_ref, timeout=60.0, initial_demand=16),
        name,
    )
    return await client.ask(consumer_ref, lambda r: GetSource(reply_to=r), timeout=10.0)
