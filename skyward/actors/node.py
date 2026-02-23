"""Merged node actor — one actor per compute node.

Tells this story:
  idle → polling → connecting → bootstrapping → post_bootstrap
       → starting_worker → ready → active
       → replacing → polling → … (loop back)

Absorbs the old instance_actor responsibilities (SSH, bootstrap,
worker startup) plus task execution (formerly in executor_actor).
The pool owns the ClusterClient and sends JoinCluster once the
client is ready.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, Final

from casty import ActorContext, ActorRef, Behavior, Behaviors

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
class ActiveState:
    cluster: Any
    provider: Any
    client: Any
    worker_ref: Any
    pending_tasks: tuple[_PendingTask, ...] = ()
    inflight: dict[str, ActorRef] = field(default_factory=dict)
    task_counter: int = 0
    current_node_instance: NodeInstance | None = None
    head_info: HeadAddressKnown | None = None
    transport: Any = None
    listener: Any = None


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

    # ── idle ──────────────────────────────────────────────────────────

    def idle() -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case Provision(cluster=cluster, provider=provider, instance=instance):
                    log.info("Node {nid} provisioning instance {iid}", nid=node_id, iid=instance.id)
                    return _start_polling(ctx, cluster, provider, instance)
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── polling ───────────────────────────────────────────────────────

    def _start_polling(
        ctx: ActorContext[NodeMsg],
        cluster: Any,
        provider: Any,
        instance: Any,
    ) -> Behavior[NodeMsg]:
        instance_id = instance.id
        start_time = asyncio.get_event_loop().time()

        async def _do_poll() -> _PollResult:
            _, inst = await provider.get_instance(cluster, instance_id)
            return _PollResult(instance=inst)

        ctx.pipe_to_self(
            _do_poll(),
            on_failure=lambda e: _PollResult(instance=None),
        )
        return polling(cluster, provider, instance_id, start_time, head_info=None)

    def polling(
        cluster: Any,
        provider: Any,
        instance_id: str,
        start_time: float,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        provider_name = cluster.spec.provider or "aws"

        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return polling(cluster, provider, instance_id, start_time, h, pending_tasks)
                case _PollResult(instance=inst) if (
                    inst and inst.status == "provisioned" and inst.ip
                ):
                    ni = _bind_to_node(inst, node_id, provider_name, cluster)
                    log.info("Instance ready at {ip}", ip=inst.ip)
                    return _start_connecting(ctx, cluster, provider, ni, head_info, pending_tasks)
                case _PollResult():
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > poll_timeout:
                        log.error("Instance not ready within {t}s", t=poll_timeout)
                        pool.tell(
                            NodeLost(
                                node_id=node_id, reason=f"Instance not ready within {poll_timeout}s"
                            )
                        )
                        return _start_replacing(
                            ctx, cluster, provider, instance_id, pending_tasks, head_info
                        )

                    async def _poll_after_delay() -> _PollResult:
                        await asyncio.sleep(poll_interval)
                        _, inst = await provider.get_instance(cluster, instance_id)
                        return _PollResult(instance=inst)

                    ctx.pipe_to_self(
                        _poll_after_delay(),
                        on_failure=lambda e: _PollResult(instance=None),
                    )
                    return polling(
                        cluster, provider, instance_id, start_time, head_info, pending_tasks
                    )
                case Preempted():
                    log.warning("Preempted during polling")
                    pool.tell(NodeLost(node_id=node_id, reason="preempted"))
                    return _start_replacing(
                        ctx, cluster, provider, instance_id, pending_tasks, head_info
                    )
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(
                        ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout
                    )
                    return polling(
                        cluster, provider, instance_id, start_time, head_info, (*pending_tasks, pt)
                    )
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── connecting ────────────────────────────────────────────────────

    def _start_connecting(
        ctx: ActorContext[NodeMsg],
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        log.info("Opening SSH tunnel to {ip}", ip=ni.instance.ip)
        ctx.pipe_to_self(
            _open_tunnel(ni, cluster, ssh_timeout, ssh_retry_interval),
            mapper=lambda result: _Connected(transport=result[0], listener=result[1]),
            on_failure=lambda e: _ConnectionFailed(error=str(e)),
        )
        return connecting(cluster, provider, ni, head_info, pending_tasks)

    def connecting(
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case _Connected(transport=transport, listener=listener):
                    log.info("SSH tunnel established")
                    return _start_bootstrapping(
                        ctx,
                        cluster,
                        provider,
                        ni,
                        transport,
                        listener,
                        head_info,
                        pending_tasks,
                    )
                case _ConnectionFailed(error=error):
                    log.error("SSH connection failed: {error}", error=error)
                    pool.tell(NodeLost(node_id=node_id, reason=error))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case HeadAddressKnown() as h:
                    return connecting(cluster, provider, ni, h, pending_tasks)
                case Preempted():
                    pool.tell(NodeLost(node_id=node_id, reason="preempted"))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(
                        ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout
                    )
                    return connecting(cluster, provider, ni, head_info, (*pending_tasks, pt))
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── bootstrapping ─────────────────────────────────────────────────

    def _start_bootstrapping(
        ctx: ActorContext[NodeMsg],
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        spec = cluster.spec
        log.info("Starting bootstrap")

        if not _skip_monitor:
            ctx.spawn(
                instance_monitor(
                    info=ni,
                    transport=transport,
                    event_listener=pool,
                    reply_to=ctx.self,
                ),
                f"monitor-{ni.instance.id}",
            )

        if cluster.prebaked:
            log.info("Prebaked image detected, skipping bootstrap")
            return _enter_post_bootstrap(
                ctx,
                cluster,
                provider,
                ni,
                transport,
                listener,
                head_info,
                pending_tasks,
            )

        ctx.pipe_to_self(
            _run_bootstrap(transport, ni, cluster, spec),
            mapper=lambda _: _BootstrapUploaded(),
            on_failure=lambda e: _BootstrapUploadFailed(error=str(e)),
        )
        return bootstrapping(cluster, provider, ni, transport, listener, head_info, pending_tasks)

    def bootstrapping(
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return bootstrapping(
                        cluster, provider, ni, transport, listener, h, pending_tasks
                    )
                case BootstrapDone(success=True, instance=done_info):
                    log.info("Bootstrap completed successfully")
                    final_ni = done_info or ni
                    match provider:
                        case WarmableProvider() if node_id == 0:
                            log.info("Snapshotting provider image...")
                            ctx.pipe_to_self(
                                provider.save(cluster),
                                mapper=lambda _: _SnapshotSaved(),
                                on_failure=lambda e: _SnapshotFailed(error=str(e)),
                            )
                    return _enter_post_bootstrap(
                        ctx,
                        cluster,
                        provider,
                        final_ni,
                        transport,
                        listener,
                        head_info,
                        pending_tasks,
                    )
                case BootstrapDone(success=False, error=error):
                    log.error("Bootstrap failed: {error}", error=error)
                    reason = error or "bootstrap failed"
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case _BootstrapUploaded():
                    log.info("Bootstrap script uploaded and started")
                    return Behaviors.same()
                case _BootstrapUploadFailed(error=error):
                    log.error("Bootstrap upload failed: {error}", error=error)
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason=error))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case Preempted():
                    log.warning("Preempted during bootstrap")
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason="preempted"))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(
                        ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout
                    )
                    return bootstrapping(
                        cluster,
                        provider,
                        ni,
                        transport,
                        listener,
                        head_info,
                        (*pending_tasks, pt),
                    )
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── post_bootstrap ────────────────────────────────────────────────

    def _enter_post_bootstrap(
        ctx: ActorContext[NodeMsg],
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        spec = cluster.spec
        if spec.image and getattr(spec.image, "skyward_source", None) == "local":
            ctx.pipe_to_self(
                _install_local_skyward(transport, ni, cluster),
                mapper=lambda _, m=ni: _LocalInstallDone(instance=m),
                on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
            )
            return post_bootstrap(
                cluster, provider, ni, transport, listener, head_info, pending_tasks
            )

        if spec.image and getattr(spec.image, "includes", None):
            ctx.pipe_to_self(
                _sync_user_code(transport, ni, spec, cluster),
                mapper=lambda _, m=ni: _UserCodeSyncDone(instance=m),
                on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
            )
            return post_bootstrap(
                cluster, provider, ni, transport, listener, head_info, pending_tasks
            )

        return _enter_ready(
            ctx, cluster, provider, ni, transport, listener, head_info, pending_tasks
        )

    def post_bootstrap(
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return post_bootstrap(
                        cluster, provider, ni, transport, listener, h, pending_tasks
                    )
                case _LocalInstallDone(instance=info):
                    s = cluster.spec
                    if s.image and getattr(s.image, "includes", None):
                        ctx.pipe_to_self(
                            _sync_user_code(transport, info, s, cluster),
                            mapper=lambda _, m=info: _UserCodeSyncDone(instance=m),
                            on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
                        )
                        return Behaviors.same()
                    return _enter_ready(
                        ctx, cluster, provider, ni, transport, listener, head_info, pending_tasks
                    )
                case _UserCodeSyncDone():
                    return _enter_ready(
                        ctx, cluster, provider, ni, transport, listener, head_info, pending_tasks
                    )
                case _PostBootstrapFailed(error=err):
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason=err))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
                case Preempted():
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason="preempted"))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(
                        ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout
                    )
                    return post_bootstrap(
                        cluster,
                        provider,
                        ni,
                        transport,
                        listener,
                        head_info,
                        (*pending_tasks, pt),
                    )
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── ready (worker started, waiting for JoinCluster) ───────────────

    def _enter_ready(
        ctx: ActorContext[NodeMsg],
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        spec = cluster.spec
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
            return _start_worker_process(
                ctx, cluster, provider, ni, transport, listener, head_info, pending_tasks
            )

        if head_info is not None:
            log.info("Starting worker (role=worker)")
            return _start_worker_process(
                ctx, cluster, provider, ni, transport, listener, head_info, pending_tasks
            )

        log.info("Waiting for head address before starting worker")
        return waiting_for_head(cluster, provider, ni, transport, listener, pending_tasks)

    def waiting_for_head(
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    log.info("Head address received, starting worker")
                    return _start_worker_process(
                        ctx, cluster, provider, ni, transport, listener, h, pending_tasks
                    )
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
                case Preempted():
                    log.warning("Preempted while waiting for head")
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason="preempted"))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info=None
                    )
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(
                        ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout
                    )
                    return waiting_for_head(
                        cluster, provider, ni, transport, listener, (*pending_tasks, pt)
                    )
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── starting worker ───────────────────────────────────────────────

    def _start_worker_process(
        ctx: ActorContext[NodeMsg],
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        ctx.pipe_to_self(
            _do_start_worker(transport, listener, ni, head_info, node_id, cluster, cluster.spec),
            mapper=lambda result: _WorkerStarted(local_port=result[0], private_ip=result[1]),
            on_failure=lambda e: _WorkerFailed(error=str(e)),
        )
        return starting_worker(cluster, provider, ni, transport, listener, head_info, pending_tasks)

    def starting_worker(
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
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
                    return ready(
                        cluster, provider, ni, transport, listener, head_info, pending_tasks
                    )
                case _WorkerFailed(error=error):
                    log.error("Worker failed to start: {error}", error=error)
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason=error))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
                case Preempted():
                    log.warning("Preempted while starting worker")
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason="preempted"))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(
                        ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout
                    )
                    return starting_worker(
                        cluster,
                        provider,
                        ni,
                        transport,
                        listener,
                        head_info,
                        (*pending_tasks, pt),
                    )
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── ready (worker running, waiting for JoinCluster from pool) ─────

    def ready(
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...] = (),
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case JoinCluster(client=client, pool_info_json=pij, env_vars=ev):
                    log.info("JoinCluster received, discovering worker")
                    ctx.pipe_to_self(
                        _discover_own_worker(client, ni),
                        mapper=lambda ref: _WorkerDiscovered(worker_ref=ref),
                        on_failure=lambda e: _WorkerDiscoveryFailed(error=str(e)),
                    )
                    return joining(
                        cluster,
                        provider,
                        ni,
                        transport,
                        listener,
                        head_info,
                        pending_tasks,
                        client=client,
                        pool_info_json=pij,
                        env_vars=ev,
                    )
                case HeadAddressKnown() as h:
                    return ready(cluster, provider, ni, transport, listener, h, pending_tasks)
                case Preempted():
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason="preempted"))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(
                        ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout
                    )
                    return ready(
                        cluster, provider, ni, transport, listener, head_info, (*pending_tasks, pt)
                    )
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── joining (discovering worker + setting up env) ─────────────────

    def joining(
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...],
        client: Any,
        pool_info_json: str,
        env_vars: dict[str, str],
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case _WorkerDiscovered(worker_ref=worker_ref):
                    log.info("Worker discovered, setting up env")
                    ctx.pipe_to_self(
                        _setup_worker_env(client, worker_ref, pool_info_json, env_vars),
                        mapper=lambda _: _EnvSetupDone(),
                        on_failure=lambda e: _EnvSetupFailed(error=str(e)),
                    )
                    return joining_env_setup(
                        cluster,
                        provider,
                        ni,
                        transport,
                        listener,
                        head_info,
                        pending_tasks,
                        client=client,
                        worker_ref=worker_ref,
                    )
                case _WorkerDiscoveryFailed(error=error):
                    log.error("Worker discovery failed: {error}", error=error)
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason=f"Worker discovery failed: {error}"))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case Preempted():
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason="preempted"))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(
                        ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout
                    )
                    return joining(
                        cluster,
                        provider,
                        ni,
                        transport,
                        listener,
                        head_info,
                        (*pending_tasks, pt),
                        client=client,
                        pool_info_json=pool_info_json,
                        env_vars=env_vars,
                    )
            return Behaviors.same()

        return Behaviors.receive(receive)

    def joining_env_setup(
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...],
        client: Any,
        worker_ref: Any,
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case _EnvSetupDone():
                    log.info("Worker env configured, transitioning to active")
                    return _enter_active(
                        ctx,
                        cluster,
                        provider,
                        ni,
                        transport,
                        listener,
                        head_info,
                        pending_tasks,
                        client,
                        worker_ref,
                    )
                case _EnvSetupFailed(error=error):
                    log.error("Worker env setup failed: {error}", error=error)
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason=f"Env setup failed: {error}"))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case Preempted():
                    await _cleanup_transport(transport, listener)
                    pool.tell(NodeLost(node_id=node_id, reason="preempted"))
                    return _start_replacing(
                        ctx, cluster, provider, ni.instance.id, pending_tasks, head_info
                    )
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(
                        ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout
                    )
                    return joining_env_setup(
                        cluster,
                        provider,
                        ni,
                        transport,
                        listener,
                        head_info,
                        (*pending_tasks, pt),
                        client=client,
                        worker_ref=worker_ref,
                    )
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── active ────────────────────────────────────────────────────────

    def _enter_active(
        ctx: ActorContext[NodeMsg],
        cluster: Any,
        provider: Any,
        ni: NodeInstance,
        transport: Any,
        listener: Any,
        head_info: HeadAddressKnown | None,
        pending_tasks: tuple[_PendingTask, ...],
        client: Any,
        worker_ref: Any,
    ) -> Behavior[NodeMsg]:
        s = ActiveState(
            cluster=cluster,
            provider=provider,
            client=client,
            worker_ref=worker_ref,
            current_node_instance=ni,
            head_info=head_info,
            transport=transport,
            listener=listener,
        )
        new_inflight: dict[str, ActorRef] = {}
        counter = 0
        for pt in pending_tasks:
            tid = pt.task_id or str(counter)
            _dispatch_task(ctx, s, tid, pt.fn, pt.args, pt.kwargs, pt.timeout)
            new_inflight[tid] = pt.reply_to
            counter += 1

        return active(
            replace(
                s,
                inflight=new_inflight,
                task_counter=counter,
                pending_tasks=(),
            )
        )

    def _dispatch_task(
        ctx: ActorContext[NodeMsg],
        s: ActiveState,
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
            ),
        )

    def active(s: ActiveState) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return active(replace(s, head_info=h))
                case JoinCluster(client=client, pool_info_json=pij, env_vars=ev):
                    log.info("Re-joining cluster after replacement")
                    ni = s.current_node_instance
                    if ni is None:
                        log.error("Cannot re-join: no node instance")
                        return Behaviors.same()
                    ctx.pipe_to_self(
                        _discover_own_worker(client, ni),
                        mapper=lambda ref: _WorkerDiscovered(worker_ref=ref),
                        on_failure=lambda e: _WorkerDiscoveryFailed(error=str(e)),
                    )
                    return joining(
                        s.cluster,
                        s.provider,
                        ni,
                        s.transport,
                        s.listener,
                        s.head_info,
                        s.pending_tasks,
                        client=client,
                        pool_info_json=pij,
                        env_vars=ev,
                    )
                case ExecuteOnNode() as ex:
                    local_tid = ex.task_id or str(s.task_counter)
                    log.debug("Node {nid} dispatching task {tid}", nid=node_id, tid=local_tid)
                    _dispatch_task(ctx, s, local_tid, ex.fn, ex.args, ex.kwargs, ex.timeout)
                    new_inflight = {**s.inflight, local_tid: ex.reply_to}
                    return active(
                        replace(
                            s,
                            inflight=new_inflight,
                            task_counter=s.task_counter + 1,
                        )
                    )
                case _RemoteTaskDone(task_id=tid, value=value, error=is_err):
                    log.debug("Node {nid} received task result (tid={tid})", nid=node_id, tid=tid)
                    caller = s.inflight.get(tid)
                    if caller:
                        caller.tell(
                            TaskResult(
                                value=value,
                                node_id=node_id,
                                task_id=tid,
                                error=is_err,
                            )
                        )
                        new_inflight = {k: v for k, v in s.inflight.items() if k != tid}
                        return active(replace(s, inflight=new_inflight))
                    return Behaviors.same()
                case Preempted(reason=reason):
                    log.warning("Preempted while active: {reason}", reason=reason)
                    await _cleanup_transport(s.transport, s.listener)
                    pool.tell(NodeLost(node_id=node_id, reason=reason))
                    return _start_replacing(
                        ctx,
                        s.cluster,
                        s.provider,
                        s.current_node_instance.instance.id if s.current_node_instance else "",
                        s.pending_tasks,
                        s.head_info,
                    )
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── replacing ─────────────────────────────────────────────────────

    def _start_replacing(
        ctx: ActorContext[NodeMsg],
        cluster: Any,
        provider: Any,
        dead_id: str,
        pending_tasks: tuple[_PendingTask, ...],
        head_info: HeadAddressKnown | None,
    ) -> Behavior[NodeMsg]:
        ctx.pipe_to_self(
            _terminate_and_replace(provider, cluster, dead_id),
            mapper=lambda inst: Provision(
                cluster=cluster,
                provider=provider,
                instance=inst,
            ),
        )
        return replacing(cluster, provider, pending_tasks, head_info)

    def replacing(
        cluster: Any,
        provider: Any,
        pending_tasks: tuple[_PendingTask, ...] = (),
        head_info: HeadAddressKnown | None = None,
    ) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return replacing(cluster, provider, pending_tasks, head_info=h)
                case Provision(instance=instance):
                    log.info("Replacement provisioned, re-entering polling")
                    return _start_polling(ctx, cluster, provider, instance)
                case ExecuteOnNode() as ex:
                    pt = _PendingTask(
                        ex.fn, ex.args, ex.kwargs, ex.reply_to, ex.task_id, ex.timeout
                    )
                    return replacing(cluster, provider, (*pending_tasks, pt), head_info=head_info)
            return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()


# ─── Helpers (module-level, shared with old instance.py) ──────────────


async def _open_tunnel(
    ni: NodeInstance,
    cluster: Any,
    ssh_timeout: float = 300.0,
    ssh_retry_interval: float = 5.0,
) -> tuple[Any, Any]:
    from skyward.infra.ssh import SSHTransport

    transport = SSHTransport(
        host=ni.instance.ip or "",
        user=ni.ssh_user or cluster.ssh_user,
        key_path=ni.ssh_key_path or cluster.ssh_key_path,
        port=ni.instance.ssh_port,
        retry_max_attempts=int(ssh_timeout / ssh_retry_interval) + 1,
        retry_delay=ssh_retry_interval,
    )
    await transport.connect()
    listener = await transport._conn.forward_local_port(  # type: ignore[union-attr]
        "",
        0,
        "127.0.0.1",
        25520,
    )
    return transport, listener


async def _do_start_worker(
    transport: Any,
    listener: Any,
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

    exit_code, stdout, stderr = await transport.run(casty_cmd, timeout=60.0)
    if exit_code != 0:
        raise RuntimeError(f"Failed to start Casty node {node_id}: {stderr}")
    await transport.run(tail_cmd, timeout=10.0)
    log.debug("Casty worker started, PID: {pid}", pid=stdout.strip())

    if spec.image:
        _check_python_version(spec.image.python)

    local_port = listener.get_port()
    return local_port, private_ip


async def _cleanup_transport(transport: Any, listener: Any) -> None:
    with suppress(Exception):
        listener.close()
    with suppress(Exception):
        await transport.close()


async def _install_local_skyward(transport: Any, ni: NodeInstance, cluster: Any) -> None:
    from skyward.providers._bootstrap_ssh import install_local_skyward

    await install_local_skyward(transport=transport, ni=ni, use_sudo=cluster.use_sudo)


async def _sync_user_code(transport: Any, ni: NodeInstance, spec: Any, cluster: Any) -> None:
    from skyward.providers._bootstrap_ssh import upload_user_code
    from skyward.providers.common import build_user_code_tarball

    image = spec.image
    includes = getattr(image, "includes", ())
    if not includes:
        return

    excludes = getattr(image, "excludes", ())
    tarball = build_user_code_tarball(includes=includes, excludes=excludes)

    ssh_user = ni.ssh_user or cluster.ssh_user
    use_sudo = ssh_user != "root"
    await upload_user_code(transport=transport, tarball=tarball, use_sudo=use_sudo)


async def _run_bootstrap(
    transport: Any,
    ni: NodeInstance,
    cluster: Any,
    spec: Any,
) -> None:
    from skyward.providers._bootstrap_ssh import run_bootstrap_via_ssh

    image = spec.image
    if not image:
        return

    ssh_user = ni.ssh_user or cluster.ssh_user
    use_sudo = ssh_user != "root"

    postamble = None
    if spec.volumes and cluster.mount_endpoint:
        from skyward.providers.bootstrap import mount_volumes, phase

        postamble = phase("volumes", mount_volumes(spec.volumes, cluster.mount_endpoint))

    bootstrap_script = image.generate_bootstrap(ttl=0, postamble=postamble)

    await run_bootstrap_via_ssh(
        transport=transport,
        ni=ni,
        bootstrap_script=bootstrap_script,
        use_sudo=use_sudo,
    )


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
) -> None:
    from skyward.infra.worker import ExecuteTask

    def setup_env(info_json: str, extra: dict[str, str]) -> str:
        import os

        os.environ["COMPUTE_POOL"] = info_json
        for k, v in extra.items():
            os.environ[k] = v
        return "ok"

    await client.ask(
        worker_ref,
        lambda rto: ExecuteTask(
            fn=setup_env,
            args=(pool_info_json, env_vars),
            kwargs={},
            reply_to=rto,
        ),
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

    result = await client.ask(
        worker_ref,
        lambda rto: ExecuteTask(
            fn=fn,
            args=resolved_args,
            kwargs=kwargs,
            reply_to=rto,
            input_streams=stream_refs,
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

    consumer_ref = client.spawn(
        stream_consumer(producer_ref, timeout=60.0, initial_demand=16),
        f"out-consumer-{uuid4().hex[:8]}",
    )
    return await client.ask(consumer_ref, lambda r: GetSource(reply_to=r), timeout=10.0)
