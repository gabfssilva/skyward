from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    BootstrapDone,
    Bootstrapped,
    Bootstrapping,
    Execute,
    InstanceBecameReady,
    InstanceDied,
    InstanceMetadata,
    InstanceMsg,
    Log,
    Metric,
    Preempted,
    Running,
    SetWorkerRef,
    TaskResult,
)
from skyward.actors.streaming import instance_monitor
from skyward.infra.worker import (
    ExecuteTask as WorkerExecuteTask,
)
from skyward.infra.worker import (
    TaskFailed as WorkerTaskFailed,
)
from skyward.infra.worker import (
    TaskSucceeded as WorkerTaskSucceeded,
)

if TYPE_CHECKING:
    from casty import ClusterClient

type TunnelFactory = Callable[[str], Coroutine[Any, Any, Any]]


def instance_actor(
    instance_id: str,
    provider_ref: ActorRef,
    worker_ref: ActorRef | None,
    parent: ActorRef,
    client: ClusterClient | None = None,
    metadata: InstanceMetadata | None = None,
    _skip_tunnel: bool = False,
    _tunnel_factory: TunnelFactory | None = None,
    _skip_monitor: bool = False,
) -> Behavior[InstanceMsg]:
    """An instance tells this story: waiting → bootstrapping → ready."""

    def waiting(
        wref: ActorRef | None = worker_ref,
        cl: ClusterClient | None = client,
    ) -> Behavior[InstanceMsg]:
        async def receive(
            ctx: ActorContext[InstanceMsg], msg: InstanceMsg,
        ) -> Behavior[InstanceMsg]:
            match msg:
                case SetWorkerRef(worker_ref=new_wref, client=new_cl):
                    return waiting(new_wref, new_cl or cl)
                case Running(ip):
                    if _skip_tunnel:
                        return bootstrapping(ip, ctx, wref, cl)
                    return starting(ip, wref, cl)
                case Preempted():
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def starting(
        ip: str, wref: ActorRef | None, cl: ClusterClient | None,
    ) -> Behavior[InstanceMsg]:
        async def setup(ctx: ActorContext[InstanceMsg]) -> Behavior[InstanceMsg]:
            if _tunnel_factory:
                await _tunnel_factory(ip)
            return bootstrapping(ip, ctx, wref, cl)
        return Behaviors.setup(setup)

    def bootstrapping(
        ip: str, ctx: ActorContext[InstanceMsg],
        wref: ActorRef | None, cl: ClusterClient | None,
    ) -> Behavior[InstanceMsg]:
        if not _skip_monitor and metadata and metadata.ssh_user and metadata.ssh_key_path:
            ctx.spawn(
                instance_monitor(
                    info=metadata,
                    ssh_user=metadata.ssh_user,
                    ssh_key_path=metadata.ssh_key_path,
                    event_listener=parent,
                    reply_to=ctx.self,
                ),
                f"monitor-{instance_id}",
            )

        async def receive(
            ctx: ActorContext[InstanceMsg], msg: InstanceMsg,
        ) -> Behavior[InstanceMsg]:
            match msg:
                case SetWorkerRef(worker_ref=new_wref, client=new_cl):
                    return bootstrapping(ip, ctx, new_wref, new_cl or cl)
                case BootstrapDone(success=True) as done:
                    provider_ref.tell(done)
                    return ready(ip, wref, cl)
                case BootstrapDone(success=False, error=error):
                    reason = error or "bootstrap failed"
                    parent.tell(InstanceDied(instance_id=instance_id, reason=reason))
                    return Behaviors.stopped()
                case Bootstrapping():
                    return Behaviors.same()
                case Bootstrapped():
                    parent.tell(InstanceBecameReady(instance_id=instance_id, ip=ip))
                    return ready(ip, wref, cl)
                case Preempted():
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def ready(
        ip: str, wref: ActorRef | None, cl: ClusterClient | None,
    ) -> Behavior[InstanceMsg]:
        async def receive(
            ctx: ActorContext[InstanceMsg], msg: InstanceMsg,
        ) -> Behavior[InstanceMsg]:
            match msg:
                case SetWorkerRef(worker_ref=new_wref, client=new_cl):
                    return ready(ip, new_wref, new_cl or cl)
                case Execute(fn_bytes=fn_bytes) if wref is not None and cl is not None:
                    ctx.pipe_to_self(
                        cl.ask(
                            wref,
                            lambda rto: WorkerExecuteTask(fn_bytes=fn_bytes, reply_to=rto),  # type: ignore[arg-type]
                            timeout=600.0,
                        ),
                        on_failure=lambda e: WorkerTaskFailed(  # type: ignore[return-value]
                            error=str(e), traceback="", node_id=0,
                        ),
                    )
                    return Behaviors.same()
                case Execute():
                    return Behaviors.same()
                case WorkerTaskSucceeded(result=value, node_id=nid):
                    parent.tell(TaskResult(value=value, node_id=nid))
                    return Behaviors.same()
                case WorkerTaskFailed(error=error, node_id=nid):
                    parent.tell(TaskResult(value=RuntimeError(error), node_id=nid))
                    return Behaviors.same()
                case Log() | Metric():
                    pass
                case Preempted():
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    return waiting()
