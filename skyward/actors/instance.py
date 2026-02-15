from collections.abc import Callable, Coroutine
from typing import Any

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
)
from skyward.actors.streaming import instance_monitor

type TunnelFactory = Callable[[str], Coroutine[Any, Any, Any]]


def instance_actor(
    instance_id: str,
    provider_ref: ActorRef,
    cluster_client: object,
    parent: ActorRef,
    metadata: InstanceMetadata | None = None,
    _skip_tunnel: bool = False,
    _tunnel_factory: TunnelFactory | None = None,
    _skip_monitor: bool = False,
) -> Behavior[InstanceMsg]:

    def waiting() -> Behavior[InstanceMsg]:
        async def receive(ctx: ActorContext[InstanceMsg], msg: InstanceMsg) -> Behavior[InstanceMsg]:
            match msg:
                case Running(ip):
                    if _skip_tunnel:
                        return bootstrapping(ip, ctx)
                    return starting(ip)
                case Preempted():
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def starting(ip: str) -> Behavior[InstanceMsg]:
        async def setup(ctx: ActorContext[InstanceMsg]) -> Behavior[InstanceMsg]:
            if _tunnel_factory:
                await _tunnel_factory(ip)
            return bootstrapping(ip, ctx)
        return Behaviors.setup(setup)

    def bootstrapping(ip: str, ctx: ActorContext[InstanceMsg]) -> Behavior[InstanceMsg]:
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

        async def receive(ctx: ActorContext[InstanceMsg], msg: InstanceMsg) -> Behavior[InstanceMsg]:
            match msg:
                case BootstrapDone(success=True) as done:
                    provider_ref.tell(done)
                    return ready(ip)
                case BootstrapDone(success=False, error=error):
                    parent.tell(InstanceDied(instance_id=instance_id, reason=error or "bootstrap failed"))
                    return Behaviors.stopped()
                case Bootstrapping():
                    return Behaviors.same()
                case Bootstrapped():
                    parent.tell(InstanceBecameReady(instance_id=instance_id, ip=ip))
                    return ready(ip)
                case Preempted():
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def ready(ip: str) -> Behavior[InstanceMsg]:
        async def receive(ctx: ActorContext[InstanceMsg], msg: InstanceMsg) -> Behavior[InstanceMsg]:
            match msg:
                case Execute(fn_bytes, reply_to):
                    if cluster_client and hasattr(cluster_client, "lookup"):
                        worker_ref = cluster_client.lookup(instance_id)
                        if worker_ref:
                            worker_ref.tell(Execute(fn_bytes=fn_bytes, reply_to=reply_to))
                case Log() | Metric():
                    pass
                case Preempted():
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    return waiting()
