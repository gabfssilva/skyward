"""PoolActor - top-level supervisor for cluster lifecycle.

A pool tells this story: idle -> requesting -> provisioning -> ready -> shutting_down.

The PoolActor supervises NodeActors and coordinates the full cluster lifecycle
from provisioning through execution to shutdown. Observability is provided
transparently via Behaviors.spy() — the pool has no knowledge of observers.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import Provision
from skyward.actors.node import NodeMsg, node_actor
from skyward.actors.provider import ProviderMsg
from skyward.messages import (
    BootstrapRequested,
    ClusterId,
    ClusterProvisioned,
    ClusterRequested,
    InstanceBootstrapped,
    InstanceMetadata,
    InstancePreempted,
    InstanceProvisioned,
    InstanceRunning,
    NodeId,
    ProviderName,
    ShutdownRequested,
)
from skyward.spec import PoolSpec


# =============================================================================
# Reply Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class PoolStarted:
    cluster_id: ClusterId
    instances: tuple[InstanceMetadata, ...]


@dataclass(frozen=True, slots=True)
class PoolStopped:
    pass


@dataclass(frozen=True, slots=True)
class ExecuteResult:
    value: Any
    node_id: int


@dataclass(frozen=True, slots=True)
class BroadcastResult:
    values: tuple[Any, ...]


# =============================================================================
# Commands (external messages)
# =============================================================================


@dataclass(frozen=True, slots=True)
class StartPool:
    spec: PoolSpec
    provider_config: Any
    provider_ref: ActorRef[ProviderMsg]
    reply_to: ActorRef[PoolStarted]


@dataclass(frozen=True, slots=True)
class StopPool:
    reply_to: ActorRef[PoolStopped]


@dataclass(frozen=True, slots=True)
class ExecuteTask:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    node: int | None
    reply_to: ActorRef[ExecuteResult]


@dataclass(frozen=True, slots=True)
class BroadcastTask:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[BroadcastResult]


# =============================================================================
# Internal Messages (from node actors)
# =============================================================================


@dataclass(frozen=True, slots=True)
class NodeBecameReady:
    node_id: NodeId
    instance: InstanceMetadata


# =============================================================================
# Message Union
# =============================================================================

type PoolMsg = (
    StartPool
    | StopPool
    | ExecuteTask
    | BroadcastTask
    | ClusterProvisioned
    | InstanceRunning
    | InstanceProvisioned
    | InstanceBootstrapped
    | InstancePreempted
    | NodeBecameReady
)


def _to_metadata(ev: InstanceRunning) -> InstanceMetadata:
    return InstanceMetadata(
        id=ev.instance_id,
        node=ev.node_id,
        provider=ev.provider,
        ip=ev.ip,
        private_ip=ev.private_ip or "",
        network_interface=ev.network_interface,
        spot=ev.spot,
        ssh_port=ev.ssh_port,
        hourly_rate=ev.hourly_rate,
        on_demand_rate=ev.on_demand_rate,
        billing_increment=ev.billing_increment,
        instance_type=ev.instance_type,
        gpu_count=ev.gpu_count,
        gpu_model=ev.gpu_model,
        vcpus=ev.vcpus,
        memory_gb=ev.memory_gb,
        gpu_vram_gb=ev.gpu_vram_gb,
        region=ev.region,
    )


# =============================================================================
# Behavior
# =============================================================================


def pool_actor() -> Behavior[PoolMsg]:
    """A pool tells this story: idle -> requesting -> provisioning -> ready -> shutting_down."""

    def idle() -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case StartPool(spec=spec, provider_config=provider_config, provider_ref=prov_ref, reply_to=reply_to):
                    request_id = f"pool-{uuid.uuid4().hex[:8]}"
                    provider = spec.provider or "aws"

                    prov_ref.tell(
                        ClusterRequested(
                            request_id=request_id,
                            provider=provider,
                            spec=spec,
                        )
                    )

                    return requesting(
                        request_id=request_id,
                        spec=spec,
                        provider=provider,
                        provider_config=provider_config,
                        provider_ref=prov_ref,
                        start_reply=reply_to,
                    )
                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def requesting(
        request_id: str,
        spec: PoolSpec,
        provider: ProviderName,
        provider_config: Any,
        provider_ref: ActorRef[ProviderMsg],
        start_reply: ActorRef[PoolStarted],
    ) -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case ClusterProvisioned(
                    request_id=rid,
                    cluster_id=cluster_id,
                    provider=prov,
                ) if rid == request_id:
                    node_refs: dict[NodeId, ActorRef[NodeMsg]] = {}
                    for i in range(spec.nodes):
                        node_behavior = node_actor(
                            node_id=i,
                            cluster_id=cluster_id,
                            provider=prov,
                            provider_ref=provider_ref,
                            pool_ref=ctx.self,
                        )
                        ref = ctx.spawn(
                            node_behavior,
                            f"node-{i}",
                        )
                        node_refs[i] = ref
                        ref.tell(Provision(cluster_id=cluster_id, provider=prov))

                    return provisioning(
                        cluster_id=cluster_id,
                        spec=spec,
                        provider=prov,
                        provider_config=provider_config,
                        provider_ref=provider_ref,
                        node_refs=node_refs,
                        ready_instances={},
                        start_reply=start_reply,
                    )
                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def provisioning(
        cluster_id: ClusterId,
        spec: PoolSpec,
        provider: ProviderName,
        provider_config: Any,
        provider_ref: ActorRef[ProviderMsg],
        node_refs: dict[NodeId, ActorRef[NodeMsg]],
        ready_instances: dict[NodeId, InstanceMetadata],
        start_reply: ActorRef[PoolStarted],
    ) -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case InstanceRunning() as ev:
                    info = _to_metadata(ev)
                    provisioned_event = InstanceProvisioned(request_id=ev.request_id, instance=info)
                    ref = node_refs.get(info.node)
                    if ref:
                        ref.tell(provisioned_event)

                    provider_ref.tell(BootstrapRequested(
                        request_id=ev.request_id,
                        instance=info,
                        cluster_id=ev.cluster_id,
                    ))
                    return Behaviors.same()

                case InstanceBootstrapped(instance=info):
                    ref = node_refs.get(info.node)
                    if ref:
                        ref.tell(InstanceBootstrapped(instance=info))
                    return Behaviors.same()

                case InstancePreempted(instance=info, reason=reason):
                    ref = node_refs.get(info.node)
                    if ref:
                        ref.tell(InstancePreempted(instance=info, reason=reason))
                    return Behaviors.same()

                case NodeBecameReady(node_id=nid, instance=info):
                    new_ready = {**ready_instances, nid: info}

                    if len(new_ready) == spec.nodes:
                        instances = tuple(
                            new_ready[i] for i in sorted(new_ready)
                        )

                        start_reply.tell(PoolStarted(
                            cluster_id=cluster_id,
                            instances=instances,
                        ))

                        return ready(
                            cluster_id=cluster_id,
                            spec=spec,
                            provider=provider,
                            provider_config=provider_config,
                            provider_ref=provider_ref,
                            node_refs=node_refs,
                            instances=new_ready,
                        )

                    return provisioning(
                        cluster_id=cluster_id,
                        spec=spec,
                        provider=provider,
                        provider_config=provider_config,
                        provider_ref=provider_ref,
                        node_refs=node_refs,
                        ready_instances=new_ready,
                        start_reply=start_reply,
                    )

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def ready(
        cluster_id: ClusterId,
        spec: PoolSpec,
        provider: ProviderName,
        provider_config: Any,
        provider_ref: ActorRef[ProviderMsg],
        node_refs: dict[NodeId, ActorRef[NodeMsg]],
        instances: dict[NodeId, InstanceMetadata],
        next_node: int = 0,
    ) -> Behavior[PoolMsg]:
        async def receive(ctx: ActorContext[PoolMsg], msg: PoolMsg) -> Behavior[PoolMsg]:
            match msg:
                case ExecuteTask(
                    fn=fn,
                    args=args,
                    kwargs=kwargs,
                    node=target_node,
                    reply_to=reply_to,
                ):
                    sorted_nodes = sorted(instances.keys())
                    match target_node:
                        case int(n) if n in instances:
                            node_id = n
                            new_next = next_node
                        case None:
                            idx = next_node % len(sorted_nodes)
                            node_id = sorted_nodes[idx]
                            new_next = idx + 1
                        case _:
                            reply_to.tell(ExecuteResult(
                                value=ValueError(f"Node {target_node} not available"),
                                node_id=-1,
                            ))
                            return Behaviors.same()

                    try:
                        result = fn(*args, **kwargs)
                    except Exception as exc:
                        result = exc

                    reply_to.tell(ExecuteResult(value=result, node_id=node_id))

                    return ready(
                        cluster_id=cluster_id,
                        spec=spec,
                        provider=provider,
                        provider_config=provider_config,
                        provider_ref=provider_ref,
                        node_refs=node_refs,
                        instances=instances,
                        next_node=new_next,
                    )

                case BroadcastTask(
                    fn=fn,
                    args=args,
                    kwargs=kwargs,
                    reply_to=reply_to,
                ):
                    results: list[Any] = []
                    for nid in sorted(instances.keys()):
                        try:
                            results.append(fn(*args, **kwargs))
                        except Exception as exc:
                            results.append(exc)

                    reply_to.tell(BroadcastResult(values=tuple(results)))
                    return Behaviors.same()

                case StopPool(reply_to=reply_to):
                    provider_ref.tell(
                        ShutdownRequested(cluster_id=cluster_id)
                    )
                    reply_to.tell(PoolStopped())
                    return Behaviors.stopped()

                case InstancePreempted(instance=info, reason=reason):
                    ref = node_refs.get(info.node)
                    if ref:
                        ref.tell(InstancePreempted(instance=info, reason=reason))

                    new_instances = {
                        k: v for k, v in instances.items()
                        if k != info.node
                    }

                    return ready(
                        cluster_id=cluster_id,
                        spec=spec,
                        provider=provider,
                        provider_config=provider_config,
                        provider_ref=provider_ref,
                        node_refs=node_refs,
                        instances=new_instances,
                        next_node=next_node,
                    )

                case NodeBecameReady(node_id=nid, instance=info):
                    new_instances = {**instances, nid: info}
                    return ready(
                        cluster_id=cluster_id,
                        spec=spec,
                        provider=provider,
                        provider_config=provider_config,
                        provider_ref=provider_ref,
                        node_refs=node_refs,
                        instances=new_instances,
                        next_node=next_node,
                    )

                case InstanceRunning() as ev:
                    info = _to_metadata(ev)
                    provisioned_event = InstanceProvisioned(request_id=ev.request_id, instance=info)
                    ref = node_refs.get(info.node)
                    if ref:
                        ref.tell(provisioned_event)

                    provider_ref.tell(BootstrapRequested(
                        request_id=ev.request_id,
                        instance=info,
                        cluster_id=ev.cluster_id,
                    ))
                    return Behaviors.same()

                case InstanceBootstrapped(instance=info):
                    ref = node_refs.get(info.node)
                    if ref:
                        ref.tell(InstanceBootstrapped(instance=info))
                    return Behaviors.same()

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PoolStarted",
    "PoolStopped",
    "ExecuteResult",
    "BroadcastResult",
    "StartPool",
    "StopPool",
    "ExecuteTask",
    "BroadcastTask",
    "NodeBecameReady",
    "PoolMsg",
    "pool_actor",
]
