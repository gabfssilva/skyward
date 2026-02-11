from __future__ import annotations

import uuid

from casty import ActorContext, ActorRef, Behavior, Behaviors
from loguru import logger

from .messages import (
    ClusterId,
    InstanceBootstrapped,
    InstanceMetadata,
    InstancePreempted,
    InstanceProvisioned,
    InstanceRequested,
    NodeBecameReady,
    NodeId,
    NodeMsg,
    ProviderMsg,
    ProviderName,
    Provision,
)


def _make_request_id(node_id: NodeId) -> str:
    return f"node-{node_id}-{uuid.uuid4().hex[:8]}"


def node_actor(
    node_id: NodeId,
    cluster_id: ClusterId,
    provider: ProviderName,
    provider_ref: ActorRef[ProviderMsg],
    pool_ref: ActorRef,
) -> Behavior[NodeMsg]:
    log = logger.bind(actor="node", node_id=node_id, cluster_id=cluster_id)

    def idle() -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case Provision():
                    request_id = _make_request_id(node_id)
                    log.debug("Requesting instance, request_id={request_id}", request_id=request_id)
                    provider_ref.tell(
                        InstanceRequested(
                            request_id=request_id,
                            provider=provider,
                            cluster_id=cluster_id,
                            node_id=node_id,
                            replacing=None,
                        )
                    )
                    return provisioning(request_id)
                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def provisioning(request_id: str) -> Behavior[NodeMsg]:
        log.debug("State -> provisioning, request_id={request_id}", request_id=request_id)

        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case InstanceProvisioned(instance=info) if info.node == node_id:
                    log.debug("Instance provisioned, instance_id={iid}", iid=info.id)
                    return bootstrapping(info)
                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def bootstrapping(info: InstanceMetadata) -> Behavior[NodeMsg]:
        log.debug("State -> bootstrapping, instance_id={iid}", iid=info.id)

        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case InstanceBootstrapped(instance=i) if i.node == node_id:
                    log.debug("Bootstrap complete, notifying pool")
                    pool_ref.tell(NodeBecameReady(node_id=node_id, instance=i))
                    return ready(i)
                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def ready(info: InstanceMetadata) -> Behavior[NodeMsg]:
        log.debug("State -> ready, instance_id={iid}", iid=info.id)

        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case InstancePreempted(instance=i) if i.node == node_id:
                    request_id = _make_request_id(node_id)
                    log.debug(
                        "Preempted, requesting replacement, replacing={old} request_id={rid}",
                        old=info.id, rid=request_id,
                    )
                    provider_ref.tell(
                        InstanceRequested(
                            request_id=request_id,
                            provider=provider,
                            cluster_id=cluster_id,
                            node_id=node_id,
                            replacing=info.id,
                        )
                    )
                    return provisioning(request_id)
                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()
