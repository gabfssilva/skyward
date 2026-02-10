from __future__ import annotations

import uuid

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import Provision
from skyward.actors.provider import ProviderMsg
from skyward.messages import (
    ClusterId,
    InstanceBootstrapped,
    InstanceMetadata,
    InstancePreempted,
    InstanceProvisioned,
    InstanceRequested,
    NodeId,
    ProviderName,
)

type NodeMsg = (
    Provision
    | InstanceProvisioned
    | InstanceBootstrapped
    | InstancePreempted
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

    def idle() -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case Provision():
                    request_id = _make_request_id(node_id)
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
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case InstanceProvisioned(instance=info) if info.node == node_id:
                    return bootstrapping(info)
                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def bootstrapping(info: InstanceMetadata) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case InstanceBootstrapped(instance=i) if i.node == node_id:
                    from skyward.actors.pool import NodeBecameReady
                    pool_ref.tell(NodeBecameReady(node_id=node_id, instance=i))
                    return ready(i)
                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def ready(info: InstanceMetadata) -> Behavior[NodeMsg]:
        async def receive(ctx: ActorContext[NodeMsg], msg: NodeMsg) -> Behavior[NodeMsg]:
            match msg:
                case InstancePreempted(instance=i) if i.node == node_id:
                    request_id = _make_request_id(node_id)
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
