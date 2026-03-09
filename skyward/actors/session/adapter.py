from __future__ import annotations

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.pool.messages import PoolMsg, PoolStarted, ProvisionFailed

from .messages import SessionMsg, _PoolFailed, _PoolReady

type _AdapterMsg = PoolStarted | ProvisionFailed


def start_adapter(
    name: str,
    session: ActorRef[SessionMsg],
    pool_ref: ActorRef[PoolMsg],
) -> Behavior[_AdapterMsg]:
    """One-shot adapter: converts pool reply into session-internal message.

    Parameters
    ----------
    name : str
        Logical pool name, forwarded in the converted message.
    session : ActorRef[SessionMsg]
        The session actor that should receive the converted message.
    pool_ref : ActorRef[PoolMsg]
        Reference to the pool actor, attached to ``_PoolReady``.

    Returns
    -------
    Behavior[_AdapterMsg]
        A one-shot behavior that stops after handling a single message.
    """

    async def receive(
        _ctx: ActorContext[_AdapterMsg],
        msg: _AdapterMsg,
    ) -> Behavior[_AdapterMsg]:
        match msg:
            case PoolStarted(cluster_id=cid, instances=instances, cluster=cluster):
                session.tell(_PoolReady(
                    name=name,
                    cluster_id=cid,
                    instances=instances,
                    cluster=cluster,
                    pool_ref=pool_ref,
                ))
            case ProvisionFailed(reason=reason):
                session.tell(_PoolFailed(name=name, reason=reason))
        return Behaviors.stopped()

    return Behaviors.receive(receive)
