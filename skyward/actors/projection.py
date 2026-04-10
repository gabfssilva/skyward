"""ProjectionActor — bridges raw Casty SpyEvents into SessionProjection.

Receives SpyEvent messages (as a spy observer), extracts the pool name,
translates to domain events via ``spy_adapter.translate``, and feeds the
projection.  Ignores events without a pool name, events that translate
to None, and Terminated events.
"""

from __future__ import annotations

from casty import ActorContext, Behavior, Behaviors, SpyEvent, Terminated

from skyward.actors.spy_adapter import pool_name_from_path, translate
from skyward.api.projection import SessionProjection

__all__ = ["projection_actor"]

type ProjectionMsg = SpyEvent


def projection_actor(projection: SessionProjection) -> Behavior[ProjectionMsg]:
    """Actor that observes SpyEvents and feeds a SessionProjection.

    Parameters
    ----------
    projection
        The mutable projection instance that accumulates session state.

    Returns
    -------
    Behavior[ProjectionMsg]
        A behavior that processes spy events into domain events.
    """

    async def receive(
        ctx: ActorContext[ProjectionMsg], msg: ProjectionMsg,
    ) -> Behavior[ProjectionMsg]:
        match msg:
            case SpyEvent(event=Terminated()):
                return Behaviors.same()
            case SpyEvent(actor_path=path) as spy:
                pool_name = pool_name_from_path(path)
                if pool_name is None:
                    return Behaviors.same()
                result = translate(spy, pool_name)
                match result:
                    case tuple() as events:
                        for event in events:
                            projection.handle(event)
                    case None:
                        pass
                    case event:
                        projection.handle(event)
        return Behaviors.same()

    return Behaviors.receive(receive)
