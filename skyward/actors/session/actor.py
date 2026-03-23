from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.observability.logger import logger

from .adapter import start_adapter
from .messages import (
    GetSessionSnapshot,
    PoolInfo,
    PoolSpawned,
    PoolSpawnFailed,
    PoolStateChanged,
    SessionMsg,
    SessionSnapshot,
    SessionStopped,
    SpawnPool,
    StopSession,
    _PoolFailed,
    _PoolReady,
    _SnapshotReady,
)

log = logger.bind(actor="session")

type _PendingReplies = MappingProxyType[str, ActorRef[PoolSpawned | PoolSpawnFailed]]
type _Pools = MappingProxyType[str, PoolInfo]

_EMPTY_POOLS: _Pools = MappingProxyType({})
_EMPTY_PENDING: _PendingReplies = MappingProxyType({})


def session_actor() -> Behavior[SessionMsg]:
    """Session actor managing the lifecycle of multiple named compute pools.

    Parameters
    ----------
    None

    Returns
    -------
    Behavior[SessionMsg]
        A behavior that tracks spawned pools, relays lifecycle events,
        and responds to snapshot queries.
    """
    return active(pools=_EMPTY_POOLS, pending_replies=_EMPTY_PENDING)


def active(
    pools: _Pools,
    pending_replies: _PendingReplies,
) -> Behavior[SessionMsg]:

    async def receive(
        ctx: ActorContext[SessionMsg], msg: SessionMsg,
    ) -> Behavior[SessionMsg]:
        match msg:
            case SpawnPool(
                name=name, spec=spec, provider_config=pc,
                provider=provider, offers=offers,
                provision_timeout=_, reply_to=reply_to,
            ):
                from skyward.actors.pool.actor import pool_actor
                from skyward.actors.pool.messages import StartPool

                pool_ref = ctx.spawn(
                    pool_actor(session_ref=ctx.self, pool_name=name),
                    f"pool-{name}",
                )
                adapter_ref = ctx.spawn(
                    start_adapter(name=name, session=ctx.self, pool_ref=pool_ref),
                    f"start-{name}",
                )
                pool_ref.tell(StartPool(
                    spec=spec, provider_config=pc,
                    provider=provider, offers=offers,
                    reply_to=adapter_ref,
                ))
                log.info("Spawning pool {name}", name=name)
                info = PoolInfo(
                    name=name, ref=pool_ref, spec=spec,
                    phase="provisioning", nodes_ready=0,
                    nodes_total=spec.nodes.min,
                )
                return active(
                    pools=MappingProxyType({**pools, name: info}),
                    pending_replies=MappingProxyType({**pending_replies, name: reply_to}),
                )

            case _PoolReady(
                name=name, cluster_id=cid,
                instances=instances, cluster=cluster,
                pool_ref=pool_ref,
            ):
                if reply_to := pending_replies.get(name):
                    reply_to.tell(PoolSpawned(
                        name=name, pool_ref=pool_ref,
                        cluster_id=cid, instances=instances,
                        cluster=cluster,
                    ))
                    log.info("Pool {name} ready", name=name)
                if existing := pools.get(name):
                    updated = replace(existing, phase="ready", ref=pool_ref)
                    return active(
                        pools=MappingProxyType({**pools, name: updated}),
                        pending_replies=MappingProxyType(
                            {k: v for k, v in pending_replies.items() if k != name},
                        ),
                    )
                return active(
                    pools=pools,
                    pending_replies=MappingProxyType(
                        {k: v for k, v in pending_replies.items() if k != name},
                    ),
                )

            case _PoolFailed(name=name, reason=reason):
                if reply_to := pending_replies.get(name):
                    reply_to.tell(PoolSpawnFailed(name=name, reason=reason))
                    log.warning("Pool {name} failed: {reason}", name=name, reason=reason)
                return active(
                    pools=MappingProxyType(
                        {k: v for k, v in pools.items() if k != name},
                    ),
                    pending_replies=MappingProxyType(
                        {k: v for k, v in pending_replies.items() if k != name},
                    ),
                )

            case PoolStateChanged(
                name=name, phase=phase,
                nodes_ready=ready, nodes_total=total,
            ):
                if existing := pools.get(name):
                    updated = replace(
                        existing, phase=phase,
                        nodes_ready=ready, nodes_total=total,
                    )
                    log.debug(
                        "Pool {name} state: {phase} ({ready}/{total})",
                        name=name, phase=phase, ready=ready, total=total,
                    )
                    return active(
                        pools=MappingProxyType({**pools, name: updated}),
                        pending_replies=pending_replies,
                    )
                return Behaviors.same()

            case GetSessionSnapshot(reply_to=reply_to):
                pool_refs = tuple(info.ref for info in pools.values())

                async def _gather_snapshots() -> _SnapshotReady:
                    import asyncio

                    from skyward.actors.messages import GetPoolSnapshot

                    snapshots = await asyncio.gather(*(
                        ctx.system.ask(ref, lambda r: GetPoolSnapshot(reply_to=r), timeout=2.0)
                        for ref in pool_refs
                    ))
                    return _SnapshotReady(snapshots=tuple(snapshots), reply_to=reply_to)

                ctx.pipe_to_self(
                    _gather_snapshots(),
                    mapper=lambda r: r,
                    on_failure=lambda _: _SnapshotReady(snapshots=(), reply_to=reply_to),
                )
                return Behaviors.same()

            case _SnapshotReady(snapshots=snaps, reply_to=snap_reply):
                snap_reply.tell(SessionSnapshot(pools=snaps))
                return Behaviors.same()

            case StopSession(reply_to=reply_to):
                log.info("Session stopping")
                reply_to.tell(SessionStopped())
                return Behaviors.stopped()

    return Behaviors.receive(receive)
