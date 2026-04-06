from __future__ import annotations

import asyncio
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    DesiredCountChanged,
    DrainComplete,
    DrainNode,
    NodeId,
    NodeJoined,
    ReconcilerNodeLost,
    ReconciliationExhausted,
    SpawnNodes,
)

if TYPE_CHECKING:
    from skyward.actors.pool.messages import PoolMsg
from skyward.observability.logger import logger

from .messages import (
    ReconcilerMsg,
    _ProvisionError,
    _ProvisionResult,
    _ReconcileTick,
    _TerminateError,
    _TerminateResult,
)
from .state import _State

log = logger.bind(actor="reconciler")


def reconciler_actor(
    pool: ActorRef[PoolMsg],
    provider: Any,
    cluster: Any,
    min_nodes: int,
    max_nodes: int,
    initial_node_ids: frozenset[NodeId],
    initial_instance_map: MappingProxyType[NodeId, str] | dict[NodeId, str],
    next_node_id: int,
    tick_interval: float = 15.0,
    max_provision_retries: int = 10,
) -> Behavior[ReconcilerMsg]:

    def _effective(s: _State) -> int:
        return len(s.current) + len(s.pending)

    def _schedule_tick(ctx: ActorContext[ReconcilerMsg]) -> None:
        async def _tick() -> _ReconcileTick:
            await asyncio.sleep(tick_interval)
            return _ReconcileTick()

        ctx.pipe_to_self(
            _tick(),
            mapper=lambda r: r,
            on_failure=lambda _: _ReconcileTick(),
        )

    async def setup(ctx: ActorContext[ReconcilerMsg]) -> Behavior[ReconcilerMsg]:
        _schedule_tick(ctx)
        state = _State(
            desired=len(initial_node_ids),
            current=initial_node_ids,
            pending=frozenset(),
            draining=frozenset(),
            next_node_id=next_node_id,
            instance_map=MappingProxyType(dict(initial_instance_map)),
            cluster=cluster,
        )
        log.info(
            "Reconciler started: desired={d}, current={c}, next_id={nid}",
            d=state.desired, c=len(state.current), nid=state.next_node_id,
        )
        return watching(state)

    def _start_scaling_up(
        ctx: ActorContext[ReconcilerMsg], s: _State,
    ) -> Behavior[ReconcilerMsg]:
        needed = s.desired - _effective(s)
        if needed <= 0:
            return watching(s)

        log.info("Scaling up: provisioning {n} instances", n=needed)

        ctx.pipe_to_self(
            provider.provision(s.cluster, needed),
            mapper=lambda result: _ProvisionResult(
                instances=tuple(result[1]), cluster=result[0],
            ),
            on_failure=lambda err: _ProvisionError(error=str(err)),
        )
        return scaling_up(s)

    def _start_draining(
        ctx: ActorContext[ReconcilerMsg], s: _State,
    ) -> Behavior[ReconcilerMsg]:
        excess = len(s.current) - s.desired
        if excess <= 0:
            return watching(s)

        victims = sorted(s.current, reverse=True)[:excess]
        victims = [nid for nid in victims if nid != 0]
        if not victims:
            return watching(s)

        log.info("Draining {n} nodes: {nids}", n=len(victims), nids=victims)
        to_drain = frozenset(victims)
        for nid in victims:
            pool.tell(DrainNode(node_id=nid, reply_to=ctx.self))
        return draining(replace(s, draining=to_drain))

    def watching(s: _State) -> Behavior[ReconcilerMsg]:

        async def receive(
            ctx: ActorContext[ReconcilerMsg], msg: ReconcilerMsg,
        ) -> Behavior[ReconcilerMsg]:
            match msg:
                case DesiredCountChanged(desired=desired, reason=reason):
                    log.info(
                        "Desired count changed: {old} → {new} ({reason})",
                        old=s.desired, new=desired, reason=reason,
                    )
                    new_s = replace(s, desired=desired)
                    effective = _effective(new_s)
                    if desired > effective:
                        return _start_scaling_up(ctx, new_s)
                    if desired < len(new_s.current):
                        return _start_draining(ctx, new_s)
                    return watching(new_s)

                case ReconcilerNodeLost(node_id=nid, reason=reason):
                    log.warning("Node {nid} lost: {reason}", nid=nid, reason=reason)
                    new_current = s.current - {nid}
                    new_pending = s.pending - {nid}

                    dead_iid = s.instance_map.get(nid)
                    if dead_iid:
                        ctx.pipe_to_self(
                            provider.terminate(s.cluster, (dead_iid,)),
                            mapper=lambda _: _TerminateResult(node_ids=(nid,)),
                            on_failure=lambda err: _TerminateError(
                                node_ids=(nid,), error=str(err),
                            ),
                        )

                    new_s = replace(s, current=new_current, pending=new_pending)
                    if new_s.desired > _effective(new_s):
                        return _start_scaling_up(ctx, new_s)
                    return watching(new_s)

                case _TerminateResult(node_ids=nids):
                    log.debug("Terminated dead instances for nodes {nids}", nids=nids)
                    return Behaviors.same()

                case _TerminateError(node_ids=nids, error=error):
                    log.error(
                        "Failed to terminate dead instances for nodes {nids}: {err}",
                        nids=nids, err=error,
                    )
                    return Behaviors.same()

                case NodeJoined(node_id=nid):
                    log.info("Node {nid} joined", nid=nid)
                    new_s = replace(
                        s,
                        current=s.current | {nid},
                        pending=s.pending - {nid},
                        consecutive_failures=0,
                    )
                    return watching(new_s)

                case _ReconcileTick():
                    _schedule_tick(ctx)
                    if s.consecutive_failures >= max_provision_retries:
                        return Behaviors.same()
                    effective = _effective(s)
                    if s.desired > effective:
                        log.debug(
                            "Tick: drift detected, desired={d} effective={e}",
                            d=s.desired, e=effective,
                        )
                        return _start_scaling_up(ctx, s)
                    return Behaviors.same()

            return Behaviors.same()
        return Behaviors.receive(receive)

    def scaling_up(s: _State) -> Behavior[ReconcilerMsg]:

        async def receive(
            ctx: ActorContext[ReconcilerMsg], msg: ReconcilerMsg,
        ) -> Behavior[ReconcilerMsg]:
            match msg:
                case _ProvisionResult(instances=instances, cluster=upd_cluster):
                    if not instances:
                        new_failures = s.consecutive_failures + 1
                        if new_failures >= max_provision_retries:
                            log.error(
                                "Provision returned 0 instances ({n}/{max} attempts exhausted)",
                                n=new_failures, max=max_provision_retries,
                            )
                            pool.tell(ReconciliationExhausted(
                                reason=f"failed to provision after {new_failures} consecutive attempts",
                            ))
                        else:
                            log.warning(
                                "Provision returned 0 instances (attempt {n}/{max})",
                                n=new_failures, max=max_provision_retries,
                            )
                        return watching(replace(s, consecutive_failures=new_failures))

                    start_id = s.next_node_id
                    new_pending_ids: list[NodeId] = []
                    new_map: MappingProxyType[NodeId, str] = s.instance_map
                    for i, inst in enumerate(instances):
                        nid = start_id + i
                        new_pending_ids.append(nid)
                        new_map = MappingProxyType({**new_map, nid: inst.id})

                    log.info(
                        "Provisioned {n} instances (ids {start}..{end})",
                        n=len(instances), start=start_id,
                        end=start_id + len(instances) - 1,
                    )
                    pool.tell(SpawnNodes(
                        instances=tuple(instances),
                        cluster=upd_cluster,
                        start_node_id=start_id,
                    ))
                    new_s = replace(
                        s,
                        pending=s.pending | frozenset(new_pending_ids),
                        next_node_id=start_id + len(instances),
                        instance_map=new_map,
                        cluster=upd_cluster,
                        consecutive_failures=0,
                    )
                    if new_s.desired <= _effective(new_s):
                        return watching(new_s)
                    return _start_scaling_up(ctx, new_s)

                case _ProvisionError(error=error):
                    new_failures = s.consecutive_failures + 1
                    if new_failures >= max_provision_retries:
                        log.error(
                            "Provision failed ({n}/{max} attempts exhausted): {err}",
                            n=new_failures, max=max_provision_retries, err=error,
                        )
                        pool.tell(ReconciliationExhausted(
                            reason=f"provision error after {new_failures} consecutive attempts: {error}",
                        ))
                    else:
                        log.error(
                            "Provision failed (attempt {n}/{max}): {err}",
                            n=new_failures, max=max_provision_retries, err=error,
                        )
                    return watching(replace(s, consecutive_failures=new_failures))

                case DesiredCountChanged(desired=desired, reason=reason):
                    log.info(
                        "Desired changed during scale-up: {old} → {new} ({reason})",
                        old=s.desired, new=desired, reason=reason,
                    )
                    new_s = replace(s, desired=desired)
                    if desired <= _effective(new_s):
                        return watching(new_s)
                    return scaling_up(new_s)

                case ReconcilerNodeLost(node_id=nid, reason=reason):
                    log.warning("Node {nid} lost during scale-up: {reason}", nid=nid, reason=reason)
                    dead_iid = s.instance_map.get(nid)
                    if dead_iid:
                        ctx.pipe_to_self(
                            provider.terminate(s.cluster, (dead_iid,)),
                            mapper=lambda _: _TerminateResult(node_ids=(nid,)),
                            on_failure=lambda err: _TerminateError(
                                node_ids=(nid,), error=str(err),
                            ),
                        )
                    return scaling_up(replace(
                        s,
                        current=s.current - {nid},
                        pending=s.pending - {nid},
                    ))

                case _TerminateResult(node_ids=nids):
                    log.debug("Terminated dead instances for nodes {nids}", nids=nids)
                    return Behaviors.same()

                case _TerminateError(node_ids=nids, error=error):
                    log.error(
                        "Failed to terminate dead instances for nodes {nids}: {err}",
                        nids=nids, err=error,
                    )
                    return Behaviors.same()

                case NodeJoined(node_id=nid):
                    log.info("Node {nid} joined during scale-up", nid=nid)
                    new_s = replace(
                        s,
                        current=s.current | {nid},
                        pending=s.pending - {nid},
                        consecutive_failures=0,
                    )
                    if new_s.desired <= _effective(new_s):
                        return watching(new_s)
                    return scaling_up(new_s)

                case _ReconcileTick():
                    _schedule_tick(ctx)
                    return Behaviors.same()

            return Behaviors.same()
        return Behaviors.receive(receive)

    def draining(s: _State) -> Behavior[ReconcilerMsg]:

        async def receive(
            ctx: ActorContext[ReconcilerMsg], msg: ReconcilerMsg,
        ) -> Behavior[ReconcilerMsg]:
            match msg:
                case DrainComplete(node_id=nid, instance_id=iid):
                    log.info("Node {nid} drained, terminating instance {iid}", nid=nid, iid=iid)
                    if iid:
                        ctx.pipe_to_self(
                            provider.terminate(s.cluster, (iid,)),
                            mapper=lambda _: _TerminateResult(node_ids=(nid,)),
                            on_failure=lambda err: _TerminateError(
                                node_ids=(nid,), error=str(err),
                            ),
                        )
                    new_s = replace(
                        s,
                        current=s.current - {nid},
                        draining=s.draining - {nid},
                    )
                    if not new_s.draining:
                        return watching(new_s)
                    return draining(new_s)

                case _TerminateResult(node_ids=nids):
                    log.debug("Terminated instances for nodes {nids}", nids=nids)
                    return Behaviors.same()

                case _TerminateError(node_ids=nids, error=error):
                    log.error(
                        "Failed to terminate instances for nodes {nids}: {err}",
                        nids=nids, err=error,
                    )
                    return Behaviors.same()

                case DesiredCountChanged(desired=desired, reason=reason):
                    log.info(
                        "Desired changed during drain: {old} → {new} ({reason})",
                        old=s.desired, new=desired, reason=reason,
                    )
                    new_s = replace(s, desired=desired)
                    if desired >= len(s.current):
                        log.info("Aborting drain — desired >= current")
                        new_s = replace(new_s, draining=frozenset())
                        if desired > _effective(new_s):
                            return _start_scaling_up(ctx, new_s)
                        return watching(new_s)
                    return draining(new_s)

                case ReconcilerNodeLost(node_id=nid, reason=reason):
                    log.warning("Node {nid} lost during drain: {reason}", nid=nid, reason=reason)
                    dead_iid = s.instance_map.get(nid)
                    if dead_iid:
                        ctx.pipe_to_self(
                            provider.terminate(s.cluster, (dead_iid,)),
                            mapper=lambda _: _TerminateResult(node_ids=(nid,)),
                            on_failure=lambda err: _TerminateError(
                                node_ids=(nid,), error=str(err),
                            ),
                        )
                    new_s = replace(
                        s,
                        current=s.current - {nid},
                        pending=s.pending - {nid},
                        draining=s.draining - {nid},
                    )
                    if not new_s.draining:
                        return watching(new_s)
                    return draining(new_s)

                case NodeJoined(node_id=nid):
                    return draining(replace(
                        s,
                        current=s.current | {nid},
                        pending=s.pending - {nid},
                    ))

                case _ReconcileTick():
                    _schedule_tick(ctx)
                    return Behaviors.same()

            return Behaviors.same()
        return Behaviors.receive(receive)

    return Behaviors.setup(setup)
