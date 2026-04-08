from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    DesiredCountChanged,
    DrainComplete,
    NodeId,
    NodeJoined,
    ReconcilerNodeLost,
    ReconciliationExhausted,
    RequestScaleDown,
    RequestScaleUp,
    ScaleDownComplete,
    ScaleUpComplete,
    ScaleUpFailed,
)

if TYPE_CHECKING:
    from skyward.actors.pool.messages import PoolMsg

from skyward.observability.logger import logger

from .messages import ReconcilerMsg, _ReconcileTick
from .state import _State

log = logger.bind(actor="reconciler")


def reconciler_actor(
    pool: ActorRef[PoolMsg],
    min_nodes: int,
    max_nodes: int,
    initial_node_ids: frozenset[NodeId],
    tick_interval: float = 15.0,
    max_provision_retries: int = 10,
) -> Behavior[ReconcilerMsg]:

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
        desired = len(initial_node_ids)
        state = _State(desired=desired, current=initial_node_ids)
        ctx.self.tell(DesiredCountChanged(desired=desired, reason="initial"))
        log.info(
            "Reconciler started: desired={d}, current={c}",
            d=state.desired, c=len(state.current),
        )
        return watching(state)

    def _maybe_scale(
        ctx: ActorContext[ReconcilerMsg], s: _State,
    ) -> Behavior[ReconcilerMsg]:
        if s.desired > s.effective:
            count = s.desired - s.effective
            log.info("Scaling up: requesting {n} instances", n=count)
            pool.tell(RequestScaleUp(count=count))
            return waiting_for_scale_up(replace(s, pending=s.pending + count))
        if s.desired < len(s.current):
            excess = len(s.current) - s.desired
            log.info("Scaling down: requesting drain of {n} nodes", n=excess)
            pool.tell(RequestScaleDown(count=excess))
            return watching(replace(s, draining=s.draining + excess))
        return watching(s)

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
                    return _maybe_scale(ctx, replace(s, desired=desired))

                case ReconcilerNodeLost(node_id=nid, reason=reason):
                    log.warning("Node {nid} lost: {reason}", nid=nid, reason=reason)
                    new_s = replace(s, current=s.current - {nid})
                    if new_s.desired > new_s.effective:
                        count = new_s.desired - new_s.effective
                        log.info("Auto-repair: requesting {n} instances", n=count)
                        pool.tell(RequestScaleUp(count=count))
                        return waiting_for_scale_up(replace(new_s, pending=new_s.pending + count))
                    return watching(new_s)

                case NodeJoined(node_id=nid):
                    log.info("Node {nid} joined", nid=nid)
                    return watching(replace(
                        s,
                        current=s.current | {nid},
                        pending=max(0, s.pending - 1),
                        consecutive_failures=0,
                    ))

                case DrainComplete(node_id=nid):
                    return watching(replace(
                        s,
                        current=s.current - {nid},
                        draining=max(0, s.draining - 1),
                    ))

                case ScaleDownComplete(drained=n):
                    log.info("Scale down: {n} nodes drain initiated", n=n)
                    return Behaviors.same()

                case _ReconcileTick():
                    _schedule_tick(ctx)
                    if s.consecutive_failures >= max_provision_retries:
                        return Behaviors.same()
                    if s.desired > s.effective:
                        log.debug(
                            "Tick: drift detected, desired={d} effective={e}",
                            d=s.desired, e=s.effective,
                        )
                        count = s.desired - s.effective
                        pool.tell(RequestScaleUp(count=count))
                        return waiting_for_scale_up(replace(s, pending=s.pending + count))
                    return Behaviors.same()

            return Behaviors.same()
        return Behaviors.receive(receive)

    def waiting_for_scale_up(s: _State) -> Behavior[ReconcilerMsg]:

        async def receive(
            ctx: ActorContext[ReconcilerMsg], msg: ReconcilerMsg,
        ) -> Behavior[ReconcilerMsg]:
            match msg:
                case ScaleUpComplete(provisioned=n):
                    if n == 0:
                        new_failures = s.consecutive_failures + 1
                        new_s = replace(s, pending=0, consecutive_failures=new_failures)
                        if new_failures >= max_provision_retries:
                            if len(s.current) >= min_nodes:
                                log.warning(
                                    "Provision returned 0 after {n} attempts "
                                    "but min satisfied ({cur}/{min}), will keep retrying",
                                    n=new_failures, cur=len(s.current), min=min_nodes,
                                )
                                return watching(replace(new_s, consecutive_failures=0))
                            log.error(
                                "Provision exhausted ({n}/{max} attempts)",
                                n=new_failures, max=max_provision_retries,
                            )
                            pool.tell(ReconciliationExhausted(
                                reason=f"failed to provision after {new_failures} consecutive attempts",
                            ))
                        else:
                            log.warning(
                                "Provision returned 0 (attempt {n}/{max})",
                                n=new_failures, max=max_provision_retries,
                            )
                        return watching(new_s)

                    log.info("Scale up: {n} instances provisioned", n=n)
                    new_s = replace(s, consecutive_failures=0)
                    if new_s.desired > new_s.effective:
                        count = new_s.desired - new_s.effective
                        pool.tell(RequestScaleUp(count=count))
                        return waiting_for_scale_up(replace(new_s, pending=new_s.pending + count))
                    return watching(new_s)

                case ScaleUpFailed(error=error):
                    new_failures = s.consecutive_failures + 1
                    new_s = replace(s, pending=0, consecutive_failures=new_failures)
                    if new_failures >= max_provision_retries:
                        if len(s.current) >= min_nodes:
                            log.warning(
                                "Provision failed after {n} attempts "
                                "but min satisfied ({cur}/{min}), will keep retrying: {err}",
                                n=new_failures, cur=len(s.current), min=min_nodes, err=error,
                            )
                            return watching(replace(new_s, consecutive_failures=0))
                        log.error(
                            "Provision failed ({n}/{max} attempts exhausted): {err}",
                            n=new_failures, max=max_provision_retries, err=error,
                        )
                        pool.tell(ReconciliationExhausted(
                            reason=f"provision error after {new_failures} attempts: {error}",
                        ))
                    else:
                        log.error(
                            "Provision failed (attempt {n}/{max}): {err}",
                            n=new_failures, max=max_provision_retries, err=error,
                        )
                    return watching(new_s)

                case DesiredCountChanged(desired=desired, reason=reason):
                    log.info(
                        "Desired changed during scale-up: {old} → {new} ({reason})",
                        old=s.desired, new=desired, reason=reason,
                    )
                    new_s = replace(s, desired=desired)
                    if desired <= new_s.effective:
                        return watching(new_s)
                    return waiting_for_scale_up(new_s)

                case ReconcilerNodeLost(node_id=nid, reason=reason):
                    log.warning("Node {nid} lost during scale-up: {reason}", nid=nid, reason=reason)
                    return waiting_for_scale_up(replace(s, current=s.current - {nid}))

                case NodeJoined(node_id=nid):
                    log.info("Node {nid} joined during scale-up", nid=nid)
                    new_s = replace(
                        s,
                        current=s.current | {nid},
                        pending=max(0, s.pending - 1),
                        consecutive_failures=0,
                    )
                    if new_s.desired <= new_s.effective:
                        return watching(new_s)
                    return waiting_for_scale_up(new_s)

                case _ReconcileTick():
                    _schedule_tick(ctx)
                    return Behaviors.same()

            return Behaviors.same()
        return Behaviors.receive(receive)

    return Behaviors.setup(setup)
