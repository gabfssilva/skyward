from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    BoundsChanged,
    DesiredCountChanged,
    DrainComplete,
    NodeBecameBusy,
    NodeBecameIdle,
    NodeJoined,
    PressureReport,
    ReapIdleNodes,
)
from skyward.actors.reconciler.messages import ReconcilerMsg
from skyward.observability.logger import logger

from .messages import AutoscalerMsg, _ScaleTick
from .state import _apply_bounds, _compute_desired, _State

log = logger.bind(actor="autoscaler")


def autoscaler_actor(
    min_nodes: int,
    max_nodes: int,
    reconciler: ActorRef[ReconcilerMsg],
    slots_per_node: int,
    initial_count: int,
    cooldown: float = 30.0,
) -> Behavior[AutoscalerMsg]:

    async def setup(ctx: ActorContext[AutoscalerMsg]) -> Behavior[AutoscalerMsg]:
        now = time.monotonic()
        state = _State(
            desired=initial_count,
            last_scale_time=now,
            last_pressure=None,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            idle=frozenset(),
            reaping=frozenset(),
            known_nodes=frozenset(),
        )

        async def _tick() -> _ScaleTick:
            await asyncio.sleep(cooldown)
            return _ScaleTick()

        ctx.pipe_to_self(
            _tick(),
            mapper=lambda r: r,
            on_failure=lambda _: _ScaleTick(),
        )
        return observing(state)

    def observing(s: _State) -> Behavior[AutoscalerMsg]:

        async def receive(
            ctx: ActorContext[AutoscalerMsg], msg: AutoscalerMsg,
        ) -> Behavior[AutoscalerMsg]:
            match msg:
                case PressureReport() as report:
                    now = time.monotonic()
                    new_s = replace(s, last_pressure=report)

                    if now - s.last_scale_time < cooldown:
                        return observing(new_s)

                    new_desired = _compute_desired(
                        report, s.desired, s.min_nodes, s.max_nodes, slots_per_node,
                    )
                    if new_desired != s.desired:
                        direction = "up" if new_desired > s.desired else "down"
                        log.info(
                            "Scaling {dir}: {old} → {new}",
                            dir=direction, old=s.desired, new=new_desired,
                        )
                        reconciler.tell(DesiredCountChanged(
                            desired=new_desired,
                            reason=f"scale-{direction}: queued={report.queued} "
                                   f"inflight={report.inflight} "
                                   f"capacity={report.total_capacity}",
                        ))
                        return observing(replace(
                            new_s, desired=new_desired, last_scale_time=now,
                        ))
                    return observing(new_s)

                case NodeBecameIdle(node_id=nid):
                    return observing(replace(s, idle=s.idle | {nid}))

                case NodeBecameBusy(node_id=nid):
                    return observing(replace(s, idle=s.idle - {nid}))

                case NodeJoined(node_id=nid):
                    return observing(replace(s, known_nodes=s.known_nodes | {nid}))

                case DrainComplete(node_id=nid):
                    return observing(replace(
                        s,
                        known_nodes=s.known_nodes - {nid},
                        idle=s.idle - {nid},
                        reaping=s.reaping - {nid},
                    ))

                case BoundsChanged() as bmsg:
                    log.info(
                        "Bounds changed: min={min}, max={max}, desired={desired}",
                        min=bmsg.min, max=bmsg.max, desired=bmsg.desired,
                    )
                    return observing(_apply_bounds(s, bmsg))

                case _ScaleTick():
                    if s.last_pressure is not None:
                        ctx.self.tell(s.last_pressure)

                    budget = len(s.known_nodes) - len(s.reaping) - s.min_nodes
                    new_s = s
                    if s.idle and budget > 0:
                        k = min(len(s.idle), budget)
                        chosen = frozenset(list(s.idle)[:k])
                        reconciler.tell(ReapIdleNodes(
                            node_ids=chosen,
                            reason=f"node-announced idle, budget={budget}",
                        ))
                        new_s = replace(s, reaping=s.reaping | chosen, idle=s.idle - chosen)

                    async def _tick() -> _ScaleTick:
                        await asyncio.sleep(cooldown)
                        return _ScaleTick()

                    ctx.pipe_to_self(
                        _tick(),
                        mapper=lambda r: r,
                        on_failure=lambda _: _ScaleTick(),
                    )
                    return observing(new_s)

            return Behaviors.same()
        return Behaviors.receive(receive)

    return Behaviors.setup(setup)
