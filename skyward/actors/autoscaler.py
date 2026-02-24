from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from math import ceil

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    AutoscalerMsg,
    DesiredCountChanged,
    PressureReport,
    ReconcilerMsg,
    _ScaleTick,
)
from skyward.observability.logger import logger

log = logger.bind(actor="autoscaler")


def _compute_desired(
    report: PressureReport,
    current_desired: int,
    min_nodes: int,
    max_nodes: int,
    slots_per_node: int,
    now: float,
    last_busy_time: float,
    scale_down_idle_seconds: float,
) -> int:
    if report.node_count == 0:
        return min_nodes

    if report.queued > 0:
        nodes_for_queue = ceil(report.queued / max(slots_per_node, 1))
        return min(max_nodes, max(current_desired, report.node_count + nodes_for_queue))

    if report.inflight == 0 and now - last_busy_time > scale_down_idle_seconds:
        return min_nodes

    if report.queued == 0 and report.total_capacity > 0:
        utilization = report.inflight / report.total_capacity
        if utilization < 0.3 and now - last_busy_time > scale_down_idle_seconds:
            needed = ceil(report.inflight / max(slots_per_node, 1)) + 1
            return min(current_desired, max(min_nodes, needed))

    return current_desired


@dataclass(frozen=True, slots=True)
class _State:
    desired: int
    last_scale_time: float
    last_pressure: PressureReport | None
    last_busy_time: float


def autoscaler_actor(
    min_nodes: int,
    max_nodes: int,
    reconciler: ActorRef[ReconcilerMsg],
    slots_per_node: int,
    initial_count: int,
    cooldown: float = 30.0,
    scale_down_idle_seconds: float = 60.0,
) -> Behavior[AutoscalerMsg]:

    async def setup(ctx: ActorContext[AutoscalerMsg]) -> Behavior[AutoscalerMsg]:
        now = time.monotonic()
        state = _State(
            desired=initial_count,
            last_scale_time=now,
            last_pressure=None,
            last_busy_time=now,
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
                    busy = now if report.inflight > 0 or report.queued > 0 else s.last_busy_time
                    new_s = replace(s, last_pressure=report, last_busy_time=busy)

                    if now - s.last_scale_time < cooldown:
                        return observing(new_s)

                    new_desired = _compute_desired(
                        report, s.desired, min_nodes, max_nodes,
                        slots_per_node, now, busy, scale_down_idle_seconds,
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

                case _ScaleTick():
                    if s.last_pressure is not None:
                        ctx.self.tell(s.last_pressure)

                    async def _tick() -> _ScaleTick:
                        await asyncio.sleep(cooldown)
                        return _ScaleTick()

                    ctx.pipe_to_self(
                        _tick(),
                        mapper=lambda r: r,
                        on_failure=lambda _: _ScaleTick(),
                    )
                    return Behaviors.same()

            return Behaviors.same()
        return Behaviors.receive(receive)

    return Behaviors.setup(setup)
