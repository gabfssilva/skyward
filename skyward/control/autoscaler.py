"""Autoscaler — pressure-based scaling decisions as a plain asyncio loop.

Receives pressure reports from the task manager, computes a desired node
count, and pushes it to the reconciler. A cooldown tick re-applies the
last pressure report and reaps nodes that announced themselves idle.
"""

from __future__ import annotations

import asyncio
import time
from math import ceil
from typing import Protocol

from skyward.control.types import PressureReport
from skyward.observability.logger import logger

log = logger.bind(component="autoscaler")


class DesiredSink(Protocol):
    def set_desired(self, desired: int, reason: str) -> None: ...
    def reap_idle(self, node_ids: frozenset[int], reason: str) -> None: ...


def _compute_desired(
    report: PressureReport,
    current_desired: int,
    min_nodes: int,
    max_nodes: int,
    slots_per_node: int,
    deadline: float | None = None,
    now: float | None = None,
) -> int:
    if deadline is not None and now is not None and now > deadline:
        return 0
    if report.queued > 0:
        nodes_for_queue = ceil(report.queued / max(slots_per_node, 1))
        return min(max_nodes, max(current_desired, report.node_count + nodes_for_queue))
    if report.node_count == 0:
        return min_nodes
    return current_desired


class Autoscaler:
    def __init__(
        self,
        *,
        min_nodes: int,
        max_nodes: int,
        reconciler: DesiredSink,
        slots_per_node: int,
        initial_count: int,
        initial_nodes: frozenset[int] = frozenset(),
        cooldown: float = 30.0,
    ) -> None:
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._reconciler = reconciler
        self._slots_per_node = slots_per_node
        self._desired = initial_count
        self._cooldown = cooldown
        self._last_scale_time = time.monotonic()
        self._last_pressure: PressureReport | None = None
        self._idle: set[int] = set()
        self._reaping: set[int] = set()
        self._known_nodes: set[int] = set(initial_nodes)
        self._tick_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        if self._tick_task is not None:
            self._tick_task.cancel()
            await asyncio.gather(self._tick_task, return_exceptions=True)
            self._tick_task = None

    # ── events ────────────────────────────────────────────────────

    def report_pressure(self, report: PressureReport) -> None:
        self._last_pressure = report
        now = time.monotonic()
        if now - self._last_scale_time < self._cooldown:
            return
        new_desired = _compute_desired(
            report, self._desired, self._min_nodes, self._max_nodes,
            self._slots_per_node,
        )
        if new_desired != self._desired:
            direction = "up" if new_desired > self._desired else "down"
            log.info(
                "Scaling {dir}: {old} → {new}",
                dir=direction, old=self._desired, new=new_desired,
            )
            self._desired = new_desired
            self._last_scale_time = now
            self._reconciler.set_desired(
                new_desired,
                f"scale-{direction}: queued={report.queued} "
                f"inflight={report.inflight} capacity={report.total_capacity}",
            )

    def node_idle(self, node_id: int) -> None:
        self._idle.add(node_id)

    def node_busy(self, node_id: int) -> None:
        self._idle.discard(node_id)

    def node_joined(self, node_id: int) -> None:
        self._known_nodes.add(node_id)

    def drain_complete(self, node_id: int) -> None:
        self._known_nodes.discard(node_id)
        self._idle.discard(node_id)
        self._reaping.discard(node_id)

    def set_bounds(self, min_nodes: int, max_nodes: int, desired: int) -> None:
        log.info(
            "Bounds changed: min={min}, max={max}, desired={desired}",
            min=min_nodes, max=max_nodes, desired=desired,
        )
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._desired = max(min_nodes, min(desired, max_nodes))

    # ── tick ──────────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cooldown)
            if self._last_pressure is not None:
                self.report_pressure(self._last_pressure)
            budget = len(self._known_nodes) - len(self._reaping) - self._min_nodes
            if self._idle and budget > 0:
                k = min(len(self._idle), budget)
                chosen = frozenset(list(self._idle)[:k])
                self._reaping |= chosen
                self._idle -= chosen
                self._reconciler.reap_idle(
                    chosen, f"node-announced idle, budget={budget}",
                )
