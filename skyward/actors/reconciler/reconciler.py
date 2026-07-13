"""Reconciler — keeps the pool's node count converging on the desired count.

A plain asyncio loop: a periodic tick corrects drift, and event methods
(``node_lost``, ``node_joined``, ``set_desired``, ``reap_idle``) trigger
immediate reconciliation. Scale-ups are awaited on the pool directly,
so at most one provision request is in flight at a time.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from skyward.api import events
from skyward.observability.logger import logger

log = logger.bind(actor="reconciler")


def _no_emit(_event: events.SessionEvent) -> None:
    pass


class ReconcilerPool(Protocol):
    async def scale_up(self, count: int) -> int: ...
    def scale_down(self, count: int) -> int: ...
    def drain_nodes(self, node_ids: frozenset[int]) -> int: ...
    def reconciliation_exhausted(self, reason: str) -> None: ...


class Reconciler:
    def __init__(
        self,
        pool: ReconcilerPool,
        *,
        min_nodes: int,
        desired_count: int,
        initial_node_ids: frozenset[int],
        tick_interval: float = 15.0,
        max_provision_retries: int = 10,
        emit: events.Emit | None = None,
        pool_name: str = "",
    ) -> None:
        self._pool = pool
        self._min_nodes = min_nodes
        self._desired = desired_count
        self._current: set[int] = set(initial_node_ids)
        self._pending = 0
        self._draining = 0
        self._consecutive_failures = 0
        self._tick_interval = tick_interval
        self._max_provision_retries = max_provision_retries
        self._emit = emit or _no_emit
        self._pool_name = pool_name
        self._scale_task: asyncio.Task[None] | None = None
        self._tick_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._tick_task = asyncio.create_task(self._tick_loop())
        log.info(
            "Reconciler started: desired={d}, current={c}",
            d=self._desired, c=len(self._current),
        )
        self._reconcile()

    async def stop(self) -> None:
        tasks = [t for t in (self._tick_task, self._scale_task) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tick_task = None
        self._scale_task = None

    @property
    def _effective(self) -> int:
        return len(self._current) + self._pending

    # ── events ────────────────────────────────────────────────────

    def set_desired(self, desired: int, reason: str) -> None:
        self._emit(events.Scaling.DesiredChanged(self._pool_name, desired, reason))
        log.info(
            "Desired count changed: {old} → {new} ({reason})",
            old=self._desired, new=desired, reason=reason,
        )
        self._desired = desired
        self._reconcile()

    def set_min_nodes(self, min_nodes: int) -> None:
        self._min_nodes = min_nodes

    def node_lost(self, node_id: int, reason: str) -> None:
        log.warning("Node {nid} lost: {reason}", nid=node_id, reason=reason)
        if node_id in self._current:
            self._current.discard(node_id)
        else:
            self._pending = max(0, self._pending - 1)
        self._reconcile()

    def node_joined(self, node_id: int) -> None:
        log.info("Node {nid} joined", nid=node_id)
        if node_id not in self._current:
            self._current.add(node_id)
            self._pending = max(0, self._pending - 1)
        self._consecutive_failures = 0
        self._reconcile()

    def drain_complete(self, node_id: int) -> None:
        self._current.discard(node_id)
        self._draining = max(0, self._draining - 1)

    def reap_idle(self, node_ids: frozenset[int], reason: str) -> None:
        if len(self._current) - len(node_ids) < self._min_nodes:
            log.warning(
                "Ignoring reap_idle: would violate min_nodes "
                "(current={c}, reap={r}, min={m})",
                c=len(self._current), r=len(node_ids), m=self._min_nodes,
            )
            return
        log.info("Reaping {n} idle nodes ({reason})", n=len(node_ids), reason=reason)
        self._desired -= len(node_ids)
        drained = self._pool.drain_nodes(node_ids)
        self._draining += drained

    # ── reconciliation ────────────────────────────────────────────

    def _reconcile(self) -> None:
        if self._desired > self._effective:
            if self._scale_task is not None and not self._scale_task.done():
                return
            if self._consecutive_failures >= self._max_provision_retries:
                return
            count = self._desired - self._effective
            log.info("Scaling up: requesting {n} instances", n=count)
            self._pending += count
            self._scale_task = asyncio.create_task(self._do_scale_up(count))
        elif self._desired < len(self._current) and self._draining == 0:
            excess = len(self._current) - self._desired
            log.info("Scaling down: requesting drain of {n} nodes", n=excess)
            drained = self._pool.scale_down(excess)
            self._draining += drained

    async def _do_scale_up(self, requested: int) -> None:
        try:
            provisioned = await self._pool.scale_up(requested)
        except Exception as e:  # noqa: BLE001 — provision failure is a counted outcome
            log.error("Provision failed: {err}", err=e)
            provisioned = 0
        self._pending = max(0, self._pending - (requested - provisioned))
        if provisioned == 0:
            self._pending = 0
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_provision_retries:
                if len(self._current) >= self._min_nodes:
                    log.warning(
                        "Provision returned 0 after {n} attempts but min satisfied "
                        "({cur}/{min}), will keep retrying",
                        n=self._consecutive_failures,
                        cur=len(self._current), min=self._min_nodes,
                    )
                    self._consecutive_failures = 0
                else:
                    log.error(
                        "Provision exhausted ({n}/{max} attempts)",
                        n=self._consecutive_failures, max=self._max_provision_retries,
                    )
                    self._pool.reconciliation_exhausted(
                        f"failed to provision after {self._consecutive_failures} "
                        "consecutive attempts",
                    )
            else:
                log.warning(
                    "Provision returned 0 (attempt {n}/{max})",
                    n=self._consecutive_failures, max=self._max_provision_retries,
                )
            return
        log.info("Scale up: {n} instances provisioned", n=provisioned)
        self._consecutive_failures = 0
        self._reconcile()

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(self._tick_interval)
            if self._consecutive_failures >= self._max_provision_retries:
                continue
            self._reconcile()
