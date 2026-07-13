"""Reconciler: drift correction, auto-repair, drain, exhaustion."""

from __future__ import annotations

import asyncio

import pytest

from skyward.control.reconciler import Reconciler

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class FakePool:
    def __init__(self, *, provision: int | None = None) -> None:
        self.scale_up_calls: list[int] = []
        self.scale_down_calls: list[int] = []
        self.drained: list[frozenset[int]] = []
        self.exhausted: list[str] = []
        self._provision = provision

    async def scale_up(self, count: int) -> int:
        self.scale_up_calls.append(count)
        return self._provision if self._provision is not None else count

    def scale_down(self, count: int) -> int:
        self.scale_down_calls.append(count)
        return count

    def drain_nodes(self, node_ids: frozenset[int]) -> int:
        self.drained.append(node_ids)
        return len(node_ids)

    def reconciliation_exhausted(self, reason: str) -> None:
        self.exhausted.append(reason)


def _reconciler(pool: FakePool, **kw: object) -> Reconciler:
    defaults: dict = {
        "min_nodes": 1,
        "desired_count": 2,
        "initial_node_ids": frozenset({0, 1}),
        "tick_interval": 3600.0,
    }
    defaults.update(kw)
    return Reconciler(pool, **defaults)


async def test_balanced_state_does_nothing() -> None:
    pool = FakePool()
    rec = _reconciler(pool)
    rec.start()
    await asyncio.sleep(0.05)
    assert pool.scale_up_calls == []
    assert pool.scale_down_calls == []
    await rec.stop()


async def test_set_desired_scales_up() -> None:
    pool = FakePool()
    rec = _reconciler(pool)
    rec.start()
    rec.set_desired(4, "test")
    await asyncio.sleep(0.05)
    assert pool.scale_up_calls == [2]
    await rec.stop()


async def test_node_lost_triggers_auto_repair() -> None:
    pool = FakePool()
    rec = _reconciler(pool)
    rec.start()
    rec.node_lost(1, "preempted")
    await asyncio.sleep(0.05)
    assert pool.scale_up_calls == [1]
    await rec.stop()


async def test_set_desired_scales_down() -> None:
    pool = FakePool()
    rec = _reconciler(pool)
    rec.start()
    rec.set_desired(1, "scale-down")
    await asyncio.sleep(0.05)
    assert pool.scale_down_calls == [1]
    await rec.stop()


async def test_node_joined_clears_pending() -> None:
    pool = FakePool()
    rec = _reconciler(pool)
    rec.start()
    rec.set_desired(3, "test")
    await asyncio.sleep(0.05)
    rec.node_joined(2)
    await asyncio.sleep(0.05)
    # one scale-up of 1 satisfies desired=3; the join must not re-trigger
    assert pool.scale_up_calls == [1]
    await rec.stop()


async def test_reap_idle_respects_min_nodes() -> None:
    pool = FakePool()
    rec = _reconciler(pool, min_nodes=2)
    rec.start()
    rec.reap_idle(frozenset({1}), "idle")
    assert pool.drained == []
    await rec.stop()


async def test_reap_idle_drains_when_min_satisfied() -> None:
    pool = FakePool()
    rec = _reconciler(pool, min_nodes=1)
    rec.start()
    rec.reap_idle(frozenset({1}), "idle")
    assert pool.drained == [frozenset({1})]
    await rec.stop()


async def test_provision_exhaustion_below_min_reports() -> None:
    pool = FakePool(provision=0)
    rec = _reconciler(
        pool,
        min_nodes=1,
        desired_count=1,
        initial_node_ids=frozenset(),
        max_provision_retries=2,
        tick_interval=0.01,
    )
    rec.start()
    await asyncio.sleep(0.2)
    assert len(pool.exhausted) >= 1
    assert len(pool.scale_up_calls) == 2
    await rec.stop()


async def test_provision_zero_with_min_satisfied_keeps_retrying() -> None:
    pool = FakePool(provision=0)
    rec = _reconciler(
        pool,
        min_nodes=1,
        desired_count=2,
        initial_node_ids=frozenset({0}),
        max_provision_retries=2,
        tick_interval=0.01,
    )
    rec.start()
    await asyncio.sleep(0.2)
    assert pool.exhausted == []
    assert len(pool.scale_up_calls) > 2
    await rec.stop()


async def test_drain_complete_updates_current() -> None:
    pool = FakePool()
    rec = _reconciler(pool, desired_count=2, initial_node_ids=frozenset({0, 1}))
    rec.start()
    rec.set_desired(1, "down")
    await asyncio.sleep(0.05)
    rec.drain_complete(1)
    await asyncio.sleep(0.05)
    # drain complete; desired==current==1, no further scaling
    assert pool.scale_up_calls == []
    assert pool.scale_down_calls == [1]
    await rec.stop()
