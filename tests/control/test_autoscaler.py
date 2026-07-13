"""Autoscaler: pressure-driven desired count and idle reaping."""

from __future__ import annotations

import asyncio

import pytest

from skyward.control.autoscaler import Autoscaler
from skyward.control.autoscaler import _compute_desired
from skyward.control.types import PressureReport

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class FakeReconciler:
    def __init__(self) -> None:
        self.desired: list[tuple[int, str]] = []
        self.reaped: list[frozenset[int]] = []

    def set_desired(self, desired: int, reason: str) -> None:
        self.desired.append((desired, reason))

    def reap_idle(self, node_ids: frozenset[int], reason: str) -> None:
        self.reaped.append(node_ids)


def _autoscaler(rec: FakeReconciler, **kw: object) -> Autoscaler:
    defaults: dict = {
        "min_nodes": 1,
        "max_nodes": 8,
        "reconciler": rec,
        "slots_per_node": 2,
        "initial_count": 2,
        "initial_nodes": frozenset({0, 1}),
        "cooldown": 0.05,
    }
    defaults.update(kw)
    return Autoscaler(**defaults)


class TestComputeDesired:
    def test_queued_pressure_grows(self) -> None:
        report = PressureReport(queued=5, inflight=4, total_capacity=4, node_count=2)
        assert _compute_desired(report, 2, 1, 8, 2) == 5  # 2 nodes + ceil(5/2)

    def test_capped_at_max(self) -> None:
        report = PressureReport(queued=100, inflight=4, total_capacity=4, node_count=2)
        assert _compute_desired(report, 2, 1, 8, 2) == 8

    def test_no_queue_keeps_desired(self) -> None:
        report = PressureReport(queued=0, inflight=1, total_capacity=4, node_count=2)
        assert _compute_desired(report, 2, 1, 8, 2) == 2

    def test_zero_nodes_returns_min(self) -> None:
        report = PressureReport(queued=0, inflight=0, total_capacity=0, node_count=0)
        assert _compute_desired(report, 2, 1, 8, 2) == 1


async def test_pressure_after_cooldown_sets_desired() -> None:
    rec = FakeReconciler()
    scaler = _autoscaler(rec, cooldown=0.0)
    scaler.report_pressure(
        PressureReport(queued=4, inflight=4, total_capacity=4, node_count=2),
    )
    assert rec.desired
    assert rec.desired[0][0] == 4


async def test_pressure_within_cooldown_is_deferred() -> None:
    rec = FakeReconciler()
    scaler = _autoscaler(rec, cooldown=3600.0)
    scaler.report_pressure(
        PressureReport(queued=4, inflight=4, total_capacity=4, node_count=2),
    )
    assert rec.desired == []


async def test_tick_reaps_idle_nodes_within_budget() -> None:
    rec = FakeReconciler()
    scaler = _autoscaler(rec, min_nodes=1, cooldown=0.02)
    scaler.node_idle(0)
    scaler.node_idle(1)
    scaler.start()
    await asyncio.sleep(0.1)
    await scaler.stop()
    assert rec.reaped
    assert len(rec.reaped[0]) == 1  # budget = 2 known - 0 reaping - 1 min


async def test_no_reap_when_budget_exhausted() -> None:
    rec = FakeReconciler()
    scaler = _autoscaler(rec, min_nodes=2, cooldown=0.02)
    scaler.node_idle(0)
    scaler.start()
    await asyncio.sleep(0.1)
    await scaler.stop()
    assert rec.reaped == []


async def test_node_busy_cancels_idle() -> None:
    rec = FakeReconciler()
    scaler = _autoscaler(rec, min_nodes=0, cooldown=0.02)
    scaler.node_idle(0)
    scaler.node_busy(0)
    scaler.start()
    await asyncio.sleep(0.1)
    await scaler.stop()
    assert rec.reaped == []


async def test_drain_complete_forgets_node() -> None:
    rec = FakeReconciler()
    scaler = _autoscaler(rec)
    scaler.node_idle(1)
    scaler.drain_complete(1)
    assert 1 not in scaler._idle
    assert 1 not in scaler._known_nodes


async def test_set_bounds_clamps_desired() -> None:
    rec = FakeReconciler()
    scaler = _autoscaler(rec)
    scaler.set_bounds(2, 4, 10)
    assert scaler._desired == 4
    scaler.set_bounds(3, 8, 1)
    assert scaler._desired == 3
