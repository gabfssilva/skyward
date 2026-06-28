"""End-to-end test for scale-to-zero: a pool starts with N nodes, reaps to
zero when idle, and wakes on demand on the next task."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(300), pytest.mark.xdist_group("scale_to_zero")]


@sky.function
def add(x: int, y: int) -> int:
    return x + y


def _wait_until(
    value: Callable[[], int], target: int, timeout: float = 120.0, interval: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout
    last: int | None = None
    while time.monotonic() < deadline:
        last = value()
        if last == target:
            return
        time.sleep(interval)
    raise AssertionError(
        f"Expected value={target} within {timeout}s; last observed={last}"
    )


class TestScaleToZero:
    def test_starts_full_reaps_to_zero_and_wakes_on_demand(self) -> None:
        with sky.Compute(
            provider=sky.Container(),
            nodes=sky.Nodes(desired=5, min=0, max=5),
            options=sky.Options(autoscale_idle_timeout=15.0, autoscale_cooldown=5.0),
        ) as compute:
            _wait_until(compute.current_nodes, target=5)

            assert (add(2, 3) >> compute) == 5

            _wait_until(compute.current_nodes, target=0)

            assert (add(10, 20) >> compute) == 30
            assert compute.current_nodes() >= 1
