"""End-to-end tests for compute.resize()."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(300), pytest.mark.xdist_group("resize")]


@sky.function
def add(x: int, y: int) -> int:
    return x + y


def _wait_until(value: Callable[[], int], target: int, timeout: float = 90.0, interval: float = 1.0) -> None:
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


class TestResize:
    def test_grows_and_shrinks(self) -> None:
        with sky.Compute(provider=sky.Container(), nodes=1) as compute:
            assert compute.current_nodes() == 1

            compute.resize(2)
            _wait_until(lambda: compute.current_nodes(), target=2)

            assert (add(1, 2) >> compute) == 3

            compute.resize(1)
            _wait_until(lambda: compute.current_nodes(), target=1)
