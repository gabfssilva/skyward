"""End-to-end tests for Session and Compute APIs."""

from __future__ import annotations

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120), pytest.mark.xdist_group("session")]


@sky.function
def add(x: int, y: int) -> int:
    return x + y


class TestComputeSinglePool:
    def test_compute_single_pool(self) -> None:
        with sky.Compute(
            provider=sky.Container(), nodes=1,
        ) as pool:
            result = add(2, 3) >> pool
            assert result == 5

    def test_compute_sky_singleton(self) -> None:
        with sky.Compute(
            provider=sky.Container(), nodes=1,
        ):
            result = add(2, 3) >> sky.sky
            assert result == 5


class TestSessionMultiPool:
    def test_session_two_pools(self) -> None:
        with sky.Session(console=False) as session:
            pool_a = session.compute(provider=sky.Container(), nodes=1, name="a")
            pool_b = session.compute(provider=sky.Container(), nodes=1, name="b")

            result_a = add(1, 2) >> pool_a
            result_b = add(3, 4) >> pool_b

            assert result_a == 3
            assert result_b == 7
