from __future__ import annotations

import pytest

import skyward as sky
from skyward import gather

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120), pytest.mark.xdist_group("pool")]


class TestAndOperator:
    def test_parallel_execution_returns_tuple(self, pool):
        @sky.compute
        def double(x):
            return x * 2

        @sky.compute
        def triple(x):
            return x * 3

        a, b = (double(5) & triple(5)) >> pool
        assert a == 10
        assert b == 15

    def test_three_tasks_in_parallel(self, pool):
        @sky.compute
        def identity(x):
            return x

        a, b, c = (identity(1) & identity(2) & identity(3)) >> pool
        assert (a, b, c) == (1, 2, 3)


class TestGather:
    def test_gather_returns_results_in_order(self, pool):
        @sky.compute
        def identity(x):
            return x

        results = gather(identity(1), identity(2), identity(3)) >> pool
        assert results == (1, 2, 3)

    def test_gather_streaming(self, pool):
        @sky.compute
        def identity(x):
            return x

        results = []
        for r in gather(identity(1), identity(2), identity(3), stream=True) >> pool:
            results.append(r)

        assert sorted(results) == [1, 2, 3]

    def test_gather_streaming_unordered(self, pool):
        @sky.compute
        def identity(x):
            return x

        results = list(
            gather(identity(1), identity(2), stream=True, ordered=False) >> pool
        )
        assert sorted(results) == [1, 2]
