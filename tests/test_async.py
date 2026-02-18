from __future__ import annotations

from concurrent.futures import Future

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120), pytest.mark.xdist_group("pool")]


class TestAsyncExecution:
    def test_returns_future(self, pool):
        @sky.compute
        def add(a, b):
            return a + b

        future = add(2, 3) > pool
        assert isinstance(future, Future)

    def test_future_result_returns_value(self, pool):
        @sky.compute
        def add(a, b):
            return a + b

        future = add(2, 3) > pool
        assert future.result(timeout=30) == 5

    def test_multiple_futures_execute_concurrently(self, pool):
        @sky.compute
        def identity(x):
            return x

        futures = [identity(i) > pool for i in range(5)]
        results = [f.result(timeout=30) for f in futures]
        assert results == [0, 1, 2, 3, 4]
