from __future__ import annotations

from collections.abc import Iterator

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120), pytest.mark.xdist_group("pool")]


class TestOutputStreaming:
    def test_generator_returns_iterable(self, pool):
        @sky.compute
        def counter(n):
            yield from range(n)

        result = counter(5) >> pool
        assert list(result) == [0, 1, 2, 3, 4]

    def test_generator_single_yield(self, pool):
        @sky.compute
        def single():
            yield 42

        result = single() >> pool
        assert list(result) == [42]

    def test_generator_empty(self, pool):
        @sky.compute
        def empty():
            return
            yield  # noqa: RET504

        result = empty() >> pool
        assert list(result) == []

    def test_generator_yields_complex_types(self, pool):
        @sky.compute
        def dicts(n):
            for i in range(n):
                yield {"index": i, "squared": i**2}

        result = dicts(3) >> pool
        assert list(result) == [
            {"index": 0, "squared": 0},
            {"index": 1, "squared": 1},
            {"index": 2, "squared": 4},
        ]


class TestInputStreaming:
    def test_iterator_param_consumed(self, pool):
        @sky.compute
        def summer(data: Iterator[int]) -> int:
            return sum(data)

        result = summer(iter(range(100))) >> pool
        assert result == 4950

    def test_iterator_with_regular_args(self, pool):
        @sky.compute
        def multiply_sum(data: Iterator[int], factor: int) -> int:
            return sum(data) * factor

        result = multiply_sum(iter(range(10)), 2) >> pool
        assert result == 90


class TestBidirectional:
    def test_input_and_output_streaming(self, pool):
        @sky.compute
        def transform(data: Iterator[int]):
            for x in data:
                yield x * 2

        result = transform(iter(range(5))) >> pool
        assert list(result) == [0, 2, 4, 6, 8]


class TestStreamingBroadcast:
    def test_generator_broadcast(self, pool):
        @sky.compute
        def counter(n):
            yield from range(n)

        results = counter(3) @ pool
        for stream in results:
            assert list(stream) == [0, 1, 2]


class TestStreamingParallel:
    def test_generator_in_parallel_group(self, pool):
        @sky.compute
        def counter(n):
            yield from range(n)

        @sky.compute
        def add(a, b):
            return a + b

        gen_result, sum_result = (counter(3) & add(1, 2)) >> pool
        assert list(gen_result) == [0, 1, 2]
        assert sum_result == 3
