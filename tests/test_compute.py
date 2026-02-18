from __future__ import annotations

import pytest

import skyward as sky
from skyward import PendingCompute

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120), pytest.mark.xdist_group("pool")]


class TestComputeDecorator:
    def test_returns_pending_compute(self):
        @sky.compute
        def add(a, b):
            return a + b

        pending = add(2, 3)
        assert isinstance(pending, PendingCompute)

    def test_does_not_execute_on_call(self):
        calls = []

        @sky.compute
        def side_effect():
            calls.append(1)
            return 42

        side_effect()
        assert calls == []


class TestSingleDispatch:
    def test_execute_returns_result(self, pool):
        @sky.compute
        def add(a, b):
            return a + b

        result = add(2, 3) >> pool
        assert result == 5

    def test_execute_with_kwargs(self, pool):
        @sky.compute
        def greet(name, greeting="hello"):
            return f"{greeting} {name}"

        result = greet("world", greeting="hi") >> pool
        assert result == "hi world"

    def test_closure_captures_locals(self, pool):
        factor = 10

        @sky.compute
        def multiply(x):
            return x * factor

        result = multiply(5) >> pool
        assert result == 50

    def test_returns_complex_type(self, pool):
        @sky.compute
        def make_dict():
            return {"key": "value", "nested": [1, 2, 3]}

        result = make_dict() >> pool
        assert result == {"key": "value", "nested": [1, 2, 3]}

    def test_worker_exception_propagates(self, pool):
        @sky.compute
        def fail():
            raise ValueError("boom")

        with pytest.raises(Exception, match="boom"):
            fail() >> pool  # noqa: B018  # pyright: ignore[reportUnusedExpression]

    def test_sky_singleton_resolves_module_pool(self, pool):
        @sky.compute
        def ping():
            return "pong"

        result = ping() >> sky
        assert result == "pong"
