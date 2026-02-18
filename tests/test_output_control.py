from __future__ import annotations

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120), pytest.mark.xdist_group("pool")]


class TestStdoutControl:
    def test_head_only_stdout(self, pool):
        @sky.compute
        @sky.stdout(only="head")
        def print_node():
            import sys

            print("hello from node")
            sys.stdout.flush()
            info = sky.instance_info()
            return info.node if info else None

        results = print_node() @ pool
        assert sorted(results) == [0, 1]

    def test_predicate_based_stdout(self, pool):
        @sky.compute
        @sky.stdout(only=lambda i: i.node == 0)
        def print_node():
            print("hello")
            info = sky.instance_info()
            return info.node if info else None

        results = print_node() @ pool
        assert sorted(results) == [0, 1]


class TestSilent:
    def test_silent_suppresses_output(self, pool):
        @sky.compute
        @sky.silent
        def noisy():
            print("this should be silenced")
            return 42

        result = noisy() >> pool
        assert result == 42

    def test_silent_preserves_return_value(self, pool):
        @sky.compute
        @sky.silent
        def compute_with_print():
            print("noise")
            return {"key": "value"}

        result = compute_with_print() >> pool
        assert result == {"key": "value"}
