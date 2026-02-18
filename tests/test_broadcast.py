from __future__ import annotations

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120), pytest.mark.xdist_group("pool")]


class TestBroadcast:
    def test_executes_on_all_nodes(self, pool):
        @sky.compute
        def whoami():
            info = sky.instance_info()
            return info.node if info else None

        nodes = whoami() @ pool
        assert sorted(nodes) == [0, 1]

    def test_returns_list_with_length_equal_to_nodes(self, pool):
        @sky.compute
        def ping():
            return "pong"

        results = ping() @ pool
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_node_produces_different_output(self, pool):
        @sky.compute
        def node_specific():
            info = sky.instance_info()
            return f"node-{info.node}" if info else None

        results = sorted(node_specific() @ pool)
        assert results == ["node-0", "node-1"]
