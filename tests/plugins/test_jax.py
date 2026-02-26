from __future__ import annotations

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180), pytest.mark.xdist_group("jax")]


class TestJAXPlugin:
    def test_distributed_initialized(self, jax_plugin_pool) -> None:
        @sky.compute
        def check_init():
            import jax

            return jax.process_count()

        results = check_init() @ jax_plugin_pool
        assert all(r == 2 for r in results)

    def test_process_index_matches_node(self, jax_plugin_pool) -> None:
        @sky.compute
        def check_index():
            import jax

            info = sky.instance_info()
            assert info is not None
            return {
                "process_index": jax.process_index(),
                "node": info.node,
            }

        results = check_index() @ jax_plugin_pool
        for r in results:
            assert r["process_index"] == r["node"]
