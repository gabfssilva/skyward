# pyright: reportOptionalMemberAccess=false
from __future__ import annotations

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120), pytest.mark.xdist_group("pool")]


class TestInstanceInfo:
    def test_returns_instance_info(self, pool):
        @sky.compute
        def get_info():
            info = sky.instance_info()
            return {
                "node": info.node,
                "total_nodes": info.total_nodes,
                "is_head": info.is_head,
            }

        result = get_info() >> pool
        assert result["total_nodes"] == 2
        assert isinstance(result["node"], int)
        assert result["node"] in (0, 1)

    def test_head_node_is_node_zero(self, pool):
        @sky.compute
        def check_head():
            info = sky.instance_info()
            return info.is_head == (info.node == 0)

        results = check_head() @ pool
        assert all(results)

    def test_each_node_has_unique_index(self, pool):
        @sky.compute
        def get_node():
            return sky.instance_info().node

        nodes = get_node() @ pool
        assert len(set(nodes)) == 2

    def test_job_id_consistent(self, pool):
        @sky.compute
        def get_job_id():
            return sky.instance_info().job_id

        job_ids = get_job_id() @ pool
        assert len(set(job_ids)) == 1


class TestShard:
    def test_shard_list(self, pool):
        @sky.compute
        def shard_data():
            data = list(range(10))
            return sky.shard(data)

        results = shard_data() @ pool
        combined = []
        for chunk in results:
            combined.extend(chunk)
        assert sorted(combined) == list(range(10))

    def test_shard_preserves_coverage(self, pool):
        @sky.compute
        def shard_and_count():
            data = list(range(100))
            chunk = sky.shard(data)
            return len(chunk)

        counts = shard_and_count() @ pool
        assert sum(counts) == 100

    def test_shard_multiple_arrays(self, pool):
        @sky.compute
        def shard_aligned():
            x = list(range(10))
            y = [v * 10 for v in range(10)]
            x_chunk, y_chunk = sky.shard(x, y)
            return list(zip(x_chunk, y_chunk, strict=True))

        results = shard_aligned() @ pool
        all_pairs = []
        for chunk in results:
            all_pairs.extend(chunk)
        for x, y in all_pairs:
            assert y == x * 10

    def test_shard_with_shuffle(self, pool):
        @sky.compute
        def shard_shuffled():
            data = list(range(20))
            return sky.shard(data, shuffle=True, seed=42)

        results = shard_shuffled() @ pool
        combined = []
        for chunk in results:
            combined.extend(chunk)
        assert sorted(combined) == list(range(20))
