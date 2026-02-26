from __future__ import annotations

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180), pytest.mark.xdist_group("torch")]


class TestTorchPlugin:
    def test_process_group_initialized(self, torch_plugin_pool) -> None:
        @sky.compute
        def check_init():
            import torch.distributed as dist

            return dist.is_initialized()

        results = check_init() @ torch_plugin_pool
        assert all(results)

    def test_correct_rank_and_world_size(self, torch_plugin_pool) -> None:
        @sky.compute
        def check_rank():
            import torch.distributed as dist

            info = sky.instance_info()
            return {
                "rank": dist.get_rank(),
                "world_size": dist.get_world_size(),
                "node": info.node,
                "total_nodes": info.total_nodes,
            }

        results = check_rank() @ torch_plugin_pool
        for r in results:
            assert r["rank"] == r["node"]
            assert r["world_size"] == r["total_nodes"] == 2

    def test_allreduce(self, torch_plugin_pool) -> None:
        @sky.compute
        def allreduce():
            import torch
            import torch.distributed as dist

            rank = dist.get_rank()
            tensor = torch.tensor([rank + 1.0])
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor.item()

        results = allreduce() @ torch_plugin_pool
        # sum of (rank+1) for ranks 0,1 → 1 + 2 = 3
        assert all(r == 3.0 for r in results)
