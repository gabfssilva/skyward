from __future__ import annotations

import pytest

import skyward as sky

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(300),
    pytest.mark.slow,
    pytest.mark.xdist_group("torch"),
]


class TestTorchIntegration:
    def test_torch_sets_env_vars(self, torch_pool):
        @sky.compute
        @sky.integrations.torch(backend="gloo")
        def check_env():
            import os

            return {
                "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
                "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
                "RANK": os.environ.get("RANK"),
            }

        results = check_env() @ torch_pool
        for result in results:
            assert result["MASTER_ADDR"] is not None
            assert result["WORLD_SIZE"] == "2"
            assert result["RANK"] in ("0", "1")

    def test_torch_init_process_group(self, torch_pool):
        @sky.compute
        @sky.integrations.torch(backend="gloo")
        def check_distributed():
            import torch.distributed as dist

            return dist.is_initialized()

        results = check_distributed() @ torch_pool
        assert all(results)
