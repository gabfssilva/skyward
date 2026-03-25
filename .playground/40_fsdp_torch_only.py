"""FSDP test with raw PyTorch — no accelerate, no HuggingFace Trainer.

Validates that FSDP sharding works across Skyward nodes using only
torch.distributed.fsdp and sky.plugins.torch().

Uses a large-ish model (~800M params, ~1.6GB fp16) to make sharding
observable via VRAM usage.  With FSDP FULL_SHARD across 2 nodes, each
should hold ~half the parameters.
"""

import skyward as sky


@sky.function(timeout=300)
def test_fsdp() -> dict:
    """Create a large model, wrap with FSDP, run a forward+backward pass."""
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    info = sky.instance_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Diagnostics ──────────────────────────────────────────────
    env_keys = [
        "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
    ]
    import os
    env_snapshot = {k: os.environ.get(k, "<unset>") for k in env_keys}

    pg_initialized = dist.is_initialized()
    world_size = dist.get_world_size() if pg_initialized else -1
    rank = dist.get_rank() if pg_initialized else -1

    # ── Large model (~800M params) ───────────────────────────────
    class BigModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
            )

        def forward(self, x):
            return self.layers(x)

    param_count = sum(
        4096 * 4096 + 4096 for _ in range(12)
    )

    # Load on CPU first
    model = BigModel()
    model = model.to(torch.float16)

    vram_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    # ── FSDP wrap ────────────────────────────────────────────────
    if pg_initialized and world_size > 1:
        model = FSDP(
            model.to(device),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        )
        fsdp_active = True
    else:
        model = model.to(device)
        fsdp_active = False

    vram_after_wrap = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    # ── Forward + backward ───────────────────────────────────────
    x = torch.randn(2, 4096, dtype=torch.float16, device=device)
    loss = model(x).sum()
    loss.backward()

    vram_after_fwd = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    return {
        "node": info.node,
        "is_head": info.is_head,
        "env_vars": env_snapshot,
        "pg_initialized": pg_initialized,
        "world_size": world_size,
        "rank": rank,
        "fsdp_active": fsdp_active,
        "param_count_m": round(param_count / 1e6, 1),
        "vram_before_gb": round(vram_before, 3),
        "vram_after_wrap_gb": round(vram_after_wrap, 3),
        "vram_after_fwd_gb": round(vram_after_fwd, 3),
        "peak_vram_gb": round(peak_vram, 3),
    }


if __name__ == "__main__":
    import json

    with sky.Compute(
        provider=sky.AWS(),
        accelerator=sky.accelerators.T4(),
        nodes=2,
        plugins=[sky.plugins.torch()],
    ) as compute:
        print("=" * 60)
        print("FSDP Test — raw PyTorch + sky.plugins.torch()")
        print("=" * 60)

        results = test_fsdp() @ compute

        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)

        for r in results:
            print(json.dumps(r, indent=2))
