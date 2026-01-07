"""
MultiPool - Parallel provisioning for multiple pools.

When you need different resource configurations for different stages
of your workflow, MultiPool provisions them concurrently to minimize
total setup time.

    Sequential: sum(t_i) = t1 + t2 + t3
    MultiPool:  max(t_i) = max(t1, t2, t3)
"""

import skyward as sky


@sky.compute
def preprocess(data: dict) -> dict:
    """CPU-intensive preprocessing."""
    # Simulate preprocessing
    return {"processed": True, **data}


@sky.compute
def train_model(data: dict) -> dict:
    """GPU-intensive training."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"trained": True, "device": device, **data}


@sky.compute
def evaluate(data: dict) -> dict:
    """Evaluation on smaller GPU."""
    return {"evaluated": True, "score": 0.95, **data}


def main():
    # Define pools with different configurations
    cpu_pool = sky.ComputePool(
        provider=sky.AWS(),
        cpu=8,
        memory="16GB",
        allocation="spot-if-available",
    )

    gpu_pool = sky.ComputePool(
        provider=sky.AWS(),
        accelerator="A100",
        nodes=2,
        image=sky.Image(pip=["torch"]),
        allocation="spot-if-available",
    )

    eval_pool = sky.ComputePool(
        provider=sky.AWS(),
        accelerator="T4",
        image=sky.Image(pip=["torch"]),
        allocation="spot-if-available",
    )

    # MultiPool provisions all pools in parallel
    # Total setup time = max(cpu_setup, gpu_setup, eval_setup)
    # Instead of:     = cpu_setup + gpu_setup + eval_setup
    with sky.MultiPool(cpu_pool, gpu_pool, eval_pool) as (cpu, gpu, eval_):
        # Stage 1: Preprocess on CPU pool
        data = preprocess({"raw": "data"}) >> cpu
        print(f"Preprocessed: {data}")

        # Stage 2: Train on GPU pool (broadcast to both nodes)
        results = train_model(data) @ gpu
        print(f"Trained on {len(results)} nodes")

        # Stage 3: Evaluate on smaller GPU
        final = evaluate(data) >> eval_
        print(f"Final score: {final['score']}")


if __name__ == "__main__":
    main()
