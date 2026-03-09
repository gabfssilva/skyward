"""
Multi-pool session - provision multiple pools within one session.

When you need different resource configurations for different stages
of your workflow, Session provisions them and shares infrastructure.
"""

import skyward as sky


@sky.function
def preprocess(data: dict) -> dict:
    """CPU-intensive preprocessing."""
    # Simulate preprocessing
    return {"processed": True, **data}


@sky.function
def train_model(data: dict) -> dict:
    """GPU-intensive training."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"trained": True, "device": device, **data}


@sky.function
def evaluate(data: dict) -> dict:
    """Evaluation on smaller GPU."""
    return {"evaluated": True, "score": 0.95, **data}


def main():
    with sky.Session(console=False) as session:
        cpu = session.compute(
            provider=sky.AWS(),
            vcpus=8,
            memory_gb=16,
            allocation="spot-if-available",
            name="cpu",
        )

        gpu = session.compute(
            provider=sky.AWS(),
            accelerator=sky.accelerators.A100(),
            nodes=2,
            image=sky.Image(pip=["torch"]),
            allocation="spot-if-available",
            name="gpu",
        )

        eval_ = session.compute(
            provider=sky.AWS(),
            accelerator=sky.accelerators.T4(),
            image=sky.Image(pip=["torch"]),
            allocation="spot-if-available",
            name="eval",
        )

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
