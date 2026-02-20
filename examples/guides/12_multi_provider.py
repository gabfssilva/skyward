"""Multi-Provider Selection — compare prices across clouds."""

import skyward as sky


@sky.compute
def train(epochs: int) -> dict:
    """Simulate a training run and return metrics."""
    import time

    info = sky.instance_info()
    start = time.perf_counter()
    total = 0.0
    for _ in range(epochs):
        total += sum(range(10_000))
    duration = time.perf_counter() - start
    return {
        "node": info.node if info else -1,
        "epochs": epochs,
        "duration": round(duration, 3),
    }


if __name__ == "__main__":
    # Cheapest across providers
    with sky.ComputePool(
        sky.Spec(provider=sky.VastAI(), accelerator="A100"),
        sky.Spec(provider=sky.AWS(), accelerator="A100"),
        selection="cheapest",
        image=sky.Image(pip=["torch"]),
    ) as pool:
        result = train(10) >> pool
        print(f"Cheapest: {result}")

    # First available (priority order)
    with sky.ComputePool(
        sky.Spec(provider=sky.RunPod(), accelerator="H100", nodes=4),
        sky.Spec(provider=sky.AWS(), accelerator="H100", nodes=4),
        selection="first",
        image=sky.Image(pip=["torch"]),
    ) as pool:
        results = train(10) @ pool
        print(f"First available: {results}")

    # Per-spec constraints
    with sky.ComputePool(
        sky.Spec(
            provider=sky.VastAI(),
            accelerator="A100",
            max_hourly_cost=2.50,
            allocation="spot",
        ),
        sky.Spec(
            provider=sky.Verda(),
            accelerator="A100",
            allocation="spot-if-available",
        ),
        sky.Spec(
            provider=sky.AWS(),
            accelerator="A100",
            allocation="on-demand",
        ),
        selection="cheapest",
        image=sky.Image(pip=["torch"]),
    ) as pool:
        result = train(10) >> pool
        print(f"Constrained: {result}")
