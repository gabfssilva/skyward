"""
Monte Carlo π estimation.

    ┌─────────────┐
    │ ·  ·  ○  ·  │   ○ = inside quarter circle
    │  ○  ○    ○  │   · = outside
    │   ○  ○  ○   │
    │  ○  ○  ○  · │   π ≈ 4 × (○ count) / total
    │ ○  ○  ·  ·  │
    └─────────────┘
"""

import skyward as sky


@sky.compute
def estimate_pi(samples: int, seed: int = 0) -> float:
    """Estimate π via Monte Carlo"""
    from jax import random

    key = random.PRNGKey(seed)
    x, y = random.uniform(key, (2, samples))

    return float(4 * ((x**2 + y**2) <= 1).mean())


@sky.pool(
    provider=['aws', 'verda'],
    accelerator='L4',
    image=sky.Image(pip=["jax[cuda12]"]),
)
def main():
    π = estimate_pi(samples=100_000_000) >> sky
    print(f"π ≈ {π}")


if __name__ == "__main__":
    main()
