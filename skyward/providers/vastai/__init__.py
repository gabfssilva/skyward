"""Vast.ai provider for Skyward.

Vast.ai is a GPU marketplace where hosts offer their machines for rent.
Unlike traditional cloud providers, Vast.ai instances are Docker containers
running on marketplace hosts with varying reliability and specifications.

Example:
    from skyward import ComputePool, compute
    from skyward.providers import VastAI

    @compute
    def train(data):
        return model.fit(data)

    pool = ComputePool(
        provider=VastAI(min_reliability=0.95),
        accelerator="RTX_4090",
        allocation="spot-if-available",  # Uses interruptible pricing
        pip=["torch"],
    )

    with pool:
        result = train(data) >> pool
"""

from skyward.providers.vastai.provider import VastAI

__all__ = ["VastAI"]
