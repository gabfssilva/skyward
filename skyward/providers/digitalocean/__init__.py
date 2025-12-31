"""DigitalOcean provider for Skyward.

Example:
    from skyward.providers.digitalocean import DigitalOcean

    pool = ComputePool(
        provider=DigitalOcean(region="nyc3"),
        accelerator="H100",
    )
"""

from skyward.providers.digitalocean.provider import DigitalOcean

__all__ = ["DigitalOcean"]
