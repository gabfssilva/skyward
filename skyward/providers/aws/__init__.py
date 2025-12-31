"""AWS EC2 provider for Skyward.

Example:
    from skyward.providers.aws import AWS

    pool = ComputePool(
        provider=AWS(region="us-east-1"),
        accelerator="H100",
    )
"""

from skyward.providers.aws.discovery import NoAvailableRegionError
from skyward.providers.aws.provider import AWS

__all__ = ["AWS", "NoAvailableRegionError"]
