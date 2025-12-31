"""Verda Cloud provider for Skyward.

Example:
    from skyward.providers.verda import Verda

    pool = ComputePool(
        provider=Verda(region="FIN-01"),
        accelerator="H100",
    )
"""

from skyward.providers.verda.discovery import NoAvailableRegionError
from skyward.providers.verda.provider import Verda

__all__ = ["Verda", "NoAvailableRegionError"]
