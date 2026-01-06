"""Cloud providers for Skyward.

Providers are lazy-loaded to avoid importing heavy SDKs until needed.
This allows the core skyward package to be used on workers without
provider dependencies installed.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skyward.providers.aws import AWS
    from skyward.providers.digitalocean import DigitalOcean
    from skyward.providers.vastai import VastAI
    from skyward.providers.verda import Verda

__all__ = [
    "AWS",
    "DigitalOcean",
    "VastAI",
    "Verda",
]


def __getattr__(name: str) -> Any:
    """Lazy import providers only when accessed."""
    if name == "AWS":
        from skyward.providers.aws import AWS

        return AWS
    if name == "DigitalOcean":
        from skyward.providers.digitalocean import DigitalOcean

        return DigitalOcean
    if name == "VastAI":
        from skyward.providers.vastai import VastAI

        return VastAI
    if name == "Verda":
        from skyward.providers.verda import Verda

        return Verda
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
