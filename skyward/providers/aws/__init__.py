"""AWS Provider for Skyward.

NOTE: Only config classes are imported at package level to avoid SDK deps.

Environment Variables:
    AWS_ACCESS_KEY_ID: AWS access key (required)
    AWS_SECRET_ACCESS_KEY: AWS secret key (required)
    AWS_DEFAULT_REGION: Default region (optional, overridden by config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import Client, EC2ClientFactory
    from .provider import AWSProvider
    from .types import (
        AWSOfferSpecific,
        AWSResources,
        AWSSpecific,
        IAMStatement,
        InstanceResources,
        InstanceSpec,
    )

from .config import AWS, AllocationStrategy


def __getattr__(name: str) -> Any:
    if name in ("Client", "EC2ClientFactory"):
        from .client import Client, EC2ClientFactory

        return {"Client": Client, "EC2ClientFactory": EC2ClientFactory}[name]
    if name == "AWSProvider":
        from .provider import AWSProvider

        return AWSProvider
    if name in (
        "AWSOfferSpecific", "AWSResources", "AWSSpecific",
        "IAMStatement", "InstanceResources", "InstanceSpec",
    ):
        from . import types

        return getattr(types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AWS",
    "AllocationStrategy",
    "AWSProvider",
    "AWSOfferSpecific",
    "AWSResources",
    "AWSSpecific",
    "Client",
    "EC2ClientFactory",
    "IAMStatement",
    "InstanceResources",
    "InstanceSpec",
]
