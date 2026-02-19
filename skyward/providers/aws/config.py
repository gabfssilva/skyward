"""AWS provider configuration.

Immutable configuration dataclass for AWS provider.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Literal

from skyward.api.provider import ProviderConfig

if typing.TYPE_CHECKING:
    from skyward.providers.aws.provider import AWSProvider

type AllocationStrategy = Literal[
    "price-capacity-optimized",  # Default: balance price and capacity
    "capacity-optimized",  # Prioritize available capacity
    "lowest-price",  # Cheapest (more interruptions)
]

type UbuntuVersion = Literal["20.04", "22.04", "24.04"] | str


@dataclass(frozen=True, slots=True)
class AWS(ProviderConfig):
    """AWS provider configuration.

    Immutable configuration that defines how to connect to AWS and
    provision resources. All fields have sensible defaults.

    Example:
        >>> from skyward.providers.aws import AWS
        >>> config = AWS(region="us-west-2")

    Args:
        region: AWS region for resources. Default: us-east-1
        ami: Custom AMI ID. If None, resolves via SSM Parameter Store.
        ubuntu_version: Ubuntu LTS version for auto-resolved AMIs.
        subnet_id: Specific subnet. If None, uses default VPC subnets.
        security_group_id: Specific SG. If None, creates one.
        instance_profile_arn: IAM instance profile. If None, creates one.
        username: SSH username. Auto-detected from AMI if None.
        instance_timeout: Safety timeout in seconds. Default: 300.
        allocation_strategy: EC2 Fleet allocation strategy.
        exclude_burstable: Exclude burstable instances (t3, t4g, etc.).
    """

    region: str = "us-east-1"
    ami: str | None = None
    ubuntu_version: UbuntuVersion = "24.04"
    subnet_id: str | None = None
    security_group_id: str | None = None
    instance_profile_arn: str | None = None
    username: str | None = None
    instance_timeout: int = 300
    request_timeout: int = 30
    allocation_strategy: AllocationStrategy = "price-capacity-optimized"
    exclude_burstable: bool = False

    @property
    def type(self) -> str: return "aws"

    async def create_provider(self) -> AWSProvider:
        from skyward.providers.aws.provider import AWSProvider
        return await AWSProvider.create(self)
