"""AWS provider configuration.

Immutable configuration dataclass for AWS provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# =============================================================================
# Types
# =============================================================================

type AllocationStrategy = Literal[
    "price-capacity-optimized",  # Default: balance price and capacity
    "capacity-optimized",  # Prioritize available capacity
    "lowest-price",  # Cheapest (more interruptions)
]


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class AWS:
    """AWS provider configuration.

    Immutable configuration that defines how to connect to AWS and
    provision resources. All fields have sensible defaults.

    Example:
        >>> from skyward.providers.aws import AWS
        >>> config = AWS(region="us-west-2")

    Args:
        region: AWS region for resources. Default: us-east-1
        ami: Custom AMI ID. If None, uses DLAMI GPU AMI.
        subnet_id: Specific subnet. If None, uses default VPC subnets.
        security_group_id: Specific SG. If None, creates one.
        instance_profile_arn: IAM instance profile. If None, creates one.
        username: SSH username. Auto-detected from AMI if None.
        instance_timeout: Safety timeout in seconds. Default: 300.
        allocation_strategy: EC2 Fleet allocation strategy.
    """

    region: str = "us-east-1"
    ami: str | None = None
    subnet_id: str | None = None
    security_group_id: str | None = None
    instance_profile_arn: str | None = None
    username: str | None = None
    instance_timeout: int = 300
    allocation_strategy: AllocationStrategy = "price-capacity-optimized"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AWS",
    "AllocationStrategy",
]
