"""Preemption handling configuration for spot/bid instances.

Defines the policy for handling instance preemption (outbid, capacity reclaim, etc.)
on cloud providers that support interruptible/spot instances.

Example:
    # Simple - string shorthand
    pool = ComputePool(
        preemption="replace",  # auto-replace preempted instances
    )

    # Configured - full control
    pool = ComputePool(
        preemption=Preemption(
            policy="replace",
            max_retries=5,
            monitor_interval=5.0,
        ),
    )

    # Disabled
    pool = ComputePool(
        preemption=None,  # or "ignore"
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type PreemptionPolicy = Literal["replace", "fail", "ignore"]


@dataclass(frozen=True, slots=True)
class Preemption:
    """Configuration for spot/bid instance preemption handling.

    Attributes:
        policy: What to do when an instance is preempted.
            - "replace": Automatically provision a replacement instance.
            - "fail": Raise PreemptionError immediately.
            - "ignore": Log the event and continue without the instance.
        max_retries: Maximum replacement attempts before failing (policy="replace").
        retry_delay: Seconds to wait between retry attempts.
        monitor_interval: Seconds between status checks for preemption detection.
    """

    policy: PreemptionPolicy = "replace"
    max_retries: int = 3
    retry_delay: float = 5.0
    monitor_interval: float = 10.0


type PreemptionConfig = PreemptionPolicy | Preemption | None


def normalize_preemption(config: PreemptionConfig) -> Preemption:
    """Normalize preemption config to Preemption instance.

    Args:
        config: String policy, Preemption instance.

    Returns:
        Preemption instance with the specified policy.
    """
    match config:
        case None:
            return Preemption(policy="fail")
        case Preemption():
            return config
        case str() as policy:
            return Preemption(policy=policy)  # type: ignore[arg-type]


__all__ = ["Preemption", "PreemptionConfig", "PreemptionPolicy", "normalize_preemption"]
