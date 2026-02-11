"""Shared constants for providers.

Centralizes timeout values, intervals, and other magic numbers
to ensure consistency across provider implementations.
"""

from __future__ import annotations

# =============================================================================
# Timeouts (seconds)
# =============================================================================

DEFAULT_INSTANCE_LAUNCH_TIMEOUT = 300.0
"""Maximum time to wait for instance to reach running state."""

DEFAULT_BOOTSTRAP_TIMEOUT = 600.0
"""Maximum time to wait for bootstrap to complete."""

DEFAULT_SSH_TIMEOUT = 60.0
"""Maximum time to wait for SSH connection."""

DEFAULT_SSH_WAIT_TIMEOUT = 300.0
"""Maximum time to wait for SSH to become available (with retries)."""

# =============================================================================
# Polling Intervals (seconds)
# =============================================================================

DEFAULT_POLL_INTERVAL = 5.0
"""Default interval between polling attempts."""

DEFAULT_SSH_POLL_INTERVAL = 5.0
"""Interval between SSH connection attempts."""

# =============================================================================
# Ports
# =============================================================================

DEFAULT_SSH_PORT = 22
"""Default SSH port."""

# =============================================================================
# Billing
# =============================================================================

DEFAULT_BILLING_INCREMENT = 1
"""Default billing increment (seconds). AWS: per-minute, VastAI: per-second."""
