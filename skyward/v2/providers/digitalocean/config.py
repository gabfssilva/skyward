"""DigitalOcean provider configuration.

Immutable configuration dataclass for DigitalOcean provider.
"""

from __future__ import annotations

from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class DigitalOcean:
    """DigitalOcean provider configuration.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub and registered on DigitalOcean if needed.

    Example:
        >>> from skyward.v2.providers.digitalocean import DigitalOcean
        >>> config = DigitalOcean(region="nyc3")

    Args:
        region: DigitalOcean region (e.g., "nyc3", "sfo3", "ams3"). Default: nyc3.
        token: API token. Falls back to DIGITALOCEAN_TOKEN env var.
        ssh_key_fingerprint: Specific SSH key fingerprint to use (optional).
        instance_timeout: Safety timeout in seconds. Default: 300.
    """

    region: str = "nyc3"
    token: str | None = None
    ssh_key_fingerprint: str | None = None
    instance_timeout: int = 300


# =============================================================================
# Exports
# =============================================================================

__all__ = ["DigitalOcean"]
