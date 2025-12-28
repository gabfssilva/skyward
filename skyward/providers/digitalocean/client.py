"""DigitalOcean API client wrapper using pydo SDK."""

from __future__ import annotations

from pydo import Client


def get_client(token: str) -> Client:
    """Create authenticated pydo client.

    Args:
        token: DigitalOcean API token.

    Returns:
        Authenticated pydo Client instance.
    """
    return Client(token=token)
