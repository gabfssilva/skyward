"""Verda API client wrapper using verda SDK."""

from __future__ import annotations

from verda import VerdaClient


def get_client(client_id: str, client_secret: str) -> VerdaClient:
    """Create authenticated Verda client.

    Args:
        client_id: Verda client ID.
        client_secret: Verda client secret.

    Returns:
        Authenticated VerdaClient instance.
    """
    return VerdaClient(client_id, client_secret)
