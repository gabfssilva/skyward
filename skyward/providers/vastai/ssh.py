"""SSH key management for Vast.ai.

Vast.ai handles SSH keys differently from traditional cloud providers:
1. Keys are registered on the account via API
2. Keys are attached to instances AFTER creation (not during)
3. SSH ports are non-standard (assigned per host)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from skyward.providers.base.ssh_keys import find_local_ssh_key

if TYPE_CHECKING:
    from vastai import VastAI


def get_or_create_ssh_key(client: VastAI) -> tuple[int, str]:
    """Get or create SSH key on Vast.ai, returning key ID and public key.

    Vast.ai workflow:
    1. Find local SSH public key
    2. List existing keys on Vast.ai
    3. If key exists (by public key match), return its ID
    4. Otherwise, create new key and return ID

    Args:
        client: Authenticated VastAI client.

    Returns:
        Tuple of (SSH key ID on Vast.ai, public key content).

    Raises:
        RuntimeError: If no local SSH key found or key creation fails.
    """
    key_info = find_local_ssh_key()
    if key_info is None:
        raise RuntimeError(
            "No SSH key found. Create one with: ssh-keygen -t ed25519\n"
            "Searched: ~/.ssh/id_ed25519.pub, ~/.ssh/id_rsa.pub, ~/.ssh/id_ecdsa.pub"
        )

    key_path, public_key = key_info
    key_name = f"skyward-{os.environ.get('USER', 'user')}-{key_path.stem}"

    # Normalize public key for comparison (strip whitespace, take key part)
    local_key_parts = public_key.strip().split()
    local_key_data = local_key_parts[1] if len(local_key_parts) >= 2 else public_key

    # List existing keys
    existing_keys = client.show_ssh_keys() or []

    # Handle both list and dict responses
    if isinstance(existing_keys, dict):
        existing_keys = existing_keys.get("ssh_keys", [])

    for key in existing_keys:
        # Vast.ai returns public_key directly (not fingerprint)
        stored_key = key.get("public_key", "")
        stored_key_parts = stored_key.strip().split()
        stored_key_data = stored_key_parts[1] if len(stored_key_parts) >= 2 else stored_key

        if stored_key_data == local_key_data:
            key_id = key.get("id")
            logger.debug(f"Found existing SSH key on Vast.ai: {key_id}")
            return key_id, public_key

    # Create new key
    logger.info(f"Creating SSH key on Vast.ai: {key_name}")
    result = client.create_ssh_key(ssh_key=public_key)

    if not result:
        raise RuntimeError("Failed to create SSH key on Vast.ai: empty response")

    # Handle various response formats
    if isinstance(result, dict):
        if not result.get("success", True):
            raise RuntimeError(f"Failed to create SSH key on Vast.ai: {result}")
        key_id = result.get("ssh_key_id") or result.get("id")
    else:
        key_id = result

    if not key_id:
        raise RuntimeError(f"Failed to get SSH key ID from response: {result}")

    logger.debug(f"Created SSH key on Vast.ai: {key_id}")
    return key_id, public_key


def attach_ssh_key_to_instance(
    client: VastAI,
    instance_id: int,
    ssh_key: str,
) -> None:
    """Attach SSH key to a Vast.ai instance.

    This must be called after instance creation. Vast.ai doesn't
    inject keys during creation like traditional cloud providers.

    Args:
        client: Authenticated VastAI client.
        instance_id: Vast.ai instance ID.
        ssh_key: Public key content (e.g., "ssh-ed25519 AAAA... user@host").

    Note:
        This function logs warnings but doesn't raise on failure,
        as SSH might already be attached or available via other means.
    """
    try:
        result = client.attach_ssh(instance_id=instance_id, ssh_key=ssh_key)

        if result and isinstance(result, dict) and not result.get("success", True):
            logger.warning(f"Failed to attach SSH key to instance {instance_id}: {result}")
        else:
            logger.debug(f"Attached SSH key to instance {instance_id}")

    except Exception as e:
        logger.warning(f"Failed to attach SSH key to instance {instance_id}: {e}")


def parse_ssh_url(ssh_url: str) -> tuple[str, int, str]:
    """Parse Vast.ai SSH URL into components.

    Args:
        ssh_url: SSH URL in format "ssh://user@host:port"

    Returns:
        Tuple of (host, port, username).

    Raises:
        ValueError: If URL format is invalid.

    Examples:
        >>> parse_ssh_url("ssh://root@vast-host.example.com:22222")
        ('vast-host.example.com', 22222, 'root')
    """
    import re

    match = re.match(r"ssh://(\w+)@([\w.\-]+):(\d+)", ssh_url)
    if not match:
        raise ValueError(f"Invalid SSH URL format: {ssh_url}")

    username = match.group(1)
    host = match.group(2)
    port = int(match.group(3))

    return host, port, username
