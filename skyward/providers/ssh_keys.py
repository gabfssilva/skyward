"""SSH key utilities for cloud providers.

Provides shared functionality for SSH key management across providers.
"""

from __future__ import annotations

import base64
import hashlib
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


def compute_fingerprint(public_key: str) -> str:
    """Compute SSH key fingerprint (MD5 colon-separated format).

    MD5 colon-separated format used by cloud providers.

    Args:
        public_key: SSH public key content (e.g., "ssh-ed25519 AAAA... user@host")

    Returns:
        Fingerprint in format "aa:bb:cc:..." or empty string on error.
    """
    try:
        parts = public_key.strip().split()
        if len(parts) >= 2:
            decoded = base64.b64decode(parts[1])
            digest = hashlib.md5(decoded).hexdigest()
            return ":".join(digest[i : i + 2] for i in range(0, len(digest), 2))
    except Exception:
        pass
    return ""


def get_local_ssh_key() -> tuple[Path, str]:
    """Find local SSH key and return (path, public_key_content).

    Searches common SSH key locations in order of preference:
    - id_ed25519 (recommended)
    - id_rsa
    - id_ecdsa

    Returns:
        Tuple of (public_key_path, public_key_content).

    Raises:
        RuntimeError: If no SSH key is found.
    """
    key_names = ["id_ed25519", "id_rsa", "id_ecdsa"]
    ssh_dir = Path.home() / ".ssh"

    for name in key_names:
        public_path = ssh_dir / f"{name}.pub"
        private_path = ssh_dir / name
        if public_path.exists() and private_path.exists():
            return public_path, public_path.read_text().strip()

    raise RuntimeError(
        "No SSH key found. Create one with: ssh-keygen -t ed25519"
    )


def get_ssh_key_path() -> str:
    """Get path to local SSH private key.

    Returns:
        Absolute path to the private key.

    Raises:
        RuntimeError: If no SSH key is found.
    """
    public_path, _ = get_local_ssh_key()
    private_path = public_path.with_suffix("")
    return str(private_path)


def generate_key_name(key_path: Path) -> str:
    """Generate a standard key name for cloud provider registration.

    Args:
        key_path: Path to the public key file.

    Returns:
        Key name in format "skyward-{user}-{key_stem}".
    """
    user = os.environ.get("USER", "user")
    return f"skyward-{user}-{key_path.stem}"


async def ensure_ssh_key_on_provider(
    list_keys_fn: Callable[[], Awaitable[list[dict[str, Any]]]],
    create_key_fn: Callable[[str, str], Awaitable[dict[str, Any]]],
    provider_name: str,
) -> str:
    """Ensure SSH key exists on a cloud provider.

    This generic function works with any provider that supports listing
    and creating SSH keys.

    Args:
        list_keys_fn: Async function that returns list of dicts with
            "id", "name", and optionally "fingerprint" keys.
        create_key_fn: Async function(name, public_key) that creates a key
            and returns a dict with "id" key.
        provider_name: Provider name for logging.

    Returns:
        Key ID on the provider.

    Raises:
        RuntimeError: If no local SSH key found or key creation fails.
    """
    from loguru import logger

    # Get local key
    public_path, public_key = get_local_ssh_key()
    local_fingerprint = compute_fingerprint(public_key)
    key_name = generate_key_name(public_path)

    # Check if key already exists
    existing_keys = await list_keys_fn()
    for key in existing_keys:
        # Try fingerprint first (most reliable)
        if key.get("fingerprint") == local_fingerprint:
            logger.debug(f"{provider_name}: Found existing SSH key by fingerprint")
            return key["id"]
        # Fallback to name
        if key.get("name") == key_name:
            logger.debug(f"{provider_name}: Found existing SSH key by name")
            return key["id"]

    # Create new key
    logger.info(f"{provider_name}: Creating SSH key '{key_name}'")
    try:
        new_key = await create_key_fn(key_name, public_key)
        return new_key["id"]
    except Exception as e:
        # Handle race condition - key might have been created by another process
        if "already" in str(e).lower() or "duplicate" in str(e).lower():
            existing_keys = await list_keys_fn()
            for key in existing_keys:
                if key.get("fingerprint") == local_fingerprint:
                    return key["id"]
                if key.get("name") == key_name:
                    return key["id"]
        raise
