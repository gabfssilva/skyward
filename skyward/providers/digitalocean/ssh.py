"""SSH key management for DigitalOcean."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("skyward.digitalocean.ssh")

# Common SSH key locations in order of preference
SSH_KEY_PATHS = [
    "~/.ssh/id_ed25519.pub",
    "~/.ssh/id_rsa.pub",
    "~/.ssh/id_ecdsa.pub",
]


@dataclass
class SSHKeyInfo:
    """Information about an SSH key."""

    fingerprint: str
    public_key: str
    name: str


def find_local_ssh_key() -> tuple[Path, str] | None:
    """Find the first available SSH public key.

    Returns:
        Tuple of (path, public_key_content) or None if not found.
    """
    for key_path in SSH_KEY_PATHS:
        path = Path(key_path).expanduser()
        if path.exists():
            try:
                content = path.read_text().strip()
                if content:
                    logger.info(f"Found SSH key at {path}")
                    return path, content
            except Exception as e:
                logger.info(f"Could not read {path}: {e}")
                continue

    return None


def compute_fingerprint(public_key: str) -> str:
    """Compute MD5 fingerprint of an SSH public key.

    This matches the format used by DigitalOcean (colon-separated MD5).

    Args:
        public_key: The public key content (e.g., "ssh-rsa AAAA... user@host")

    Returns:
        Fingerprint in format "aa:bb:cc:dd:..."
    """
    # Extract the base64-encoded key data (second field)
    parts = public_key.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid SSH public key format: {public_key[:50]}...")

    key_data = parts[1]

    # Decode and compute MD5 hash
    try:
        decoded = base64.b64decode(key_data)
    except Exception as e:
        raise ValueError(f"Could not decode SSH key: {e}") from e

    md5_hash = hashlib.md5(decoded).hexdigest()

    # Format as colon-separated pairs
    fingerprint = ":".join(md5_hash[i : i + 2] for i in range(0, len(md5_hash), 2))

    return fingerprint


def get_or_create_ssh_key(api_token: str) -> SSHKeyInfo:
    """Get or create SSH key on DigitalOcean.

    Finds the local SSH key, checks if it exists on DigitalOcean,
    and creates it if not.

    Args:
        api_token: DigitalOcean API token.

    Returns:
        SSHKeyInfo with fingerprint and public key.

    Raises:
        RuntimeError: If no local SSH key found or API error.
    """
    from skyward.providers.digitalocean.client import get_client

    # Find local key
    key_info = find_local_ssh_key()
    if key_info is None:
        raise RuntimeError(
            "No SSH key found. Please create one with:\n"
            "  ssh-keygen -t ed25519 -C 'your@email.com'\n"
            f"Searched locations: {', '.join(SSH_KEY_PATHS)}"
        )

    key_path, public_key = key_info
    fingerprint = compute_fingerprint(public_key)
    key_name = f"skyward-{os.environ.get('USER', 'user')}-{key_path.stem}"

    client = get_client(api_token)

    # Check if key already exists on DO
    resp = client.ssh_keys.list()
    existing_keys = resp.get("ssh_keys", [])

    for key in existing_keys:
        if key.get("fingerprint") == fingerprint:
            logger.info(f"SSH key already registered on DigitalOcean: {fingerprint}")
            return SSHKeyInfo(
                fingerprint=fingerprint,
                public_key=public_key,
                name=key.get("name", key_name),
            )

    # Key not found, create it
    logger.info(f"Registering SSH key on DigitalOcean: {key_name}")
    try:
        resp = client.ssh_keys.create(
            body={
                "name": key_name,
                "public_key": public_key,
            }
        )
        created_key = resp.get("ssh_key", {})
        logger.info(f"SSH key registered: {created_key.get('fingerprint')}")
        return SSHKeyInfo(
            fingerprint=created_key.get("fingerprint", fingerprint),
            public_key=public_key,
            name=created_key.get("name", key_name),
        )
    except Exception as e:
        # Key might already exist with different name
        if "already been taken" in str(e).lower():
            logger.info(f"SSH key already exists (different name): {fingerprint}")
            return SSHKeyInfo(
                fingerprint=fingerprint,
                public_key=public_key,
                name=key_name,
            )
        raise
