"""SSH key management for Verda."""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from verda import VerdaClient

# Common SSH key locations in order of preference
SSH_KEY_PATHS = [
    "~/.ssh/id_ed25519.pub",
    "~/.ssh/id_rsa.pub",
    "~/.ssh/id_ecdsa.pub",
]


@dataclass
class SSHKeyInfo:
    """Information about an SSH key."""

    id: str
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
                    return path, content
            except Exception:
                continue

    return None


def compute_fingerprint(public_key: str) -> str:
    """Compute MD5 fingerprint of an SSH public key.

    Args:
        public_key: The public key content (e.g., "ssh-rsa AAAA... user@host")

    Returns:
        Fingerprint in format "aa:bb:cc:dd:..."
    """
    parts = public_key.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid SSH public key format: {public_key[:50]}...")

    key_data = parts[1]

    try:
        decoded = base64.b64decode(key_data)
    except Exception as e:
        raise ValueError(f"Could not decode SSH key: {e}") from e

    md5_hash = hashlib.md5(decoded).hexdigest()
    fingerprint = ":".join(md5_hash[i : i + 2] for i in range(0, len(md5_hash), 2))

    return fingerprint


def get_or_create_ssh_key(client: VerdaClient) -> SSHKeyInfo:
    """Get or create SSH key on Verda.

    Finds the local SSH key, checks if it exists on Verda,
    and creates it if not.

    Args:
        client: Authenticated VerdaClient.

    Returns:
        SSHKeyInfo with id, fingerprint and public key.

    Raises:
        RuntimeError: If no local SSH key found or API error.
    """
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

    # Check if key already exists on Verda
    existing_keys = client.ssh_keys.get()

    for key in existing_keys:
        # Compare by fingerprint or by public key content
        if hasattr(key, "fingerprint") and key.fingerprint == fingerprint:
            return SSHKeyInfo(
                id=key.id,
                fingerprint=fingerprint,
                public_key=public_key,
                name=getattr(key, "name", key_name),
            )
        # Also check by key content if fingerprint not available
        if hasattr(key, "key") and key.key.strip() == public_key.strip():
            return SSHKeyInfo(
                id=key.id,
                fingerprint=fingerprint,
                public_key=public_key,
                name=getattr(key, "name", key_name),
            )

    # Key not found, create it
    try:
        created_key = client.ssh_keys.create(name=key_name, key=public_key)
        return SSHKeyInfo(
            id=created_key.id,
            fingerprint=fingerprint,
            public_key=public_key,
            name=key_name,
        )
    except Exception as e:
        # Key might already exist with different name
        if "already" in str(e).lower():
            # Re-fetch and find it
            existing_keys = client.ssh_keys.get()
            for key in existing_keys:
                if hasattr(key, "key") and key.key.strip() == public_key.strip():
                    return SSHKeyInfo(
                        id=key.id,
                        fingerprint=fingerprint,
                        public_key=public_key,
                        name=getattr(key, "name", key_name),
                    )
        raise
