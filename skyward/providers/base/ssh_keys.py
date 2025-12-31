"""SSH key management utilities for cloud providers.

Provides SSHKeyInfo dataclass, key discovery utilities, and the generic
SSHKeyManager for deduplicated key registration across providers.
"""

from __future__ import annotations

import base64
import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SSH_KEY_PATHS = (
    "~/.ssh/id_ed25519.pub",
    "~/.ssh/id_rsa.pub",
    "~/.ssh/id_ecdsa.pub",
)


@dataclass(frozen=True)
class SSHKeyInfo:
    """Information about an SSH key."""

    id: str | None
    fingerprint: str
    public_key: str
    name: str


def find_local_ssh_key() -> tuple[Path, str] | None:
    """Find the first available SSH public key.

    Searches through SSH_KEY_PATHS in order, returning the first
    key found with valid content.

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


def get_private_key_path() -> str | None:
    """Get path to SSH private key (without .pub extension).

    Returns:
        Path to private key as string, or None if not found.
    """
    key_info = find_local_ssh_key()
    if key_info is None:
        return None
    pub_path, _ = key_info
    private_path = pub_path.with_suffix("")
    if private_path.exists():
        return str(private_path)
    return None


def compute_fingerprint(public_key: str) -> str:
    """Compute MD5 fingerprint of an SSH public key.

    Args:
        public_key: SSH public key string (e.g., "ssh-ed25519 AAAA... user@host")

    Returns:
        MD5 fingerprint in colon-separated format (e.g., "ab:cd:ef:...")

    Raises:
        ValueError: If public key format is invalid.
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
    return ":".join(md5_hash[i : i + 2] for i in range(0, len(md5_hash), 2))


@dataclass(frozen=True)
class SSHKeyManager[ClientT]:
    """Generic SSH key management for cloud providers.

    This class encapsulates the common pattern of:
    1. Finding local SSH key
    2. Computing fingerprint
    3. Checking if key exists on provider
    4. Creating key if needed

    Type Parameters:
        ClientT: The cloud provider client type (e.g., digitalocean.Client)

    Example:
        # DigitalOcean usage
        do_key_manager = SSHKeyManager(
            list_keys=lambda c: c.ssh_keys.list(),
            create_key=lambda c, name, pub: c.ssh_keys.create(name=name, public_key=pub),
            get_fingerprint=lambda k: k.fingerprint,
            get_id=lambda k: str(k.id),
        )

        key_info = do_key_manager.get_or_create(client)

        # Verda usage
        verda_key_manager = SSHKeyManager(
            list_keys=lambda c: c.ssh_keys.list(region=region),
            create_key=lambda c, name, pub: c.ssh_keys.create(name, pub, region),
            get_fingerprint=lambda k: k["fingerprint"],
            get_id=lambda k: k.get("id"),
        )
    """

    list_keys: Callable[[ClientT], Sequence[Any]]
    create_key: Callable[[ClientT, str, str], Any]
    get_fingerprint: Callable[[Any], str]
    get_id: Callable[[Any], str | None]

    def get_or_create(self, client: ClientT) -> SSHKeyInfo:
        """Find local SSH key and register on provider if needed.

        Args:
            client: Cloud provider client instance.

        Returns:
            SSHKeyInfo with the registered key details.

        Raises:
            RuntimeError: If no local SSH key is found.
        """
        key_info = find_local_ssh_key()
        if key_info is None:
            raise RuntimeError(
                "No SSH key found. Create one with: ssh-keygen -t ed25519\n"
                f"Searched: {', '.join(SSH_KEY_PATHS)}"
            )

        key_path, public_key = key_info
        fingerprint = compute_fingerprint(public_key)
        key_name = f"skyward-{os.environ.get('USER', 'user')}-{key_path.stem}"

        # Check existing keys
        for key in self.list_keys(client):
            if self.get_fingerprint(key) == fingerprint:
                return SSHKeyInfo(
                    id=self.get_id(key),
                    fingerprint=fingerprint,
                    public_key=public_key,
                    name=key_name,
                )

        # Create new key
        created = self.create_key(client, key_name, public_key)
        return SSHKeyInfo(
            id=self.get_id(created),
            fingerprint=fingerprint,
            public_key=public_key,
            name=key_name,
        )
