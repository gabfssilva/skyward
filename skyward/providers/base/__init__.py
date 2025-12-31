"""Base infrastructure for cloud providers.

This module provides shared abstractions and utilities for building
cloud providers with minimal code duplication.

Public API:
    # SSH key management
    SSHKeyInfo          - Dataclass for SSH key information
    SSHKeyManager       - Generic SSH key registration manager
    find_local_ssh_key  - Find local SSH public key
    get_private_key_path - Get path to private key
    compute_fingerprint - Compute MD5 fingerprint of SSH key

    # Transport abstraction
    Transport           - Protocol for remote command execution
    SSHTransport        - SSH-based transport implementation

    # Capability protocols
    SpotCapable         - Provider supports spot instances
    VolumeCapable       - Provider supports volumes
    MIGCapable          - Provider supports NVIDIA MIG
    PlacementCapable    - Provider supports placement groups

    # Polling utilities
    InstancePendingError    - Exception for retry logic
    create_instance_poller  - Factory for instance polling
    poll_instances          - Poll multiple instances concurrently
"""

from skyward.providers.base.capabilities import (
    MIGCapable,
    PlacementCapable,
    SpotCapable,
    VolumeCapable,
)
from skyward.providers.base.mixins import (
    InstancePendingError,
    create_instance_poller,
    poll_instances,
)
from skyward.providers.base.ssh_keys import (
    SSH_KEY_PATHS,
    SSHKeyInfo,
    SSHKeyManager,
    compute_fingerprint,
    find_local_ssh_key,
    get_private_key_path,
)
from skyward.providers.base.transport import (
    SSHTransport,
    Transport,
)

__all__ = [
    # SSH keys
    "SSH_KEY_PATHS",
    "SSHKeyInfo",
    "SSHKeyManager",
    "find_local_ssh_key",
    "get_private_key_path",
    "compute_fingerprint",
    # Transport
    "Transport",
    "SSHTransport",
    # Capabilities
    "SpotCapable",
    "VolumeCapable",
    "MIGCapable",
    "PlacementCapable",
    # Mixins
    "InstancePendingError",
    "create_instance_poller",
    "poll_instances",
]
