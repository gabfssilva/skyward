"""Custom exception hierarchy for Skyward.

All skyward-specific exceptions inherit from SkywardError, enabling
users to catch all skyward exceptions with a single except clause.
"""

from __future__ import annotations


class SkywardError(Exception):
    """Base exception for all Skyward errors."""


class NotProvisionedError(SkywardError):
    """Raised when compute is used outside pool context."""

    def __init__(self) -> None:
        super().__init__("Pool not provisioned. Use within pool context.")


class ProvisioningError(SkywardError):
    """Raised when instance provisioning fails."""


class NoMatchingInstanceError(ProvisioningError):
    """Raised when no instance type matches the requirements."""


class ExecutionError(SkywardError):
    """Raised when remote function execution fails."""


class SerializationError(SkywardError):
    """Raised when serialization or deserialization fails."""


class ConfigurationError(SkywardError):
    """Raised for invalid configuration or missing required settings."""


class TimeoutError(SkywardError):  # noqa: A001
    """Raised when an operation exceeds its timeout."""


class ConnectionError(SkywardError):  # noqa: A001
    """Raised when connection to remote instance fails."""


class InstanceTerminatedError(SkywardError):
    """Raised when instance was terminated - do not retry."""

    def __init__(self, instance_id: str, reason: str = "unknown") -> None:
        self.instance_id = instance_id
        self.reason = reason
        super().__init__(f"Instance {instance_id} terminated: {reason}")


class ConnectionLostError(ConnectionError):
    """Raised when connection to instance is lost during execution."""

    def __init__(self, instance_id: str, reason: str = "unknown") -> None:
        self.instance_id = instance_id
        self.reason = reason
        super().__init__(f"Connection lost to {instance_id}: {reason}")
