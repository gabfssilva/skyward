"""Protocol definitions for v2 architecture.

Minimal protocols for transport and execution. Most communication
happens via events, so protocols are focused on low-level operations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# Transport Protocol
# =============================================================================


@runtime_checkable
class Transport(Protocol):
    """Protocol for remote command execution.

    Transport provides a way to execute commands on a remote instance.
    Implementations include SSH (asyncssh) and RPyC.

    Usage:
        async with transport.connect(host, user, key) as conn:
            code, stdout, stderr = await conn.run("ls", "-la")
    """

    async def run(
        self,
        *command: str,
        timeout: float | None = None,
    ) -> tuple[int, str, str]:
        """Execute command and return result.

        Args:
            *command: Command and arguments to execute.
            timeout: Optional timeout in seconds.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        ...

    async def run_stream(
        self,
        *command: str,
        timeout: float | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """Execute command and stream output.

        Args:
            *command: Command and arguments to execute.
            timeout: Optional timeout in seconds.

        Yields:
            Tuples of (stream, line) where stream is "stdout" or "stderr".
        """
        ...

    async def upload(
        self,
        local_path: str,
        remote_path: str,
    ) -> None:
        """Upload file to remote instance.

        Args:
            local_path: Path to local file.
            remote_path: Destination path on remote instance.
        """
        ...

    async def download(
        self,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Download file from remote instance.

        Args:
            remote_path: Path on remote instance.
            local_path: Destination path locally.
        """
        ...


# =============================================================================
# Executor Protocol
# =============================================================================


type Serializable = Any  # cloudpickle can serialize most things


@runtime_checkable
class Executor(Protocol):
    """Protocol for remote function execution.

    Executor provides a way to run Python functions on remote instances.
    The function and arguments are serialized with cloudpickle.

    Usage:
        result = await executor.execute(my_function, arg1, arg2, kwarg=value)
    """

    async def execute[T](
        self,
        fn: Callable[..., T],
        *args: Serializable,
        timeout: float | None = None,
        **kwargs: Serializable,
    ) -> T:
        """Execute function remotely and return result.

        Args:
            fn: Function to execute.
            *args: Positional arguments.
            timeout: Optional timeout in seconds.
            **kwargs: Keyword arguments.

        Returns:
            Result of function execution.

        Raises:
            ExecutionError: If execution fails.
            TimeoutError: If execution times out.
        """
        ...


# =============================================================================
# Connection Factory Protocol
# =============================================================================


@runtime_checkable
class TransportFactory(Protocol):
    """Factory for creating transport connections."""

    async def connect(
        self,
        host: str,
        user: str = "root",
        key_path: str | None = None,
        port: int = 22,
    ) -> Transport:
        """Create transport connection to host.

        Args:
            host: Hostname or IP address.
            user: SSH user.
            key_path: Path to SSH private key.
            port: SSH port.

        Returns:
            Connected Transport instance.
        """
        ...


# =============================================================================
# Health Check Protocol
# =============================================================================


@runtime_checkable
class HealthChecker(Protocol):
    """Protocol for instance health checking."""

    async def check(self, host: str, port: int = 22) -> bool:
        """Check if instance is healthy.

        Args:
            host: Hostname or IP address.
            port: Port to check.

        Returns:
            True if healthy, False otherwise.
        """
        ...


# =============================================================================
# Preemption Checker Protocol
# =============================================================================


@runtime_checkable
class PreemptionChecker(Protocol):
    """Protocol for checking instance preemption status."""

    async def is_preempted(self, instance_id: str) -> tuple[bool, str | None]:
        """Check if instance was preempted.

        Args:
            instance_id: Instance identifier.

        Returns:
            Tuple of (is_preempted, reason) where reason is the preemption
            cause like "spot-interruption", "outbid", or None if not preempted.
        """
        ...


# =============================================================================
