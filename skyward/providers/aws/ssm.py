"""SSM-based connectivity for EC2 instances.

Uses AWS Systems Manager for:
1. Command execution via SendCommand (replaces SSH)
2. Port forwarding via StartSession (for RPyC tunnels)

No SSH keys, no paramiko, no EIC endpoints needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_fixed,
)

if TYPE_CHECKING:
    from mypy_boto3_ssm import SSMClient


class _SSMPendingError(Exception):
    """Command still pending - retry."""


class _SSMNotReadyError(Exception):
    """SSM agent not ready - retry."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Immutable result of SSM command execution."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Return True if command executed successfully."""
        return self.exit_code == 0

    def raise_on_failure(self, context: str = "") -> None:
        """Raise RuntimeError if command failed."""
        if not self.success:
            msg = f"{context}: {self.stderr}" if context else self.stderr
            raise RuntimeError(msg)


class SSMSession:
    """SSM session for command execution on EC2 instances.

    Uses SSM SendCommand for running commands (replaces SSH).
    Port forwarding is handled separately via subprocess calls to
    `aws ssm start-session`.
    """

    def __init__(self, region: str) -> None:
        """Initialize SSM session.

        Args:
            region: AWS region.
        """
        self.region = region

    @cached_property
    def _ssm(self) -> SSMClient:
        """Lazy SSM client."""
        import boto3

        return boto3.client("ssm", region_name=self.region)

    def run_command(
        self,
        instance_id: str,
        command: str,
        timeout: int = 300,
    ) -> CommandResult:
        """Execute command via SSM SendCommand.

        Args:
            instance_id: EC2 instance ID.
            command: Shell command to execute.
            timeout: Command timeout in seconds.

        Returns:
            CommandResult with exit_code, stdout, stderr.

        Raises:
            TimeoutError: If command doesn't complete within timeout.
            RuntimeError: If command invocation fails.
        """
        # Send command
        response = self._ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command]},
            TimeoutSeconds=min(timeout, 3600),  # SSM max is 3600
        )

        command_id = response["Command"]["CommandId"]

        @retry(
            stop=stop_after_delay(timeout + 30),
            wait=wait_fixed(2),
            retry=retry_if_exception_type(_SSMPendingError),
            reraise=True,
        )
        def _poll() -> CommandResult:
            try:
                result = self._ssm.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=instance_id,
                )
            except self._ssm.exceptions.InvocationDoesNotExist:
                raise _SSMPendingError()

            status = result["Status"]

            if status == "Success":
                return CommandResult(
                    exit_code=0,
                    stdout=result.get("StandardOutputContent", ""),
                    stderr=result.get("StandardErrorContent", ""),
                )
            elif status in ("Failed", "TimedOut", "Cancelled"):
                stderr = result.get("StandardErrorContent", "") or f"Command {status}"
                return CommandResult(
                    exit_code=1,
                    stdout=result.get("StandardOutputContent", ""),
                    stderr=stderr,
                )
            else:
                # Pending, InProgress, Delayed - keep polling
                raise _SSMPendingError()

        try:
            return _poll()
        except RetryError as e:
            raise TimeoutError(f"Command timed out after {timeout}s") from e

    def wait_for_ssm_agent(
        self,
        instance_id: str,
        timeout: int = 600,
    ) -> None:
        """Wait for SSM agent to become available on instance.

        Args:
            instance_id: EC2 instance ID.
            timeout: Maximum time to wait in seconds.

        Raises:
            TimeoutError: If SSM agent doesn't become available within timeout.
        """

        @retry(
            stop=stop_after_delay(timeout),
            wait=wait_fixed(5),
            retry=retry_if_exception_type(_SSMNotReadyError),
            reraise=True,
        )
        def _check() -> None:
            try:
                result = self.run_command(instance_id, "echo ok", timeout=30)
                if result.success:
                    return
            except Exception:
                pass
            raise _SSMNotReadyError()

        try:
            _check()
        except RetryError as e:
            raise TimeoutError(
                f"SSM agent not available on {instance_id} after {timeout}s"
            ) from e

    def cleanup(self) -> None:
        """Cleanup resources (no-op for SSM, kept for interface compatibility)."""
        pass
