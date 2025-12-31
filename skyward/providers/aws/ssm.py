"""AWS Systems Manager (SSM) transport and command execution."""

from __future__ import annotations

import json
import subprocess
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

from skyward.providers.common import create_tunnel, find_available_port

if TYPE_CHECKING:
    from subprocess import Popen

    from mypy_boto3_ssm import SSMClient


# =============================================================================
# Exceptions
# =============================================================================


class _SSMPendingError(Exception):
    """Command still pending - retry."""


class _SSMNotReadyError(Exception):
    """SSM agent not ready - retry."""


# =============================================================================
# Command Result
# =============================================================================


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Immutable result of SSM command execution."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def raise_on_failure(self, context: str = "") -> None:
        if not self.success:
            msg = f"{context}: {self.stderr}" if context else self.stderr
            raise RuntimeError(msg)


# =============================================================================
# SSM Session
# =============================================================================


class SSMSession:
    """SSM session for command execution on EC2 instances."""

    def __init__(self, region: str) -> None:
        self.region = region

    @cached_property
    def _ssm(self) -> SSMClient:
        import boto3

        return boto3.client("ssm", region_name=self.region)

    def run_command(
        self,
        instance_id: str,
        command: str,
        timeout: int = 300,
    ) -> CommandResult:
        response = self._ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command]},
            TimeoutSeconds=min(timeout, 3600),
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
            raise TimeoutError(f"SSM agent not available on {instance_id} after {timeout}s") from e

    def cleanup(self) -> None:
        pass


# =============================================================================
# SSM Transport (implements Transport protocol)
# =============================================================================


@dataclass(frozen=True, slots=True)
class SSMTransport:
    """AWS Systems Manager transport implementing Transport protocol.

    Unlike SSHTransport, SSMTransport uses AWS SSM for command execution
    and creates tunnels via session-manager-plugin for port forwarding.
    """

    instance_id: str
    region: str

    @cached_property
    def _session(self) -> SSMSession:
        return SSMSession(self.region)

    def run_command(self, command: str, timeout: int = 30) -> str:
        """Execute command via SSM RunCommand."""
        result = self._session.run_command(self.instance_id, command, timeout)
        if not result.success:
            raise RuntimeError(f"Command failed: {result.stderr}")
        return result.stdout

    def create_tunnel(self, remote_port: int) -> tuple[int, Popen[bytes]]:
        """Create SSM port forwarding tunnel."""
        # Check session-manager-plugin is installed
        try:
            result = subprocess.run(
                ["session-manager-plugin", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError("session-manager-plugin returned non-zero")
        except FileNotFoundError:
            raise RuntimeError(
                "session-manager-plugin not found. Please install it:\n"
                "  macOS: brew install session-manager-plugin\n"
                "  Linux: See https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html"
            ) from None

        local_port = find_available_port()
        cmd = [
            "aws",
            "ssm",
            "start-session",
            "--target",
            self.instance_id,
            "--document-name",
            "AWS-StartPortForwardingSession",
            "--parameters",
            json.dumps(
                {
                    "portNumber": [str(remote_port)],
                    "localPortNumber": [str(local_port)],
                }
            ),
            "--region",
            self.region,
        ]
        return create_tunnel(cmd, local_port)
