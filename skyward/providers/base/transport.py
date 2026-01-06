"""Transport abstraction for remote command execution and file transfer.

Provides a Protocol for abstracting SSH vs SSM (and future transports),
plus a concrete SSHTransport implementation for SSH-based providers.
"""

from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_fixed,
)


class _TunnelNotReadyError(Exception):
    """Tunnel not ready - retry."""


def _find_available_port() -> int:
    """Find an available local port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _wait_for_tunnel(port: int, timeout: int = 300) -> None:
    """Wait for tunnel to accept connections."""

    @retry(
        stop=stop_after_delay(timeout),
        wait=wait_fixed(0.5),
        retry=retry_if_exception_type(_TunnelNotReadyError),
        reraise=True,
    )
    def _check() -> None:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except (TimeoutError, ConnectionRefusedError, OSError):
            raise _TunnelNotReadyError() from None

    from tenacity import RetryError
    try:
        _check()
    except RetryError as e:
        raise TimeoutError(f"Tunnel not ready on port {port}") from e


def _create_tunnel(
    cmd: list[str],
    local_port: int,
    timeout: int = 30,
) -> tuple[int, Popen[bytes]]:
    """Create a tunnel process and wait for it to be ready."""
    proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    try:
        _wait_for_tunnel(local_port, timeout=timeout)
        return local_port, proc
    except Exception:
        proc.terminate()
        raise


class Transport(Protocol):
    """Abstraction for remote command execution and file transfer.

    This protocol allows providers to abstract away the underlying
    transport mechanism (SSH, SSM, etc.) while providing a consistent
    interface for command execution and file transfer.
    """

    def run_command(self, command: str, timeout: int = 30) -> str:
        """Execute command on remote instance.

        Args:
            command: Shell command to execute.
            timeout: Command timeout in seconds.

        Returns:
            Command stdout.

        Raises:
            RuntimeError: If command fails.
            TimeoutError: If command times out.
        """
        ...

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload file to remote instance.

        Args:
            local_path: Local file path.
            remote_path: Remote destination path.

        Raises:
            RuntimeError: If upload fails.
        """
        ...

    def create_tunnel(self, remote_port: int) -> tuple[int, Popen[bytes]]:
        """Create port forwarding tunnel to remote instance.

        Args:
            remote_port: Port on remote instance to forward.

        Returns:
            Tuple of (local_port, tunnel_process).

        Raises:
            TimeoutError: If tunnel cannot be established.
        """
        ...


@dataclass(frozen=True, slots=True)
class SSHTransport:
    """SSH-based transport for DigitalOcean, Verda, and similar providers.

    Implements the Transport protocol using standard SSH/SCP commands.

    Attributes:
        host: Remote host IP or hostname.
        username: SSH username (typically "root" or "ubuntu").
        key_path: Path to SSH private key (optional).
        port: SSH port (default 22).
    """

    host: str
    username: str
    key_path: str | None = None
    port: int = 22

    def _ssh_base_args(self) -> list[str]:
        """Build base SSH arguments."""
        args = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=10",
        ]
        if self.key_path:
            args.extend(["-i", self.key_path])
        if self.port != 22:
            args.extend(["-p", str(self.port)])
        return args

    def run_command(self, command: str, timeout: int = 30) -> str:
        """Execute command via SSH."""
        ssh_cmd = ["ssh", *self._ssh_base_args(), f"{self.username}@{self.host}", command]
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SSH command failed (exit {result.returncode}): {result.stderr}"
            )
        return result.stdout

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload file via SCP."""
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
        ]
        if self.key_path:
            scp_cmd.extend(["-i", self.key_path])
        if self.port != 22:
            scp_cmd.extend(["-P", str(self.port)])
        scp_cmd.extend([str(local_path), f"{self.username}@{self.host}:{remote_path}"])

        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"SCP failed: {result.stderr}")

    def create_tunnel(self, remote_port: int) -> tuple[int, Popen[bytes]]:
        """Create SSH port forwarding tunnel."""
        local_port = _find_available_port()
        tunnel_cmd = [
            "ssh",
            *self._ssh_base_args(),
            "-N",
            "-L", f"{local_port}:127.0.0.1:{remote_port}",
            f"{self.username}@{self.host}",
        ]
        return _create_tunnel(tunnel_cmd, local_port)
