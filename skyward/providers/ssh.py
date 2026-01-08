"""SSH connection pooling."""

from dataclasses import dataclass
from pathlib import Path

import paramiko
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
)

from skyward.internal.object_pool import ObjectPool


class ChannelStream:
    """Adapta Paramiko Channel para interface RPyC Stream."""

    __slots__ = ("_channel",)

    # RPyC stream protocol requires this constant
    MAX_IO_CHUNK = 8 * 1024 * 1024  # 8MB, same as rpyc.core.stream.SocketStream

    def __init__(self, channel: paramiko.Channel) -> None:
        self._channel = channel

    def read(self, count: int) -> bytes:
        """Read exactly count bytes (RPyC requirement)."""
        data = b""
        while len(data) < count:
            chunk = self._channel.recv(count - len(data))
            if not chunk:
                raise EOFError("Channel closed")
            data += chunk
        return data

    def write(self, data: bytes) -> None:
        """Write all data (RPyC requirement)."""
        while data:
            sent = self._channel.send(data)
            if sent == 0:
                raise EOFError("Channel closed")
            data = data[sent:]

    def close(self) -> None:
        self._channel.close()

    @property
    def closed(self) -> bool:
        return self._channel.closed

    def fileno(self) -> int:
        return self._channel.fileno()

    def poll(self, timeout: float) -> bool:
        return self._channel.recv_ready()


@dataclass(frozen=True, slots=True)
class SSHConfig:
    """SSH connection configuration."""

    host: str
    username: str
    port: int = 22
    key_path: str | None = None


class SSHConnection:
    """SSH connection with batched operations."""

    __slots__ = ("_client", "_uploads", "_commands")

    def __init__(self, config: SSHConfig) -> None:
        logger.debug(f"SSH: connecting to {config.host}:{config.port} ({config.username})")
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs: dict = {"hostname": config.host, "username": config.username, "port": config.port}
        if config.key_path:
            kwargs["key_filename"] = config.key_path
        self._client.connect(**kwargs)
        logger.debug(f"SSH: connected to {config.host}")
        self._uploads: list[tuple[Path, str]] = []
        self._commands: list[str] = []

    def upload(self, local: Path, remote: str) -> "SSHConnection":
        """Queue file upload."""
        self._uploads.append((local, remote))
        return self

    def run(self, *commands: str) -> "SSHConnection":
        """Queue commands for batch execution."""
        self._commands.extend(commands)
        return self

    def exec(self, command: str, timeout: int = 30) -> str:
        """Execute immediately, return stdout."""
        cmd_preview = command[:80] + "..." if len(command) > 80 else command
        logger.debug(f"SSHConnection.exec: {cmd_preview}")
        _, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        code = stdout.channel.recv_exit_status()
        logger.debug(f"SSHConnection.exec: exit_code={code}")
        if code != 0:
            raise RuntimeError(f"Command failed ({code}): {stderr.read().decode()}")
        return stdout.read().decode()

    def exec_background(self, command: str) -> None:
        """Execute command in background without waiting for completion.

        Use this for long-running commands that should run detached.
        The command should redirect its own I/O (e.g., using nohup).
        """
        cmd_preview = command[:80] + "..." if len(command) > 80 else command
        logger.debug(f"SSHConnection.exec_background: {cmd_preview}")
        # Open channel, send command, close immediately
        transport = self._client.get_transport()
        if transport is None:
            raise RuntimeError("SSH transport not available")
        channel = transport.open_session()
        channel.exec_command(command)
        # Don't wait for exit status - just close the channel
        channel.close()
        logger.debug("SSHConnection.exec_background: command sent")

    # Transport protocol compatibility
    def run_command(self, command: str, timeout: int = 30) -> str:
        """Execute command (Transport protocol)."""
        return self.exec(command, timeout)

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload file (Transport protocol)."""
        sftp = self._client.open_sftp()
        try:
            sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()

    def commit(self) -> str | None:
        """Execute pending uploads and commands."""
        if self._uploads:
            sftp = self._client.open_sftp()
            try:
                for local, remote in self._uploads:
                    sftp.put(str(local), remote)
            finally:
                sftp.close()
            self._uploads.clear()

        result = None
        if self._commands:
            result = self.exec(" && ".join(self._commands))
            self._commands.clear()
        return result

    def is_alive(self) -> bool:
        transport = self._client.get_transport()
        return transport is not None and transport.is_active()

    @property
    def client(self) -> paramiko.SSHClient:
        """Expose underlying Paramiko client for advanced operations."""
        return self._client

    def open_tunnel(self, remote_port: int) -> ChannelStream:
        """Open direct-tcpip channel for port forwarding."""
        logger.debug(f"SSH.open_tunnel: opening channel to port {remote_port}")
        transport = self._client.get_transport()
        if transport is None:
            raise RuntimeError("SSH not connected")

        channel = transport.open_channel(
            "direct-tcpip",
            dest_addr=("127.0.0.1", remote_port),
            src_addr=("127.0.0.1", 0),
        )
        logger.debug(f"SSH.open_tunnel: channel opened to port {remote_port}")
        return ChannelStream(channel)

    def close(self) -> None:
        self._client.close()


def SSHPool(config: SSHConfig, max_size: int = 4) -> ObjectPool[SSHConnection]:
    """Create lazy SSH connection pool with auth retry.

    Pool initializes instantly without blocking. Connections are created
    on-demand up to max_size, with one pre-warmed in background.
    Retries on AuthenticationException (key may not be injected yet).
    """

    @retry(
        stop=stop_after_delay(60),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(paramiko.ssh_exception.AuthenticationException),
    )
    def create_with_retry(_: int) -> SSHConnection:
        return SSHConnection(config)

    return ObjectPool(
        create=create_with_retry,
        close=SSHConnection.close,
        check=SSHConnection.is_alive,
        max_size=max_size,
        min_size=1,  # Pre-warm 1 connection in background
        health_interval=30.0,
    )
