"""Instance classes representing provisioned compute resources."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skyward.core.events import (
        BootstrapCommand,
        BootstrapConsole,
        BootstrapMetrics,
        BootstrapPhase,
    )
    from skyward.internal.object_pool import ObjectPool
    from skyward.providers.ssh import ChannelStream, SSHConfig, SSHConnection
    from skyward.types.protocols import ComputeSpec, Provider

__all__ = [
    "Instance",
    "ExitedInstance",
]


@dataclass
class Instance:
    """Represents a provisioned compute instance.

    Stores both common fields and provider-specific metadata.
    Metadata is stored as frozenset for immutability.
    """

    id: str
    provider: Provider
    ssh: SSHConfig = field(repr=False)
    spot: bool = False
    private_ip: str = ""
    public_ip: str | None = None
    node: int = 0  # 0 = head node

    # Provider-specific metadata as frozen key-value pairs
    # AWS: {"instance_id": "i-xxx", "region": "us-east-1"}
    # DigitalOcean: {"droplet_id": 12345}
    metadata: frozenset[tuple[str, Any]] = field(default_factory=frozenset)

    # Callback to destroy this instance (injected by provider)
    _destroy_fn: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    # Lock for thread-safe pool initialization (per-instance)
    _pool_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def is_head(self) -> bool:
        """True if this is the head node."""
        return self.node == 0

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Get provider-specific metadata by key."""
        for k, v in self.metadata:
            if k == key:
                return v
        return default

    @property
    def pool(self) -> ObjectPool[SSHConnection]:
        """Thread-safe lazy SSH connection pool."""
        # Fast path: already initialized
        if "_pool" in self.__dict__:
            return self.__dict__["_pool"]

        # Slow path: initialize with lock
        with self._pool_lock:
            # Double-check after acquiring lock
            if "_pool" in self.__dict__:
                return self.__dict__["_pool"]

            from skyward.providers.ssh import SSHPool

            self.__dict__["_pool"] = SSHPool(self.ssh)
            return self.__dict__["_pool"]

    @contextmanager
    def connect(self) -> Iterator[SSHConnection]:
        """Connect with auto-commit on exit."""
        conn = self.pool.acquire()
        try:
            yield conn
            conn.commit()
        finally:
            self.pool.release(conn)

    def run_command(self, command: str, timeout: int = 30) -> str:
        """Execute single command immediately."""
        from loguru import logger

        logger.debug(f"Instance.run_command: acquiring SSH connection for {self.id}")
        with self.pool() as conn:
            logger.debug(f"Instance.run_command: got connection for {self.id}, executing")
            result = conn.exec(command, timeout)
            logger.debug(f"Instance.run_command: command completed for {self.id}")
            return result

    def close(self) -> None:
        """Close instance resources (SSH pool)."""
        if "_pool" in self.__dict__:
            self.__dict__["_pool"].close_all()
            del self.__dict__["_pool"]

    def destroy(self) -> None:
        """Destroy this instance via provider callback."""
        self.close()
        if self._destroy_fn is not None:
            self._destroy_fn()

    def wait_for_ssh(self, timeout: int = 300) -> None:
        """Wait until instance is reachable via SSH."""
        from skyward.providers.common import wait_for_ssh_ready

        host = self.ssh.host or self.public_ip or self.private_ip
        wait_for_ssh_ready(host, self.ssh.port, timeout)

    def stream_events(
        self,
        timeout: float = 600,
    ) -> Iterator[BootstrapConsole | BootstrapPhase | BootstrapCommand | BootstrapMetrics]:
        """Stream events via SSH tail -F (follows rotation).

        Yields all events (bootstrap + metrics) in real-time. Uses tail -F
        which follows file rotation for continuous streaming.

        The stream does NOT automatically terminate on bootstrap completion.
        Caller should handle BootstrapPhase events and break when appropriate.

        Args:
            timeout: Maximum time between events before timeout.

        Yields:
            Bootstrap events (console, phase, command) and metrics events.

        Raises:
            BootstrapError: If bootstrap fails.
            TimeoutError: If no events received within timeout.
        """
        from skyward.providers.common import stream_events

        conn = self.pool.acquire()
        try:
            yield from stream_events(conn.client, timeout=timeout)
        finally:
            self.pool.release(conn)

    def install_skyward(self, compute: ComputeSpec, use_systemd: bool = True) -> None:
        """Install skyward wheel and start RPyC service."""
        from skyward.providers.common import install_wheel_on_instance

        install_wheel_on_instance(self, compute, use_systemd)

    def bootstrap(self, script: str, timeout: int = 600) -> None:
        """Upload and execute bootstrap script on this instance.

        Used for providers (like VastAI) that need two-stage bootstrap due to
        API constraints. The script is uploaded via SFTP and executed via SSH.

        Args:
            script: The bootstrap script content.
            timeout: Maximum time to wait for script execution.
        """
        import tempfile
        from pathlib import Path

        from loguru import logger

        logger.debug(f"[{self.id}] Starting SSH bootstrap...")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            local_script = Path(f.name)

        try:
            with self.connect() as conn:
                # Upload script to /tmp
                remote_script = "/tmp/.skyward-bootstrap.sh"
                conn.upload_file(local_script, remote_script)
                logger.debug(f"[{self.id}] Uploaded bootstrap script")

                # Execute in background - redirect all I/O to prevent blocking
                # The script runs detached and writes to /var/log/skyward-bootstrap.log
                cmd = (
                    f"chmod +x {remote_script} && "
                    f"nohup bash {remote_script} > /var/log/skyward-bootstrap.log 2>&1 < /dev/null &"
                )
                conn.exec_background(cmd)
                logger.debug(f"[{self.id}] Bootstrap script started in background")
        finally:
            local_script.unlink(missing_ok=True)

    @contextmanager
    def open_channel(self, remote_port: int = 18861) -> Iterator[ChannelStream]:
        """Open SSH tunnel channel to remote port.

        Uses Paramiko direct-tcpip channel - no subprocess needed.
        The channel implements RPyC Stream interface for use with rpyc.connect_stream().
        """
        conn = self.pool.acquire()
        try:
            yield conn.open_tunnel(remote_port)
        finally:
            self.pool.release(conn)

    def run_python(self, script: str, timeout: int = 30) -> str:
        return self.run_command(f"/opt/skyward/.venv/bin/python -c '{script}'", timeout)

    def remote[**P, R](self, fn: Callable[P, R]) -> Callable[P, R]:
        """Execute function remotely via shell + cloudpickle.

        Args:
            fn: Function to execute remotely.

        Returns:
            Callable that executes fn on this instance.

        Example:
            def add(a: int, b: int) -> int:
                return a + b

            result = instance.remote(add)(1, 2)  # Returns 3
        """
        import base64
        from functools import wraps

        import cloudpickle

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            fn_b64 = base64.b64encode(cloudpickle.dumps(fn)).decode()
            args_b64 = base64.b64encode(cloudpickle.dumps(args)).decode()
            kwargs_b64 = base64.b64encode(cloudpickle.dumps(kwargs)).decode()

            script = (
                "import sys,base64,cloudpickle;"
                f"fn=cloudpickle.loads(base64.b64decode('{fn_b64}'));"
                f"args=cloudpickle.loads(base64.b64decode('{args_b64}'));"
                f"kwargs=cloudpickle.loads(base64.b64decode('{kwargs_b64}'));"
                "r=fn(*args,**kwargs);"
                "print(base64.b64encode(cloudpickle.dumps(r)).decode())"
            )
            stdout = self.run_command(f'sudo /opt/skyward/.venv/bin/python -c "{script}"')
            return cloudpickle.loads(base64.b64decode(stdout.strip()))  # type: ignore[return-value]

        return wrapper


@dataclass(frozen=True, slots=True)
class ExitedInstance:
    """Represents an instance that has been shut down."""

    instance: Instance
    exit_code: int | None = None
    exit_reason: str = ""  # "normal", "spot_interruption", "timeout", "error"
    error_message: str | None = None
    duration_seconds: float = 0.0
