"""RPyC server for persistent function execution on compute instances.

This server runs on each instance and listens on localhost:18861.
It is accessed via SSM port forwarding (AWS) or SSH tunnel (other providers).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING

import rpyc
from rpyc.utils.server import ThreadedServer

from skyward.output import redirect_output
from skyward.serialization import deserialize, serialize

if TYPE_CHECKING:
    from typing import TextIO

# Default port for RPyC server
RPYC_PORT = 18861


# =============================================================================
# Thread-Local Stdout Dispatcher
# =============================================================================


class ThreadLocalStdout:
    """Stdout dispatcher that routes writes to thread-local callbacks.

    Installed once at server startup. Each thread registers its own
    callback for the duration of function execution. This avoids race
    conditions when multiple threads execute concurrently.
    """

    def __init__(self, original: TextIO) -> None:
        self.original = original
        self._local = threading.local()

    def register(self, callback: Callable[[str], None]) -> None:
        """Register callback for current thread."""
        self._local.callback = callback

    def unregister(self) -> None:
        """Unregister callback for current thread."""
        self._local.callback = None

    def write(self, data: str) -> int:
        """Write to thread's callback if registered, else discard."""
        if data:
            callback = getattr(self._local, "callback", None)
            if callback is not None:
                try:
                    callback(data)
                except Exception:
                    pass  # Don't fail execution for callback errors
        return len(data)

    def flush(self) -> None:
        """No-op - data is sent immediately."""
        pass

    def isatty(self) -> bool:
        """Not a TTY."""
        return False

    @property
    def encoding(self) -> str:
        """Forward encoding from original stream."""
        return self.original.encoding

    def fileno(self) -> int:
        """Forward fileno from original stream."""
        return self.original.fileno()


# Global dispatchers (installed once at server startup)
_stdout_dispatcher: ThreadLocalStdout | None = None
_stderr_dispatcher: ThreadLocalStdout | None = None


def _install_dispatchers() -> None:
    """Install thread-local stdout/stderr dispatchers (once)."""
    global _stdout_dispatcher, _stderr_dispatcher

    if _stdout_dispatcher is None:
        _stdout_dispatcher = ThreadLocalStdout(sys.stdout)
        sys.stdout = _stdout_dispatcher  # type: ignore[assignment]

    if _stderr_dispatcher is None:
        _stderr_dispatcher = ThreadLocalStdout(sys.stderr)
        sys.stderr = _stderr_dispatcher  # type: ignore[assignment]


class SkywardService(rpyc.Service):
    """RPyC service exposing function execution."""

    ALIASES = ["skyward"]

    def on_connect(self, conn: rpyc.Connection) -> None:
        """Called when a client connects."""
        pass

    def on_disconnect(self, conn: rpyc.Connection) -> None:
        """Called when a client disconnects."""
        pass

    def exposed_execute(
        self,
        fn_bytes: bytes,
        args_bytes: bytes,
        kwargs_bytes: bytes,
        stdout_callback: Callable[[str], None] | None = None,
    ) -> bytes:
        """Execute a serialized function and return serialized result.

        Args:
            fn_bytes: Cloudpickle-serialized function.
            args_bytes: Cloudpickle-serialized args tuple.
            kwargs_bytes: Cloudpickle-serialized kwargs dict.
            stdout_callback: Optional callback for streaming stdout to client.

        Returns:
            Cloudpickle-serialized dict with 'result' or 'error' key.
        """
        try:
            with redirect_output(stdout_callback):
                fn = deserialize(fn_bytes)
                args = deserialize(args_bytes)
                kwargs = deserialize(kwargs_bytes)
                result = fn(*args, **kwargs)
                self._sync_s3_volumes()
                return serialize({"result": result, "error": None})
        except Exception:
            error_msg = traceback.format_exc()
            return serialize({"result": None, "error": error_msg})

    def _sync_s3_volumes(self) -> None:
        """Unmount all S3 volumes to force pending uploads to complete.

        Mountpoint for S3 has async upload behavior - close() may return
        before data is uploaded. The only guaranteed way to sync is umount.
        """
        try:
            subprocess.run(
                ["umount", "-a", "-t", "fuse.mount-s3"],
                check=False,  # Don't fail if no mounts exist
                capture_output=True,
                text=True,
            )
        except Exception:
            pass  # Best effort - don't fail execution for unmount issues

    def exposed_ping(self) -> str:
        """Health check endpoint.

        Returns:
            "pong" if server is healthy.
        """
        return "pong"

    def exposed_setup_cluster(
        self,
        pool_info_json: str,
        env_vars_bytes: bytes,
    ) -> str:
        """Setup cluster environment before function execution.

        This method sets the COMPUTE_POOL environment variable and any
        additional environment variables needed for distributed execution.

        Args:
            pool_info_json: JSON string for COMPUTE_POOL env var.
            env_vars_bytes: Cloudpickle-serialized dict of env vars.

        Returns:
            "ok" if setup successful.
        """
        import os

        os.environ["COMPUTE_POOL"] = pool_info_json

        env_vars: dict[str, str] = deserialize(env_vars_bytes)
        for key, value in env_vars.items():
            os.environ[key] = value

        return "ok"


def main() -> None:
    """Start the RPyC server.

    Port can be configured via:
    1. SKYWARD_WORKER_PORT environment variable (for worker isolation)
    2. Command line argument (legacy)
    3. Default RPYC_PORT (18861)
    """
    import os

    # Install thread-local stdout/stderr dispatchers before starting server
    _install_dispatchers()

    # Priority: env var > command line > default
    port = int(os.environ.get("SKYWARD_WORKER_PORT", "0")) or RPYC_PORT
    if port == RPYC_PORT and len(sys.argv) > 1:
        port = int(sys.argv[1])

    server = ThreadedServer(
        SkywardService,
        hostname="127.0.0.1",  # Only localhost - accessed via port forwarding
        port=port,
        protocol_config={
            "allow_pickle": True,
            "sync_request_timeout": 3600,  # 1 hour timeout for long computations
        },
    )
    server.start()


if __name__ == "__main__":
    main()
