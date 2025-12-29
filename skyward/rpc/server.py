"""RPyC server for persistent function execution on compute instances.

This server runs on each instance and listens on localhost:18861.
It is accessed via SSM port forwarding (AWS) or SSH tunnel (other providers).
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from collections.abc import Callable
from typing import Any

import rpyc
from rpyc.utils.server import ThreadedServer

from skyward.serialization import deserialize, serialize

# Default port for RPyC server
RPYC_PORT = 18861


class StreamingStdout:
    """Redirects stdout to a callback function for real-time streaming."""

    def __init__(self, callback: Callable[[str], None], original: Any) -> None:
        self.callback = callback
        self.original = original

    def write(self, data: str) -> int:
        """Pass data directly to callback."""
        if data:
            self.callback(data)
        return len(data)

    def flush(self) -> None:
        """No-op - data is sent immediately."""
        pass


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
            fn = deserialize(fn_bytes)
            args = deserialize(args_bytes)
            kwargs = deserialize(kwargs_bytes)

            if stdout_callback:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StreamingStdout(stdout_callback, old_stdout)
                sys.stderr = StreamingStdout(stdout_callback, old_stderr)
                try:
                    result = fn(*args, **kwargs)
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            else:
                result = fn(*args, **kwargs)

            self._sync_s3_volumes()
            result_bytes: bytes = serialize({"result": result, "error": None})
            return result_bytes

        except Exception:
            error_msg = traceback.format_exc()
            error_bytes: bytes = serialize({"result": None, "error": error_msg})
            return error_bytes

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
    """Start the RPyC server."""
    port = RPYC_PORT
    if len(sys.argv) > 1:
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
