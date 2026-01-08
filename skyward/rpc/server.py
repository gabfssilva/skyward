"""RPyC server for persistent function execution on compute instances.

This server runs on each instance and listens on localhost:18861.
It is accessed via SSM port forwarding (AWS) or SSH tunnel (other providers).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any

import rpyc
from rpyc.utils.server import ThreadedServer

from skyward.core.constants import SKYWARD_DIR
from skyward.utils.serialization import deserialize, serialize

# Default port for RPyC server
RPYC_PORT = 18861

# Events log path (same as bootstrap/metrics)
EVENTS_LOG = f"{SKYWARD_DIR}/events.jsonl"


# =============================================================================
# JSONL Stdout Writer
# =============================================================================


class JsonlStdoutWriter(StringIO):
    """Writes stdout/stderr to events.jsonl for unified streaming.

    Each write appends a JSON log event to the events file.
    Thread-safe via file append mode (atomic writes on Linux).
    """

    def __init__(self, stream: str = "stdout") -> None:
        super().__init__()
        self._stream = stream
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        if not data:
            return 0

        # Split into lines and write each as a JSON event
        for line in data.splitlines():
            if line.strip():
                event = {
                    "type": "log",
                    "content": line,
                    "stream": self._stream,
                }
                with self._lock, open(EVENTS_LOG, "a") as f:
                    f.write(json.dumps(event) + "\n")

        # Also write to parent StringIO for potential capture
        return super().write(data)

    def flush(self) -> None:
        pass  # Flushed per write


class SkywardService(rpyc.Service):
    """RPyC service exposing function execution."""

    ALIASES = ["skyward"]

    def exposed_execute(
        self,
        fn_bytes: bytes,
        args_bytes: bytes,
        kwargs_bytes: bytes,
    ) -> bytes:
        """Execute a serialized function and return serialized result.

        Stdout/stderr are redirected to events.jsonl for unified streaming
        via the same tail -F mechanism used for bootstrap and metrics.

        Args:
            fn_bytes: Cloudpickle-serialized function.
            args_bytes: Cloudpickle-serialized args tuple.
            kwargs_bytes: Cloudpickle-serialized kwargs dict.

        Returns:
            Cloudpickle-serialized dict with 'result' or 'error' key.
        """
        try:
            stdout_writer = JsonlStdoutWriter("stdout")
            stderr_writer = JsonlStdoutWriter("stderr")

            with redirect_stdout(stdout_writer), redirect_stderr(stderr_writer):
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
            "allow_public_attrs": True,  # Allow .get(), .items() on netref dicts
            "sync_request_timeout": 3600,  # 1 hour timeout for long computations
        },
    )
    server.start()


if __name__ == "__main__":
    main()
