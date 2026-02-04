"""Ray Jobs API executor for remote function execution.

Provides async execution of functions on remote Ray cluster via Ray Jobs API.
Uses SSH tunnel (local port forwarding) to access Dashboard/Jobs API on head node.

This replaces Ray Client which had stability issues over SSH tunnels (gRPC timeouts).
Ray Jobs API uses HTTP which is more robust for tunneled connections.
"""

from __future__ import annotations

import asyncio
import base64
import socket
import tempfile
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import asyncssh
from loguru import logger

from .transport.ssh import SSHTransport
from .utils.serialization import deserialize, serialize

# Ray Dashboard/Jobs API port (head node)
RAY_DASHBOARD_PORT = 8265

# Threshold for using file vs env var for payload
PAYLOAD_SIZE_THRESHOLD = 50 * 1024  # 50KB


def _find_free_port() -> int:
    """Find an available local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Executor:
    """Ray Jobs API executor for remote function execution.

    Uses SSH tunnel (local port forwarding) to access Ray Dashboard/Jobs API.
    The Ray head node runs Dashboard on port 8265, and we forward
    a local port to it via SSH.

    Jobs are submitted with a runner script (job_runner.py) that:
    1. Deserializes the function + args from env var or file
    2. Executes with optional node placement via Ray resources
    3. Writes result to file for retrieval

    Example:
        executor = Executor(
            head_ip="10.0.0.1",
            user="ubuntu",
            key_path="~/.ssh/id_rsa",
            num_nodes=4,
        )

        await executor.connect()
        result = await executor.execute(fn, *args, node_id=0)
        results = await executor.broadcast(fn, *args)
        await executor.disconnect()
    """

    head_ip: str
    user: str
    key_path: str
    num_nodes: int = 1
    ssh_port: int = 22
    remote_port: int = RAY_DASHBOARD_PORT
    connect_timeout: float = 120.0
    job_timeout: float = 600.0  # Max time to wait for a job
    env_vars: dict[str, str] = field(default_factory=dict)  # Env vars for all jobs
    pool_infos: list[str] = field(default_factory=list)  # COMPUTE_POOL JSON per node

    _ssh_conn: asyncssh.SSHClientConnection | None = field(default=None, repr=False)
    _local_port: int = field(default=0, repr=False)
    _listener: asyncssh.SSHListener | None = field(default=None, repr=False)
    _connected: bool = field(default=False, repr=False)
    _transport: SSHTransport | None = field(default=None, repr=False)

    async def connect(self) -> None:
        """Establish SSH tunnel to Ray Dashboard/Jobs API."""
        if self._connected:
            return

        # Connect SSH with keepalive to prevent tunnel drops
        logger.debug(f"Connecting SSH to {self.head_ip}:{self.ssh_port}")
        self._ssh_conn = await asyncssh.connect(
            self.head_ip,
            port=self.ssh_port,
            username=self.user,
            client_keys=[self.key_path],
            known_hosts=None,
            connect_timeout=self.connect_timeout,
            keepalive_interval=15,
            keepalive_count_max=4,
        )

        # Create local port forwarding to Dashboard
        self._local_port = _find_free_port()
        self._listener = await self._ssh_conn.forward_local_port(
            "127.0.0.1",  # Explicit IPv4 localhost binding
            self._local_port,
            "127.0.0.1",  # Forward to localhost on remote
            self.remote_port,
        )

        # Brief delay for tunnel stabilization
        await asyncio.sleep(0.5)

        logger.debug(
            f"SSH tunnel: localhost:{self._local_port} -> "
            f"{self.head_ip}:localhost:{self.remote_port}"
        )

        # Create transport for file transfers
        self._transport = SSHTransport(
            host=self.head_ip,
            user=self.user,
            key_path=self.key_path,
            port=self.ssh_port,
        )
        await self._transport.connect()

        # Verify Jobs API is accessible
        await self._verify_connection()

        self._connected = True
        logger.info(f"Connected to Ray cluster at {self.head_ip} (Jobs API)")

    async def _verify_connection(self) -> None:
        """Verify Jobs API is accessible via the tunnel.

        Retries for up to 30 seconds since the dashboard may take
        time to become ready after Ray starts.
        """
        from ray.job_submission import JobSubmissionClient

        address = f"http://localhost:{self._local_port}"
        logger.debug(f"Verifying Jobs API at {address}...")

        max_attempts = 15
        delay = 2.0

        for attempt in range(max_attempts):
            try:
                # JobSubmissionClient operations are sync, run in executor
                def check_connection() -> None:
                    client = JobSubmissionClient(address)
                    # List jobs to verify connection (empty list is fine)
                    client.list_jobs()

                await asyncio.get_event_loop().run_in_executor(None, check_connection)
                logger.debug("Jobs API connection verified")
                return
            except (ConnectionError, OSError) as e:
                if attempt < max_attempts - 1:
                    logger.debug(f"Jobs API not ready (attempt {attempt + 1}/{max_attempts}): {e}")
                    await asyncio.sleep(delay)
                else:
                    raise

    async def disconnect(self) -> None:
        """Disconnect from Ray cluster and close SSH tunnel."""
        logger.debug(f"Disconnecting from {self.head_ip}")

        self._connected = False

        if self._transport is not None:
            with suppress(Exception):
                await self._transport.close()
            self._transport = None

        if self._listener is not None:
            with suppress(Exception):
                self._listener.close()
            self._listener = None

        if self._ssh_conn is not None:
            with suppress(Exception):
                self._ssh_conn.close()
            with suppress(Exception):
                await asyncio.wait_for(self._ssh_conn.wait_closed(), timeout=5.0)
            self._ssh_conn = None

        logger.debug(f"Disconnected from {self.head_ip}")

    async def __aenter__(self) -> Executor:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Whether connection is established."""
        return self._connected

    async def execute[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        node_id: int | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute function on cluster via Ray Jobs API.

        Args:
            fn: Function to execute remotely.
            *args: Positional arguments.
            node_id: Specific node to run on (0 to num_nodes-1), or None for any.
            **kwargs: Keyword arguments.

        Returns:
            Result of function execution.

        Raises:
            RuntimeError: If not connected or job fails.
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        fn_name = getattr(fn, "__name__", str(fn))
        job_id = f"skyward-{uuid.uuid4().hex[:8]}"
        logger.debug(f"execute({fn_name}) submitting job {job_id}, node_id={node_id}")

        # Build payload
        payload = {
            "fn": fn,
            "args": args,
            "kwargs": kwargs,
            "mode": "single",
            "node_id": node_id,
            "num_nodes": self.num_nodes,
            "pool_infos": self.pool_infos,
        }

        # Submit and wait for result
        result = await self._submit_and_wait(job_id, payload)
        logger.debug(f"execute({fn_name}) completed")
        return result

    async def broadcast[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> list[T]:
        """Execute function on all nodes via Ray Jobs API.

        Args:
            fn: Function to execute on each node.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            List of results from each node, ordered by node_id.
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        fn_name = getattr(fn, "__name__", str(fn))
        job_id = f"skyward-{uuid.uuid4().hex[:8]}"
        logger.debug(f"broadcast({fn_name}) submitting job {job_id}")

        # Build payload
        payload = {
            "fn": fn,
            "args": args,
            "kwargs": kwargs,
            "mode": "broadcast",
            "num_nodes": self.num_nodes,
            "pool_infos": self.pool_infos,
        }

        # Submit and wait for result
        result = await self._submit_and_wait(job_id, payload)
        logger.debug(f"broadcast({fn_name}) completed")
        return result

    async def _submit_and_wait[T](self, job_id: str, payload: dict[str, Any]) -> T:
        """Submit job and wait for result.

        Handles payload transfer (env var or file), job submission,
        status polling, and result retrieval.
        """
        from ray.job_submission import JobStatus, JobSubmissionClient

        assert self._transport is not None

        # Serialize payload
        payload_bytes = serialize(payload)
        payload_size = len(payload_bytes)

        # Determine transfer method
        remote_payload_file = f"/tmp/skyward_payload_{job_id}.pkl"
        remote_result_file = f"/tmp/skyward_result_{job_id}.pkl"
        use_file = payload_size >= PAYLOAD_SIZE_THRESHOLD

        # Set up environment (include user env vars like KERAS_BACKEND)
        env_vars = {**self.env_vars, "SKYWARD_RESULT_FILE": remote_result_file}

        # Use the Python from the skyward venv
        python_bin = "/opt/skyward/.venv/bin/python"

        if use_file:
            # Upload payload file
            logger.debug(f"Uploading payload ({payload_size} bytes) to {remote_payload_file}")
            await self._transport.write_bytes(remote_payload_file, payload_bytes)
            entrypoint = f"{python_bin} -m skyward.job_runner {remote_payload_file}"
        else:
            # Use env var for small payloads
            payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
            env_vars["SKYWARD_PAYLOAD"] = payload_b64
            entrypoint = f"{python_bin} -m skyward.job_runner"

        # Submit job
        address = f"http://localhost:{self._local_port}"

        def submit_job() -> str:
            client = JobSubmissionClient(address)
            return client.submit_job(
                entrypoint=entrypoint,
                submission_id=job_id,
                entrypoint_num_cpus=0,  # Don't reserve CPUs for entrypoint
                runtime_env={"env_vars": env_vars},
            )

        logger.debug(f"Submitting job {job_id}...")
        submitted_id = await asyncio.get_event_loop().run_in_executor(None, submit_job)
        logger.debug(f"Job submitted: {submitted_id}")

        # Poll for completion
        terminal_states = {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.STOPPED,
        }

        def get_status() -> JobStatus:
            client = JobSubmissionClient(address)
            return client.get_job_status(job_id)

        def get_logs() -> str:
            client = JobSubmissionClient(address)
            return client.get_job_logs(job_id)

        start_time = asyncio.get_event_loop().time()
        last_status = None
        last_log_len = 0

        while True:
            status = await asyncio.get_event_loop().run_in_executor(None, get_status)

            if status != last_status:
                logger.debug(f"Job {job_id} status: {status}")
                last_status = status

            # Stream logs incrementally
            logs = await asyncio.get_event_loop().run_in_executor(None, get_logs)
            if logs and len(logs) > last_log_len:
                new_logs = logs[last_log_len:]
                for line in new_logs.rstrip("\n").split("\n"):
                    if line:
                        logger.info(f"[job] {line}")
                last_log_len = len(logs)

            if status in terminal_states:
                break

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.job_timeout:
                # Try to stop the job
                with suppress(Exception):
                    def stop_job() -> None:
                        client = JobSubmissionClient(address)
                        client.stop_job(job_id)
                    await asyncio.get_event_loop().run_in_executor(None, stop_job)
                raise TimeoutError(f"Job {job_id} timed out after {self.job_timeout}s")

            await asyncio.sleep(0.5)

        # Check final status
        if status == JobStatus.FAILED:
            logs = await asyncio.get_event_loop().run_in_executor(None, get_logs)
            raise RuntimeError(f"Job {job_id} failed:\n{logs}")

        if status == JobStatus.STOPPED:
            raise RuntimeError(f"Job {job_id} was stopped")

        # Download result
        logger.debug(f"Downloading result from {remote_result_file}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            local_result_file = tmp.name

        try:
            await self._transport.download(remote_result_file, local_result_file)

            with open(local_result_file, "rb") as f:
                result_bytes = f.read()

            result = deserialize(result_bytes)

            # Check for error in result
            if isinstance(result, dict) and "error" in result and "traceback" in result:
                raise RuntimeError(f"Job execution error: {result['error']}\n{result['traceback']}")

            return result
        finally:
            # Cleanup local temp file
            import os
            with suppress(Exception):
                os.unlink(local_result_file)

            # Cleanup remote files
            with suppress(Exception):
                await self._transport.run("rm", "-f", remote_result_file, remote_payload_file)

    async def setup_cluster(
        self,
        pool_info_json: str,
        env_vars: dict[str, str],
    ) -> None:
        """Setup cluster environment on all nodes.

        Sets COMPUTE_POOL and additional environment variables
        on each node for distributed training coordination.

        Args:
            pool_info_json: JSON string for COMPUTE_POOL env var.
            env_vars: Additional environment variables.
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        def setup_env(pool_json: str, extra_vars: dict[str, str]) -> str:
            import os
            os.environ["COMPUTE_POOL"] = pool_json
            for key, value in extra_vars.items():
                os.environ[key] = value
            return "ok"

        # Setup on all nodes
        await self.broadcast(setup_env, pool_info_json, env_vars)


__all__ = [
    "Executor",
    "RAY_DASHBOARD_PORT",
]
