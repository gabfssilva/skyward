"""Bootstrap operations for Ray cluster setup.

Operations for installing Ray and starting head/worker nodes.
Replaces RPyC-based server setup with Ray cluster.
"""

from __future__ import annotations

import json

from ..constants import VENV_DIR
from .compose import Op
from .ops import pip, shell, wait_for_port


def ray_install(version: str = "2.53.0") -> Op:
    """Install Ray with default extras (includes Dashboard and Jobs API).

    Args:
        version: Ray version to install.

    Example:
        >>> ray_install("2.53.0")()
        'uv pip install ray[default]==2.53.0'
    """
    # Install ray[default] for Dashboard and Jobs API support
    return pip(f"ray[default]=={version}")


def ray_service(
    node_id: int,
    head_ip: str | None,
    num_gpus: int = 1,
    ray_port: int = 6379,
) -> Op:
    """Generate nohup command to run Ray as a background service.

    Uses nohup + background instead of systemd for compatibility
    with Docker containers that don't have systemd.

    Args:
        node_id: Node ID (0 = head, >0 = worker).
        head_ip: IP of head node (required for workers).
        num_gpus: Number of GPUs.
        ray_port: Ray GCS port (default 6379).

    Example:
        >>> ray_service(0, None, num_gpus=1)()
        'nohup ray start --head ... > /var/log/ray.log 2>&1 &'
    """
    resources = {f"node_{node_id}": 1.0}
    ray_bin = f"{VENV_DIR}/bin/ray"

    def generate() -> str:
        if node_id == 0:
            # Head node
            # Note: We don't start Ray Client server (--ray-client-server-port)
            # because we use Ray Jobs API instead, which only needs the Dashboard
            cmd = [
                ray_bin,
                "start",
                "--head",
                f"--port={ray_port}",
                "--dashboard-port=8265",
                "--dashboard-host=0.0.0.0",  # Allow remote access via SSH tunnel
                f"--num-gpus={num_gpus}",
                f"--resources='{json.dumps(resources)}'",
            ]
        else:
            # Worker node
            if head_ip is None:
                raise ValueError("worker nodes require head_ip")
            cmd = [
                ray_bin,
                "start",
                f"--address={head_ip}:{ray_port}",
                f"--num-gpus={num_gpus}",
                f"--resources='{json.dumps(resources)}'",
            ]

        # Run in background with nohup
        ray_cmd = " ".join(cmd)
        return f"""nohup {ray_cmd} > /var/log/ray.log 2>&1 &
echo $! > /var/run/ray.pid

# Stream Ray logs to events.jsonl in background
(tail -f /var/log/ray.log 2>/dev/null | while IFS= read -r line; do
    emit_console "[ray] $line"
done) &"""

    return generate


def server_ops(
    node_id: int,
    head_ip: str | None,
    num_gpus: int = 1,
    ray_version: str = "2.53.0",
) -> tuple[Op, ...]:
    """Generate all operations for Ray server setup.

    This is the main entry point for Ray bootstrap, replacing
    the RPyC systemd service setup.

    Args:
        node_id: Node ID (0 = head, >0 = worker).
        head_ip: IP of head node (required for workers).
        num_gpus: Number of GPUs on this node.
        ray_version: Ray version to install.

    Returns:
        Tuple of operations for Ray setup.

    Example:
        # Head node (node 0)
        >>> ops = server_ops(node_id=0, head_ip=None, num_gpus=1)

        # Worker node (node 1)
        >>> ops = server_ops(node_id=1, head_ip="10.0.0.1", num_gpus=1)
    """
    ops: list[Op] = [
        # Install Ray
        ray_install(ray_version),
        # Start Ray service
        ray_service(
            node_id=node_id,
            head_ip=head_ip,
            num_gpus=num_gpus,
        ),
    ]

    # Wait for Ray to be ready
    if node_id == 0:
        # Head: wait for dashboard port (Jobs API)
        # Note: We no longer wait for Ray Client port since we use Jobs API
        ops.append(wait_for_port(8265, timeout=60))  # Dashboard/Jobs API
    else:
        # Worker: wait for GCS port on localhost (confirms joined cluster)
        ops.append(shell("sleep 5"))  # Give worker time to connect

    return tuple(ops)


__all__ = [
    "ray_install",
    "ray_service",
    "server_ops",
]
