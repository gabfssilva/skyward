"""Common utilities shared between providers (wheel building, etc.)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from skyward.actors.messages import InstanceRunning, ProviderName
from skyward.infra.ssh import SSHTransport
from skyward.providers.bootstrap.compose import SKYWARD_DIR


def build_instance_running_event(
    *,
    request_id: str,
    cluster_id: str,
    node_id: int,
    provider: ProviderName,
    instance_id: str,
    ip: str,
    private_ip: str,
    ssh_port: int = 22,
    spot: bool = False,
    hourly_rate: float = 0.0,
    on_demand_rate: float = 0.0,
    billing_increment: int = 1,
    instance_type: str = "",
    gpu_count: int = 0,
    gpu_model: str = "",
    vcpus: int = 0,
    memory_gb: float = 0.0,
    gpu_vram_gb: int = 0,
    network_interface: str = "",
) -> InstanceRunning:
    """Build InstanceRunning event with consistent field handling.

    Centralizes event construction to ensure all providers emit consistent
    InstanceRunning events.

    Args:
        request_id: Original request ID.
        cluster_id: Cluster the instance belongs to.
        node_id: Node index within the cluster.
        provider: Provider name (e.g., "aws", "vastai").
        instance_id: Provider-specific instance ID.
        ip: Public IP for SSH access.
        private_ip: Private IP for internal communication.
        ssh_port: SSH port (default 22).
        spot: Whether this is a spot/interruptible instance.
        hourly_rate: Current hourly cost.
        on_demand_rate: On-demand price for comparison.
        billing_increment: Billing granularity in seconds.
        instance_type: Provider-specific instance type.
        gpu_count: Number of GPUs.
        gpu_model: GPU model name.
        vcpus: Number of vCPUs.
        memory_gb: Memory in GB.
        gpu_vram_gb: GPU VRAM per GPU in GB.
        network_interface: Network interface for overlay (if applicable).

    Returns:
        Configured InstanceRunning event.
    """
    return InstanceRunning(
        request_id=request_id,
        cluster_id=cluster_id,
        node_id=node_id,
        provider=provider,
        instance_id=instance_id,
        ip=ip,
        private_ip=private_ip,
        ssh_port=ssh_port,
        spot=spot,
        hourly_rate=hourly_rate,
        on_demand_rate=on_demand_rate,
        billing_increment=billing_increment,
        instance_type=instance_type,
        gpu_count=gpu_count,
        gpu_model=gpu_model,
        vcpus=vcpus,
        memory_gb=memory_gb,
        gpu_vram_gb=gpu_vram_gb,
        network_interface=network_interface,
    )


async def detect_network_interface(transport: SSHTransport) -> str:
    """Detect the default network interface on a remote node via SSH.

    Runs `ip -o route show default` and parses the `dev` field.
    Works regardless of AMI/distro (Ubuntu ens5, Amazon Linux eth0, etc.).
    """
    exit_code, stdout, _ = await transport.run("ip -o route show default", timeout=10.0)
    if exit_code != 0 or "dev " not in stdout:
        return ""

    return stdout.split("dev ")[1].split()[0]


def build_wheel() -> Path:
    """Build skyward wheel locally into a fresh temp directory."""
    logger.debug("Building skyward wheel locally...")
    skyward_dir = Path(__file__).parent.parent
    project_dir = skyward_dir.parent

    build_dir = tempfile.mkdtemp(prefix="skyward-wheel-")

    result = subprocess.run(
        ["uv", "build", "--wheel", "-o", build_dir],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Failed to build wheel: {err}", err=result.stderr)
        raise RuntimeError(f"Failed to build wheel: {result.stderr}")

    wheel_path = next(Path(build_dir).glob("*.whl"))
    logger.info("Built wheel: {name}", name=wheel_path.name)
    return wheel_path


def _build_wheel_install_script(wheel_name: str) -> str:
    """Build wheel installation script.

    Installs the skyward wheel into the venv. Service startup (Casty)
    is handled by bootstrap, not this script.

    Args:
        wheel_name: Name of the wheel file to install.

    Returns:
        Shell script string.
    """
    script = f"""#!/bin/bash
set -e

# Move wheel from /tmp to SKYWARD_DIR (uploaded to /tmp for permission reasons)
mv /tmp/{wheel_name} {SKYWARD_DIR}/{wheel_name} 2>/dev/null || true

# Find uv
UV_PATH=$(which uv 2>/dev/null || find /root /home -name uv -type f 2>/dev/null | head -1)
if [ -z "$UV_PATH" ]; then
    UV_PATH="/root/.local/bin/uv"
fi

# Install wheel
cd {SKYWARD_DIR}
$UV_PATH pip install {SKYWARD_DIR}/{wheel_name}

# Verify installation
{SKYWARD_DIR}/.venv/bin/python -c 'import skyward; print(skyward.__file__)'

# Verify Casty can be imported
{SKYWARD_DIR}/.venv/bin/python -c 'import casty; print("Casty OK")'
"""
    return script
