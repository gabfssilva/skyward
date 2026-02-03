"""Common utilities shared between providers (wheel building, etc.)."""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from loguru import logger

from skyward.constants import RPYC_PORT, SKYWARD_DIR
from skyward.events import InstanceRunning, ProviderName


def generate_cluster_id(provider: ProviderName) -> str:
    """Generate a unique cluster ID for a provider.

    Args:
        provider: Provider name (e.g., "aws", "vastai", "verda").

    Returns:
        Unique cluster ID in format "{provider}-{uuid8}".
    """
    return f"{provider}-{uuid.uuid4().hex[:8]}"


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


def build_wheel() -> Path:
    """Build skyward wheel locally."""
    logger.debug("Building skyward wheel locally...")
    skyward_dir = Path(__file__).parent.parent
    project_dir = skyward_dir.parent

    build_dir = Path("/tmp/skyward-wheel")
    build_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        ["uv", "build", "--wheel", "-o", str(build_dir)],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Failed to build wheel: {result.stderr}")
        raise RuntimeError(f"Failed to build wheel: {result.stderr}")

    wheel_path = next(build_dir.glob("*.whl"))
    logger.info(f"Built wheel: {wheel_path.name}")
    return wheel_path


def _build_wheel_install_script(
    wheel_name: str,
    env: dict[str, str] | None = None,
    use_systemd: bool = True,
) -> str:
    """Build complete wheel installation + service setup script.

    Single script that does everything:
    1. Find uv
    2. Install wheel
    3. Verify installation
    4. Setup single RPyC service (systemd or nohup)

    Args:
        wheel_name: Name of the wheel file to install.
        env: Environment variables to pass to the service.
        use_systemd: If True, use systemd. If False, use nohup (for Docker).

    Returns:
        Shell script string.
    """
    from skyward.bootstrap import (
        resolve,
        nohup_service,
        systemd,
        wait_for_port,
        rpyc_service_unit,
    )

    # Common preamble: move wheel from /tmp (where user can upload) and install
    preamble = f"""
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

# Verify RPyC can be imported (catches missing dependencies)
{SKYWARD_DIR}/.venv/bin/python -c 'import rpyc; from skyward.rpc.server import SkywardService; print("RPyC imports OK")'
"""

    if use_systemd:
        # Use systemd for VMs (EC2, etc.)
        # Use resolve() instead of bootstrap() to avoid duplicate headers
        unit_content = rpyc_service_unit(env=env)
        service_script = "\n".join([
            resolve(systemd("skyward-rpyc", unit_content)),
            resolve(wait_for_port(RPYC_PORT, timeout=30)),
        ])
    else:
        # Use nohup for Docker containers (Vast.ai, etc.)
        env_with_path = {"PATH": f"{SKYWARD_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin"}
        if env:
            env_with_path.update(env)

        # Use resolve() instead of bootstrap() to avoid duplicate headers
        service_script = "\n".join([
            resolve(nohup_service(
                name="skyward-rpyc",
                command=f"{SKYWARD_DIR}/.venv/bin/python -m skyward.rpc",
                working_dir=SKYWARD_DIR,
                env=env_with_path,
            )),
            resolve(wait_for_port(RPYC_PORT, timeout=30)),
        ])

    return f"#!/bin/bash\nset -e\n{preamble}\n{service_script}"


__all__ = [
    "generate_cluster_id",
    "build_instance_running_event",
    "build_wheel",
    "_build_wheel_install_script",
]
