"""Bootstrap operations for DigitalOcean Droplets.

Uses the common bootstrap module with DO-specific extensions:
- SSH for remote command execution
- Wheel upload via SCP
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from skyward.constants import SKYWARD_DIR
from skyward.events import EventCallback
from skyward.providers.common.bootstrap import (
    Checkpoint,
    generate_base_script,
)
from skyward.providers.common.bootstrap import (
    wait_for_bootstrap as _wait_for_bootstrap,
)
from skyward.types import ComputeSpec, Instance

logger = logging.getLogger("skyward.digitalocean.bootstrap")


# DO-specific checkpoints
DO_EXTRA_CHECKPOINTS = (
    Checkpoint(".step_venv", "venv"),
    Checkpoint(".step_base", "base deps"),
)


def _create_ssh_runner(ip: str, username: str) -> Callable[[str], str]:
    """Create a command runner using SSH.

    Returns a function that runs commands via SSH and returns stdout.
    """

    def run_command(cmd: str) -> str:
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            f"{username}@{ip}",
            cmd,
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout
        return ""

    return run_command


def wait_for_bootstrap(
    instances: tuple[Instance, ...],
    on_event: EventCallback = None,
    timeout: int = 300,
) -> None:
    """Wait for bootstrap to complete on all droplets.

    Args:
        instances: Instances to wait for.
        on_event: Optional callback for bootstrap progress events.
        timeout: Maximum time to wait per instance in seconds.
    """
    username = instances[0].get_meta("username", "root")

    for inst in instances:
        ip = inst.get_meta("droplet_ip")
        runner = _create_ssh_runner(ip, username)

        _wait_for_bootstrap(
            run_command=runner,
            instance_id=inst.id,
            on_event=on_event,
            timeout=timeout,
            extra_checkpoints=DO_EXTRA_CHECKPOINTS,
        )


def install_skyward_wheel(instances: tuple[Instance, ...]) -> None:
    """Build and install skyward wheel on all instances (concurrent)."""
    from skyward.conc import for_each_async

    wheel_path = _build_wheel()
    logger.info(f"Built skyward wheel: {wheel_path}")

    def install_on_instance(inst: Instance) -> None:
        ip = inst.get_meta("droplet_ip")
        username = inst.get_meta("username", "root")

        # Find uv path dynamically
        find_uv_cmd = "which uv || find /root -name uv -type f 2>/dev/null | head -1"
        find_uv_result = _ssh_run(ip, username, find_uv_cmd, timeout=30)
        uv_path = find_uv_result.stdout.strip()
        if not uv_path:
            raise RuntimeError(f"uv not found on {ip}")
        logger.info(f"Found uv at: {uv_path}")

        # Upload wheel
        remote_wheel = f"{SKYWARD_DIR}/{wheel_path.name}"
        logger.info(f"Uploading wheel to {ip}...")
        _scp_upload(ip, username, wheel_path, remote_wheel)

        # Install wheel
        logger.info(f"Installing wheel on {ip}...")
        install_cmd = f"{uv_path} pip install --python {SKYWARD_DIR}/venv/bin/python {remote_wheel}"
        result = _ssh_run(ip, username, install_cmd, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install wheel on {ip}: {result.stderr}")

        # Verify skyward is importable
        logger.info(f"Verifying skyward installation on {ip}...")
        verify_cmd = f"{SKYWARD_DIR}/venv/bin/python -c 'import skyward; print(skyward.__file__)'"
        verify_result = _ssh_run(ip, username, verify_cmd, timeout=30)
        if verify_result.returncode != 0:
            raise RuntimeError(f"skyward not importable on {ip}: {verify_result.stderr}")
        logger.info(f"skyward installed at: {verify_result.stdout.strip()}")

        # Start RPyC server
        logger.info(f"Starting RPyC server on {ip}...")
        start_result = _ssh_run(ip, username, "systemctl start skyward-rpyc", timeout=30)
        if start_result.returncode != 0:
            raise RuntimeError(f"Failed to start RPyC on {ip}: {start_result.stderr}")

        logger.info(f"Skyward ready on {ip}")

    for_each_async(install_on_instance, instances)


def create_user_data_script(
    compute: ComputeSpec,
    instance_timeout: int | None = None,
) -> str:
    """Create cloud-init user_data script.

    Uses the common base script generator with DO-specific configuration.
    """
    from skyward.image import Image

    image = Image(
        python=compute.python,
        pip=list(compute.pip),
        apt=list(compute.apt),
        env=dict(compute.env),
    )

    # DO doesn't pre-upload wheel - it's installed later via SCP
    # So we skip the wheel step in user_data and do it in setup phase
    return generate_base_script(
        python=compute.python or "3.12",
        pip=tuple(image.pip) if image.pip else (),
        apt=tuple(image.apt) if image.apt else (),
        env=dict(image.env) if image.env else None,
        instance_timeout=instance_timeout,
        preamble="",  # No DO-specific preamble needed
        postamble="# Skyward wheel installed via SCP in setup phase",
    )


def _ssh_run(
    ip: str,
    username: str,
    command: str,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Run SSH command on droplet."""
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        f"{username}@{ip}",
        command,
    ]
    return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)


def _build_wheel() -> Path:
    """Build skyward wheel locally."""
    skyward_dir = Path(__file__).parent.parent.parent
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
        raise RuntimeError(f"Failed to build wheel: {result.stderr}")

    return next(build_dir.glob("*.whl"))


def _scp_upload(ip: str, username: str, local_path: Path, remote_path: str) -> None:
    """Upload file to remote via SCP."""
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        str(local_path),
        f"{username}@{ip}:{remote_path}",
    ]
    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")
