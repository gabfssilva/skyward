"""Bootstrap operations for Verda instances.

Uses the common bootstrap module with Verda-specific extensions:
- SSH for remote command execution
- Wheel upload via SCP
- Startup scripts via Verda API
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from skyward.constants import SKYWARD_DIR
from skyward.providers.common.bootstrap import (
    Checkpoint,
    generate_base_script,
)
from skyward.providers.common.bootstrap import (
    wait_for_bootstrap as _wait_for_bootstrap,
)
from skyward.providers.verda.ssh import find_local_ssh_key
from skyward.types import ComputeSpec, Instance


# Verda-specific checkpoints (same as DO since we use similar bootstrap)
VERDA_EXTRA_CHECKPOINTS = (
    Checkpoint(".step_venv", "venv"),
    Checkpoint(".step_base", "base deps"),
)


def _get_ssh_private_key_path() -> str | None:
    """Get path to SSH private key (without .pub extension)."""
    key_info = find_local_ssh_key()
    if key_info is None:
        return None
    pub_path, _ = key_info
    # Remove .pub to get private key path
    private_path = pub_path.with_suffix("")
    if private_path.exists():
        return str(private_path)
    return None


def _create_ssh_runner(ip: str, username: str) -> Callable[[str], str]:
    """Create a command runner using SSH.

    Returns a function that runs commands via SSH and returns stdout.
    Raises RuntimeError on SSH failures to trigger retry logic.
    """
    key_path = _get_ssh_private_key_path()

    def run_command(cmd: str) -> str:
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
        ]
        if key_path:
            ssh_cmd.extend(["-i", key_path])
        ssh_cmd.extend([f"{username}@{ip}", cmd])

        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            # Raise to trigger retry in wait_for_bootstrap
            raise RuntimeError(f"SSH command failed (exit {result.returncode}): {result.stderr}")
        return result.stdout

    return run_command


def wait_for_bootstrap(
    instances: tuple[Instance, ...],
    timeout: int = 300,
) -> None:
    """Wait for bootstrap to complete on all instances (concurrent).

    Args:
        instances: Instances to wait for.
        timeout: Maximum time to wait per instance in seconds.
    """
    from skyward.conc import for_each_async

    username = instances[0].get_meta("username", "root")

    def wait_for_instance(inst: Instance) -> None:
        ip = inst.get_meta("instance_ip") or inst.public_ip
        runner = _create_ssh_runner(ip, username)
        _wait_for_bootstrap(
            run_command=runner,
            instance_id=inst.id,
            timeout=timeout,
            extra_checkpoints=VERDA_EXTRA_CHECKPOINTS,
        )

    for_each_async(wait_for_instance, instances)


def install_skyward_wheel(instances: tuple[Instance, ...]) -> None:
    """Build and install skyward wheel on all instances (concurrent)."""
    from skyward.conc import for_each_async

    wheel_path = _build_wheel()

    def install_on_instance(inst: Instance) -> None:
        ip = inst.get_meta("instance_ip") or inst.public_ip
        username = inst.get_meta("username", "root")

        # Find uv path dynamically
        find_uv_cmd = "which uv || find /root /home -name uv -type f 2>/dev/null | head -1"
        find_uv_result = _ssh_run(ip, username, find_uv_cmd, timeout=30)
        uv_path = find_uv_result.stdout.strip()
        if not uv_path:
            raise RuntimeError(f"uv not found on {ip}")

        # Upload wheel
        remote_wheel = f"{SKYWARD_DIR}/{wheel_path.name}"
        _scp_upload(ip, username, wheel_path, remote_wheel)

        # Install wheel
        install_cmd = f"{uv_path} pip install --python {SKYWARD_DIR}/venv/bin/python {remote_wheel}"
        result = _ssh_run(ip, username, install_cmd, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install wheel on {ip}: {result.stderr}")

        # Verify skyward is importable
        verify_cmd = f"{SKYWARD_DIR}/venv/bin/python -c 'import skyward; print(skyward.__file__)'"
        verify_result = _ssh_run(ip, username, verify_cmd, timeout=30)
        if verify_result.returncode != 0:
            raise RuntimeError(f"skyward not importable on {ip}: {verify_result.stderr}")

        # Restart RPyC server (single or multi-worker)
        restart_cmd = (
            "if systemctl is-active skyward-rpyc >/dev/null 2>&1; then "
            "systemctl restart skyward-rpyc; "
            "else "
            "systemctl list-units 'skyward-worker@*' --no-legend | awk '{print $1}' | xargs -r systemctl restart; "
            "fi"
        )
        _ssh_run(ip, username, f"sudo bash -c \"{restart_cmd}\"", timeout=60)

    for_each_async(install_on_instance, instances)


def create_user_data_script(
    compute: ComputeSpec,
    instance_timeout: int | None = None,
) -> str:
    """Create cloud-init user_data script.

    Uses the common base script generator with Verda-specific configuration.
    """
    from skyward.image import Image

    image = Image(
        python=compute.python,
        pip=list(compute.pip),
        apt=list(compute.apt),
        env=dict(compute.env),
    )

    # Verda doesn't pre-upload wheel - it's installed later via SCP
    worker_bs = getattr(compute, "worker_bootstrap_script", "")

    return generate_base_script(
        python=compute.python or "3.12",
        pip=tuple(image.pip) if image.pip else (),
        apt=tuple(image.apt) if image.apt else (),
        env=dict(image.env) if image.env else None,
        instance_timeout=instance_timeout,
        preamble="",
        postamble="# Skyward wheel installed via SCP in setup phase",
        worker_bootstrap=worker_bs,
    )


def _ssh_run(
    ip: str,
    username: str,
    command: str,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Run SSH command on instance."""
    key_path = _get_ssh_private_key_path()
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
    ]
    if key_path:
        ssh_cmd.extend(["-i", key_path])
    ssh_cmd.extend([f"{username}@{ip}", command])
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
    key_path = _get_ssh_private_key_path()
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
    ]
    if key_path:
        scp_cmd.extend(["-i", key_path])
    scp_cmd.extend([str(local_path), f"{username}@{ip}:{remote_path}"])
    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")
