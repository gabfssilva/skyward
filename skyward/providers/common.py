"""Common utilities shared between providers (SSH, tunnels, bootstrap).

NOTE: SSH key utilities have been moved to skyward.providers.base.ssh_keys.
      They are re-exported here for backwards compatibility.
"""

from __future__ import annotations

import socket
import subprocess
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING

from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
)

from skyward.callback import emit
from skyward.constants import RPYC_PORT, SKYWARD_DIR
from skyward.events import BootstrapProgress, ProvisionedInstance

if TYPE_CHECKING:
    from skyward.providers.base import Transport
    from skyward.types import ComputeSpec, Instance


# =============================================================================
# Tunnel Utilities
# =============================================================================


class TunnelNotReadyError(Exception):
    """Tunnel not ready - retry."""


def find_available_port() -> int:
    """Find an available local port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        logger.debug(f"Found available local port: {port}")
        return port


def wait_for_tunnel(port: int, timeout: int = 30) -> None:
    """Wait for tunnel to accept connections."""

    @retry(
        stop=stop_after_delay(timeout),
        wait=wait_fixed(0.5),
        retry=retry_if_exception_type(TunnelNotReadyError),
        reraise=True,
    )
    def _check() -> None:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except (TimeoutError, ConnectionRefusedError, OSError):
            raise TunnelNotReadyError() from None

    try:
        _check()
    except RetryError as e:
        raise TimeoutError(f"Tunnel not ready on port {port}") from e


def create_tunnel(
    cmd: list[str],
    local_port: int,
    timeout: int = 30,
) -> tuple[int, Popen[bytes]]:
    """Create a tunnel process and wait for it to be ready."""
    logger.debug(f"Creating tunnel on port {local_port} with cmd: {cmd[:3]}...")
    proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    try:
        wait_for_tunnel(local_port, timeout=timeout)
        logger.debug(f"Tunnel ready on port {local_port}")
        return local_port, proc
    except Exception as e:
        logger.warning(f"Tunnel creation failed on port {local_port}: {e}")
        proc.terminate()
        raise


# =============================================================================
# Bootstrap Utilities
# =============================================================================


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Bootstrap checkpoint for progress tracking."""

    file: str
    name: str


CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(".step_uv", "uv"),
    Checkpoint(".step_apt", "apt"),
    Checkpoint(".step_pip", "pip deps"),
    Checkpoint(".step_wheel", "skyward"),
    Checkpoint(".step_server", "server"),
)


class BootstrapNotReadyError(Exception):
    """Raised when bootstrap check should be retried."""


class BootstrapFailedError(Exception):
    """Raised when bootstrap script failed with an error."""

    def __init__(self, error_msg: str):
        self.error_msg = error_msg
        super().__init__(error_msg)


CommandRunner = Callable[[str], str]


@retry(
    stop=stop_after_delay(30),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(BootstrapNotReadyError),
    reraise=True,
)
def _wait_for_rpyc_service(
    run_command: CommandRunner,
    instance_id: str,
    port: int = RPYC_PORT,
) -> None:
    """Esperar a porta RPyC estar aceitando conexões.

    Importante para AMIs onde o checkpoint .ready já existe mas
    o serviço ainda está iniciando via systemd.
    """
    check_cmd = f"nc -z 127.0.0.1 {port} && echo OK || echo WAITING"
    try:
        result = run_command(check_cmd)
        if "OK" not in result:
            raise BootstrapNotReadyError()
    except BootstrapNotReadyError:
        raise
    except Exception:
        raise BootstrapNotReadyError()


def wait_for_bootstrap(
    run_command: CommandRunner,
    instance: ProvisionedInstance,
    timeout: int = 300,
    extra_checkpoints: tuple[Checkpoint, ...] = (),
) -> None:
    """Wait for instance bootstrap with progress tracking."""
    instance_id = instance.instance_id
    logger.debug(f"Waiting for bootstrap on {instance_id} (timeout={timeout}s)")
    all_checkpoints = CHECKPOINTS + extra_checkpoints

    checkpoint_files = " ".join(c.file for c in all_checkpoints)
    check_cmd = (
        f"cd {SKYWARD_DIR} 2>/dev/null || exit 0; "
        f"for f in {checkpoint_files} .ready .error; do "
        "[ -f $f ] && echo $f; done; "
        "if [ -f .error ]; then echo '---ERROR---'; cat .error; fi; exit 0"
    )

    completed_steps: set[str] = set()

    @retry(
        stop=stop_after_delay(timeout),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((BootstrapNotReadyError, TimeoutError)),
        reraise=True,
    )
    def _poll_bootstrap() -> None:
        nonlocal completed_steps

        try:
            stdout = run_command(check_cmd)
        except TimeoutError:
            raise
        except Exception as e:
            raise BootstrapNotReadyError() from e

        stdout = stdout.strip()

        if "---ERROR---" in stdout:
            error_msg = stdout.split("---ERROR---", 1)[1].strip()
            raise BootstrapFailedError(error_msg)

        found_files = set(stdout.split("\n")) if stdout else set()

        for checkpoint in all_checkpoints:
            if checkpoint.file in found_files and checkpoint.name not in completed_steps:
                emit(BootstrapProgress(instance=instance, step=checkpoint.name))
                completed_steps.add(checkpoint.name)

        if ".ready" in found_files:
            # Bootstrap complete - service will be started after wheel installation
            logger.debug(f"Bootstrap ready on {instance_id}")
            return

        raise BootstrapNotReadyError()

    try:
        _poll_bootstrap()
        logger.info(f"Bootstrap completed on {instance_id}")
    except BootstrapFailedError as e:
        logger.error(f"Bootstrap failed on {instance_id}: {e.error_msg}")
        raise RuntimeError(f"Bootstrap failed on {instance_id}:\n{e.error_msg}") from None
    except (RetryError, BootstrapNotReadyError) as e:
        logger.error(f"Bootstrap timed out on {instance_id} after {timeout}s")
        raise RuntimeError(f"Bootstrap timed out on {instance_id}") from e


# =============================================================================
# SSH Utilities
# =============================================================================

# NOTE: SSHKeyInfo, SSH_KEY_PATHS, find_local_ssh_key, get_private_key_path,
#       and compute_fingerprint have been moved to skyward.providers.base.ssh_keys
#       and are re-exported above for backwards compatibility.


def ssh_run(
    ip: str,
    username: str,
    command: str,
    timeout: int = 60,
    key_path: str | None = None,
    port: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run SSH command on remote instance."""
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=10",
    ]
    if key_path:
        ssh_cmd.extend(["-i", key_path])
    if port:
        ssh_cmd.extend(["-p", str(port)])
    ssh_cmd.extend([f"{username}@{ip}", command])
    return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)


def scp_upload(
    ip: str,
    username: str,
    local_path: Path,
    remote_path: str,
    key_path: str | None = None,
    port: int | None = None,
) -> None:
    """Upload file to remote via SCP."""
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]
    if key_path:
        scp_cmd.extend(["-i", key_path])
    if port:
        scp_cmd.extend(["-P", str(port)])
    scp_cmd.extend([str(local_path), f"{username}@{ip}:{remote_path}"])
    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")


def create_ssh_runner(
    ip: str,
    username: str,
    key_path: str | None = None,
) -> Callable[[str], str]:
    """Create a command runner using SSH."""

    def run_command(cmd: str) -> str:
        result = ssh_run(ip, username, cmd, timeout=30, key_path=key_path)
        if result.returncode != 0:
            raise RuntimeError(f"SSH command failed (exit {result.returncode}): {result.stderr}")
        return result.stdout

    return run_command


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
) -> str:
    """Build complete wheel installation + service setup script.

    Single script that does everything:
    1. Find uv
    2. Install wheel
    3. Verify installation
    4. Setup single RPyC systemd service

    Args:
        wheel_name: Name of the wheel file to install.
        env: Environment variables to pass to the systemd service.

    Returns:
        Shell script string.
    """
    from skyward.bootstrap import bootstrap as bootstrap_ops
    from skyward.bootstrap import systemd
    from skyward.bootstrap import wait_for_port as wait_for_port_op
    from skyward.bootstrap.worker import rpyc_service_unit

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
"""

    # Always use single RPyC service (concurrency handled via slots, not workers)
    unit_content = rpyc_service_unit(env=env)
    service_script = bootstrap_ops(
        systemd("skyward-rpyc", unit_content),
        wait_for_port_op(RPYC_PORT, timeout=30),
        header="",
    )

    return f"#!/bin/bash\nset -e\n{preamble}\n{service_script}"


def install_skyward_wheel_via_transport(
    instances: tuple[Instance, ...],
    get_transport: Callable[[Instance], AbstractContextManager[Transport]],
    compute: ComputeSpec | None = None,
) -> None:
    """Build and install skyward wheel on all instances using Transport abstraction.

    This is the preferred way to install the wheel - it works with any transport
    (SSH, SSM tunneled SSH, etc.) via the Transport protocol.

    Args:
        instances: Instances to install on.
        get_transport: Factory that creates a context manager yielding a Transport
            for each instance. The context manager handles lifecycle (e.g., tunnel cleanup).
        compute: Compute spec containing image with environment variables.

    Example:
        # SSH-based providers (Verda, DigitalOcean)
        @contextmanager
        def ssh_transport(inst):
            yield SSHTransport(host=get_ip(inst), username="root", key_path=key_path)

        install_skyward_wheel_via_transport(instances, ssh_transport, compute)

        # AWS with SSM tunnel
        @contextmanager
        def ssm_ssh_transport(inst):
            local_port, tunnel = ssm.create_tunnel_to_ssh(inst.id)
            try:
                yield SSHTransport(host="localhost", port=local_port, ...)
            finally:
                tunnel.terminate()

        install_skyward_wheel_via_transport(instances, ssm_ssh_transport, compute)
    """
    from skyward.conc import for_each_async

    logger.info(f"Installing skyward wheel on {len(instances)} instances...")
    wheel_path = build_wheel()

    # Get environment variables from compute spec's image
    env = compute.image.env if compute else None

    # Build single script that does everything (always single RPyC service)
    install_script = _build_wheel_install_script(wheel_path.name, env=env)

    def install_on_instance(inst: Instance) -> None:
        import tempfile

        logger.debug(f"Installing wheel on {inst.id}...")
        with get_transport(inst) as transport:
            # Upload wheel to /tmp (user has write access, script will move to SKYWARD_DIR)
            tmp_wheel = f"/tmp/{wheel_path.name}"
            transport.upload_file(wheel_path, tmp_wheel)
            logger.debug(f"Uploaded wheel to {inst.id}:{tmp_wheel}")

            # Upload install script to /tmp (safer than passing via bash -c with escaping issues)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                f.write(install_script)
                local_script = Path(f.name)

            try:
                remote_script = "/tmp/.install-wheel.sh"
                transport.upload_file(local_script, remote_script)

                # Single remote call: execute script (runs as sudo to have write access)
                logger.debug(f"Running install script on {inst.id}...")
                transport.run_command(f"sudo bash {remote_script}", timeout=180)
                logger.info(f"Wheel installed on {inst.id}")
            finally:
                local_script.unlink(missing_ok=True)

    for_each_async(install_on_instance, instances)
    logger.info(f"Wheel installation complete on all {len(instances)} instances")


def wait_for_ssh_bootstrap(
    instances: tuple[Instance, ...],
    get_ip: Callable[[Instance], str],
    make_provisioned: Callable[[Instance], ProvisionedInstance],
    timeout: int = 300,
    key_path: str | None = None,
) -> None:
    """Wait for bootstrap to complete on all instances via SSH (concurrent)."""
    from skyward.conc import for_each_async

    logger.info(f"Waiting for SSH bootstrap on {len(instances)} instances...")

    extra_checkpoints = (
        Checkpoint(".step_venv", "venv"),
        Checkpoint(".step_base", "base deps"),
    )

    username = instances[0].get_meta("username", "root")

    def wait_for_instance(inst: Instance) -> None:
        ip = get_ip(inst)
        logger.debug(f"Checking bootstrap status for {inst.id} at {ip}")
        runner = create_ssh_runner(ip, username, key_path)
        provisioned = make_provisioned(inst)
        wait_for_bootstrap(
            run_command=runner,
            instance=provisioned,
            timeout=timeout,
            extra_checkpoints=extra_checkpoints,
        )

    for_each_async(wait_for_instance, instances)
    logger.info(f"All {len(instances)} instances bootstrapped successfully")


