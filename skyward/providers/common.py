"""Common utilities shared between providers (SSH, tunnels, bootstrap).

NOTE: SSH key utilities have been moved to skyward.providers.base.ssh_keys.
      They are re-exported here for backwards compatibility.
"""

from __future__ import annotations

import socket
import subprocess
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Protocol

from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_fixed,
)

from skyward.core.constants import RPYC_PORT, SKYWARD_DIR
from skyward.core.events import ProviderName, ProvisionedInstance

if TYPE_CHECKING:
    from skyward.types import ComputeSpec, Instance, InstanceSpec


# =============================================================================
# Transport Protocol
# =============================================================================


class Transport(Protocol):
    """Protocol for remote command execution and file transfer."""

    def run_command(self, command: str, timeout: int = 30) -> str:
        """Execute command on remote instance."""
        ...

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload file to remote instance."""
        ...


# =============================================================================
# Event Helpers
# =============================================================================


def make_provisioned(
    inst: Instance,
    provider: ProviderName,
    spec: InstanceSpec | None = None,
    ip: str | None = None,
) -> ProvisionedInstance:
    """Create ProvisionedInstance from Instance for events."""
    return ProvisionedInstance(
        instance_id=inst.id,
        node=inst.node,
        provider=provider,
        spot=inst.spot,
        spec=spec,
        ip=ip or inst.public_ip or inst.private_ip,
    )


# =============================================================================
# Tunnel Utilities
# =============================================================================


class TunnelNotReadyError(Exception):
    """Tunnel not ready - retry."""


class SSHNotReadyError(Exception):
    """SSH not ready - retry."""


def wait_for_ssh_ready(host: str, port: int, timeout: int = 300) -> None:
    """Wait until SSH is reachable on host:port.

    Uses socket connection to check if SSH is accepting connections.
    Raises TimeoutError if not ready within timeout.

    Args:
        host: Hostname or IP address.
        port: SSH port number.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If SSH is not reachable within timeout.
    """

    @retry(
        stop=stop_after_delay(timeout),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(SSHNotReadyError),
        reraise=True,
    )
    def _check() -> None:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except (TimeoutError, ConnectionRefusedError, OSError):
            raise SSHNotReadyError() from None

    try:
        logger.debug(f"Waiting for SSH on {host}:{port} (timeout={timeout}s)")
        _check()
        logger.debug(f"SSH ready on {host}:{port}")
    except RetryError as e:
        raise TimeoutError(f"SSH not ready on {host}:{port} after {timeout}s") from e


class BootstrapNotReadyError(Exception):
    """Bootstrap not ready - retry."""


def wait_for_ready(run_command: Callable[[str], str], timeout: int = 300) -> None:
    """Wait for bootstrap .ready file to exist.

    Polls the remote instance via SSH until the .ready file is created
    by the bootstrap script, indicating setup is complete.

    Args:
        run_command: Function to execute remote commands (e.g., inst.run_command).
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If .ready file not found within timeout.
    """

    @retry(
        stop=stop_after_delay(timeout),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(BootstrapNotReadyError),
        reraise=True,
    )
    def _poll() -> None:
        try:
            result = run_command(f"test -f {SKYWARD_DIR}/.ready && echo ready")
            if "ready" not in result:
                raise BootstrapNotReadyError()
        except BootstrapNotReadyError:
            raise
        except Exception:
            raise BootstrapNotReadyError() from None

    try:
        logger.debug(f"Waiting for bootstrap .ready (timeout={timeout}s)")
        _poll()
        logger.debug("Bootstrap ready")
    except RetryError as e:
        raise TimeoutError(f"Bootstrap not ready after {timeout}s") from e


def find_available_port() -> int:
    """Find an available local port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        logger.debug(f"Found available local port: {port}")
        return port


def wait_for_tunnel(port: int, timeout: int = 300) -> None:
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
# Wheel Build & Installation
# =============================================================================


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
    from skyward.bootstrap import bootstrap as bootstrap_ops
    from skyward.bootstrap import nohup_service, systemd
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

    if use_systemd:
        # Use systemd for VMs (EC2, etc.)
        unit_content = rpyc_service_unit(env=env)
        service_script = bootstrap_ops(
            systemd("skyward-rpyc", unit_content),
            wait_for_port_op(RPYC_PORT, timeout=30),
            header="",
        )
    else:
        # Use nohup for Docker containers (Vast.ai, etc.)
        env_with_path = {"PATH": f"{SKYWARD_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin"}
        if env:
            env_with_path.update(env)

        service_script = bootstrap_ops(
            nohup_service(
                name="skyward-rpyc",
                command=f"{SKYWARD_DIR}/.venv/bin/python -m skyward.rpc",
                working_dir=SKYWARD_DIR,
                env=env_with_path,
            ),
            wait_for_port_op(RPYC_PORT, timeout=30),
            header="",
        )

    return f"#!/bin/bash\nset -e\n{preamble}\n{service_script}"


def install_skyward_wheel_via_transport(
    instances: tuple[Instance, ...],
    get_transport: Callable[[Instance], AbstractContextManager[Transport]],
    compute: ComputeSpec | None = None,
    use_systemd: bool = True,
) -> None:
    """Build and install skyward wheel on all instances using Transport abstraction.

    This is the preferred way to install the wheel - it works with any transport
    (SSH, SSM tunneled SSH, etc.) via the Transport protocol.

    If the compute spec's image has skyward_source != "local", this function
    skips installation since skyward was already installed via user-data.

    Args:
        instances: Instances to install on.
        get_transport: Factory that creates a context manager yielding a Transport
            for each instance. The context manager handles lifecycle (e.g., tunnel cleanup).
        compute: Compute spec containing image with environment variables.
        use_systemd: If True, use systemd to manage the RPyC service.
            If False, use nohup (for Docker containers without systemd).

    Example:
        # Use instance.connect() for all providers
        install_skyward_wheel_via_transport(
            instances,
            get_transport=lambda inst: inst.connect(),
            compute=compute,
        )

        # Docker containers (Vast.ai) - disable systemd
        install_skyward_wheel_via_transport(
            instances,
            get_transport=lambda inst: inst.connect(),
            compute=compute,
            use_systemd=False,
        )
    """
    from skyward.utils.conc import for_each_async

    # Skip if skyward was installed via user-data (github/pypi)
    if compute and compute.image.skyward_source != "local":
        logger.info(
            f"Skyward installed via user-data ({compute.image.skyward_source}), "
            "skipping wheel installation"
        )
        return

    logger.info(f"Installing skyward wheel on {len(instances)} instances...")
    wheel_path = build_wheel()

    # Get environment variables from compute spec's image
    env = compute.image.env if compute else None

    # Build single script that does everything (always single RPyC service)
    install_script = _build_wheel_install_script(wheel_path.name, env=env, use_systemd=use_systemd)

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


def install_wheel_on_instance(
    instance: Instance,
    compute: ComputeSpec,
    use_systemd: bool = True,
) -> None:
    """Install skyward wheel on a single instance.

    Convenience wrapper around install_skyward_wheel_via_transport for
    single-instance operations. Uses instance.connect() for SSH transport.

    Args:
        instance: Instance to install on.
        compute: Compute spec containing image with environment variables.
        use_systemd: If True, use systemd to manage RPyC service.
            If False, use nohup (for Docker containers without systemd).
    """
    install_skyward_wheel_via_transport(
        instances=(instance,),
        get_transport=lambda inst: inst.connect(),
        compute=compute,
        use_systemd=use_systemd,
    )


