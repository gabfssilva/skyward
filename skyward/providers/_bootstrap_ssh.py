"""Shared bootstrap streaming utilities for v2 providers.

Provides async helper functions for waiting on bootstrap completion
via JSONL event streaming.
"""

from __future__ import annotations

from skyward.actors.messages import InstanceMetadata
from skyward.infra import SSHTransport
from skyward.observability.logger import logger
from skyward.providers.bootstrap.compose import SKYWARD_DIR


async def wait_for_ssh(
    host: str,
    user: str,
    key_path: str,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
    port: int = 22,
    log_prefix: str = "",  # noqa: ARG001 - kept for API compatibility
) -> SSHTransport:
    """Wait for SSH to become available and return connected transport.

    Retry logic is built into SSHTransport.connect().

    Args:
        host: SSH host (IP or hostname).
        user: SSH username.
        key_path: Path to SSH private key.
        timeout: Maximum time to wait for SSH (used to calculate max_attempts).
        poll_interval: Time between connection attempts.
        port: SSH port.
        log_prefix: Prefix for log messages (kept for API compatibility).

    Returns:
        Connected SSHTransport.

    Raises:
        TimeoutError: If SSH is not available within timeout.
    """
    transport = SSHTransport(
        host=host,
        user=user,
        key_path=key_path,
        port=port,
        retry_max_attempts=int(timeout / poll_interval) + 1,
        retry_delay=poll_interval,
    )

    await transport.connect()
    return transport


async def install_local_skyward(
    transport: SSHTransport,
    info: InstanceMetadata,
    log_prefix: str = "",
    use_sudo: bool = True,
    bootstrap_timeout: float = 180.0,
) -> None:
    """Install local skyward wheel.

    When skyward_source='local', the bootstrap script doesn't install skyward.
    This function builds the wheel locally, uploads it, and installs it.
    The Casty server is started by bootstrap, not here.

    Args:
        transport: Connected SSH transport.
        info: Instance info (for logging).
        log_prefix: Prefix for log messages.
        use_sudo: Whether to use sudo (False for root containers like RunPod/VastAI).

    Raises:
        RuntimeError: If wheel build or installation fails.
    """
    from skyward.providers.common import _build_wheel_install_script, build_wheel

    # Build wheel locally
    log = logger.bind(component="bootstrap_ssh")
    log.info("Building local skyward wheel")
    wheel_path = build_wheel()

    # Build install script
    install_script = _build_wheel_install_script(wheel_name=wheel_path.name)

    # Upload wheel to /tmp (script will move it to SKYWARD_DIR)
    log.info("Uploading wheel {name}", name=wheel_path.name)
    await transport.write_bytes(f"/tmp/{wheel_path.name}", wheel_path.read_bytes())

    # Upload install script
    await transport.write_file("/tmp/.install-wheel.sh", install_script)

    # Execute install script
    sudo = "sudo " if use_sudo else ""
    log.info("Running wheel install script on {iid}", iid=info.id)
    exit_code, stdout, stderr = await transport.run(
        f"{sudo}bash /tmp/.install-wheel.sh",
        timeout=bootstrap_timeout,
    )
    log.debug("Install script output:\n{out}", out=stdout)
    if stderr:
        log.debug("Install script stderr:\n{err}", err=stderr)

    if exit_code != 0:
        raise RuntimeError(f"Wheel install failed (exit {exit_code}): {stderr or stdout}")

    log.info("Local skyward wheel installed on {iid}", iid=info.id)


async def upload_user_code(
    transport: SSHTransport,
    tarball: bytes,
    use_sudo: bool = True,
    timeout: float = 60.0,
) -> None:
    """Upload and extract user code tarball into the worker's site-packages.

    Extracts directly into the venv's site-packages so modules are
    importable without sys.path manipulation.

    Args:
        transport: Connected SSH transport.
        tarball: Compressed tar.gz bytes from build_user_code_tarball.
        use_sudo: Whether to use sudo for extraction.
    """
    log = logger.bind(component="bootstrap_ssh")
    remote_tar = "/tmp/_user_code.tar.gz"

    log.info("Uploading user code ({size:.1f} KB)", size=len(tarball) / 1024)
    await transport.write_bytes(remote_tar, tarball)

    sudo = "sudo " if use_sudo else ""
    site_packages = f"{SKYWARD_DIR}/.venv/lib/python*/site-packages"

    exit_code, stdout, stderr = await transport.run(
        f"{sudo}bash -c 'SP=$(echo {site_packages}); "
        f"tar xzf {remote_tar} -C $SP && rm -f {remote_tar}'",
        timeout=timeout,
    )

    if exit_code != 0:
        raise RuntimeError(f"User code extraction failed (exit {exit_code}): {stderr or stdout}")

    log.info("User code uploaded and extracted to site-packages")


async def sync_user_code(
    host: str,
    user: str,
    key_path: str,
    port: int,
    image: object,
    use_sudo: bool = True,
    ssh_timeout: float = 60.0,
    ssh_retry_interval: float = 5.0,
) -> None:
    """Build and upload user code if image.includes is non-empty.

    Connects via SSH, builds tarball from local includes, uploads and extracts.

    Args:
        host: SSH host.
        user: SSH username.
        key_path: Path to SSH private key.
        port: SSH port.
        image: Image spec (must have includes/excludes attributes).
        use_sudo: Whether to use sudo.
    """
    includes = getattr(image, "includes", ())
    if not includes:
        return

    from skyward.providers.common import build_user_code_tarball

    excludes = getattr(image, "excludes", ())
    tarball = build_user_code_tarball(includes=includes, excludes=excludes)

    transport = await wait_for_ssh(
        host=host, user=user, key_path=key_path,
        port=port, timeout=ssh_timeout, poll_interval=ssh_retry_interval,
    )

    try:
        await upload_user_code(transport=transport, tarball=tarball, use_sudo=use_sudo)
    finally:
        await transport.close()


async def run_bootstrap_via_ssh(
    transport: SSHTransport,
    info: InstanceMetadata,
    bootstrap_script: str,
    log_prefix: str = "",
) -> None:
    """Upload and execute bootstrap script via SSH (fire-and-forget).

    For providers that can't use cloud-init (e.g., VastAI with onstart_cmd limit),
    this function uploads and starts the bootstrap script in the background.

    Event streaming is handled separately by the streaming actor.

    Args:
        transport: Connected SSH transport.
        info: Instance info (for logging).
        bootstrap_script: The complete bootstrap script content.
        log_prefix: Prefix for log messages.
    """
    log = logger.bind(component="bootstrap_ssh")
    log.info("Uploading bootstrap script to {iid}", iid=info.id)
    await transport.write_file("/opt/skyward/bootstrap.sh", bootstrap_script)
    await transport.run("chmod +x /opt/skyward/bootstrap.sh")

    log.info("Running bootstrap on {iid}", iid=info.id)
    await transport.run(
        "nohup /opt/skyward/bootstrap.sh > /opt/skyward/bootstrap.log 2>&1 &"
    )

    log.info("Bootstrap started on {iid}", iid=info.id)
