"""Shared bootstrap streaming utilities for v2 providers.

Provides async helper functions for waiting on bootstrap completion
via JSONL event streaming.
"""

from __future__ import annotations

from loguru import logger

from skyward.actors.messages import InstanceMetadata
from skyward.infra import SSHTransport


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
    from skyward.providers.common import build_wheel, _build_wheel_install_script

    # Build wheel locally
    logger.info(f"{log_prefix}Building local skyward wheel...")
    wheel_path = build_wheel()

    # Build install script
    install_script = _build_wheel_install_script(wheel_name=wheel_path.name)

    # Upload wheel to /tmp (script will move it to SKYWARD_DIR)
    logger.info(f"{log_prefix}Uploading wheel {wheel_path.name}...")
    await transport.write_bytes(f"/tmp/{wheel_path.name}", wheel_path.read_bytes())

    # Upload install script
    await transport.write_file("/tmp/.install-wheel.sh", install_script)

    # Execute install script
    sudo = "sudo " if use_sudo else ""
    logger.info(f"{log_prefix}Running wheel install script on {info.id}...")
    exit_code, stdout, stderr = await transport.run(
        f"{sudo}bash /tmp/.install-wheel.sh",
        timeout=180.0,
    )
    logger.debug(f"{log_prefix}Install script output:\n{stdout}")
    if stderr:
        logger.debug(f"{log_prefix}Install script stderr:\n{stderr}")

    if exit_code != 0:
        raise RuntimeError(f"Wheel install failed (exit {exit_code}): {stderr or stdout}")

    logger.info(f"{log_prefix}Local skyward wheel installed on {info.id}")


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
    logger.info(f"{log_prefix}Uploading bootstrap script to {info.id}...")
    await transport.write_file("/opt/skyward/bootstrap.sh", bootstrap_script)
    await transport.run("chmod +x /opt/skyward/bootstrap.sh")

    logger.info(f"{log_prefix}Running bootstrap on {info.id}...")
    await transport.run(
        "nohup /opt/skyward/bootstrap.sh > /opt/skyward/bootstrap.log 2>&1 &"
    )

    logger.info(f"{log_prefix}Bootstrap started on {info.id}")
