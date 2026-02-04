"""Shared bootstrap streaming utilities for v2 providers.

Provides async helper functions for waiting on bootstrap completion
via JSONL event streaming.
"""

from __future__ import annotations

from loguru import logger

from skyward.bus import AsyncEventBus
from skyward.events import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapFailed,
    BootstrapPhase,
    InstanceMetadata,
)
from skyward.transport import (
    BootstrapError,
    RawBootstrapCommand,
    RawBootstrapConsole,
    RawBootstrapPhase,
    SSHTransport,
)


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


async def stream_bootstrap_events(
    transport: SSHTransport,
    info: InstanceMetadata,
    bus: AsyncEventBus,
    timeout: float = 600.0,
    log_prefix: str = "",
) -> None:
    """Stream bootstrap events from instance and emit to bus.

    Streams the events.jsonl file via SSH, converting raw events to
    typed bus events with instance info.

    Args:
        transport: Connected SSH transport.
        info: Instance info (for event enrichment).
        bus: Event bus to emit events to.
        timeout: Maximum time between events (adaptive).
        log_prefix: Prefix for log messages (e.g., "AWS: ").

    Raises:
        BootstrapError: If bootstrap fails.
        TimeoutError: If timeout exceeded.
    """
    logger.info(f"{log_prefix}Starting bootstrap event streaming for {info.id}")

    try:
        async for raw_event in transport.stream_bootstrap(timeout=timeout):
            # Convert raw events to bus events with instance info
            match raw_event:
                case RawBootstrapConsole(content=content, stream=stream):
                    bus.emit(BootstrapConsole(
                        instance=info,
                        content=content,
                        stream=stream,
                    ))
                    # Log console output (truncate if too long)
                    display = content[:100] + "..." if len(content) > 100 else content
                    if stream == "stderr":
                        logger.warning(f"{log_prefix}[stderr] {display}")
                    else:
                        logger.debug(f"{log_prefix}[stdout] {display}")
                case RawBootstrapPhase(event=event, phase=phase, elapsed=elapsed, error=error):
                    bus.emit(BootstrapPhase(
                        instance=info,
                        event=event,
                        phase=phase,
                        elapsed=elapsed,
                        error=error,
                    ))
                    # Log phase transitions
                    if event == "started":
                        logger.info(f"{log_prefix}Phase '{phase}' started")
                    elif event == "completed":
                        elapsed_str = f" ({elapsed:.1f}s)" if elapsed else ""
                        logger.info(f"{log_prefix}Phase '{phase}' completed{elapsed_str}")
                    elif event == "failed":
                        logger.error(f"{log_prefix}Phase '{phase}' FAILED: {error}")
                case RawBootstrapCommand(command=command):
                    bus.emit(BootstrapCommand(
                        instance=info,
                        command=command,
                    ))
                    # Log command (truncate if too long)
                    display = command[:80] + "..." if len(command) > 80 else command
                    logger.debug(f"{log_prefix}Running: {display}")

        logger.info(f"{log_prefix}Bootstrap complete on {info.id}")

    except BootstrapError as e:
        # Emit failure event
        bus.emit(BootstrapFailed(
            instance=info,
            phase=e.phase,
            error=e.error,
        ))
        raise


async def wait_bootstrap_with_streaming(
    info: InstanceMetadata,
    bus: AsyncEventBus,
    user: str,
    key_path: str,
    timeout: float = 600.0,
    ssh_timeout: float = 300.0,
    poll_interval: float = 5.0,
    log_prefix: str = "",
    port: int = 22,
) -> None:
    """Complete bootstrap wait with SSH connection and event streaming.

    Convenience function that combines wait_for_ssh() and stream_bootstrap_events().

    Args:
        info: Instance info with IP address.
        bus: Event bus to emit events to.
        user: SSH username.
        key_path: Path to SSH private key.
        timeout: Maximum time between bootstrap events.
        ssh_timeout: Maximum time to wait for SSH.
        poll_interval: Time between SSH connection attempts.
        log_prefix: Prefix for log messages.
        port: SSH port.

    Raises:
        TimeoutError: If SSH or bootstrap times out.
        BootstrapError: If bootstrap fails.
    """
    logger.info(f"{log_prefix}Waiting for bootstrap on {info.id} at {info.ip}...")

    # Wait for SSH
    transport = await wait_for_ssh(
        host=info.ip,
        user=user,
        key_path=key_path,
        timeout=ssh_timeout,
        poll_interval=poll_interval,
        port=port,
        log_prefix=log_prefix,
    )

    try:
        # Stream bootstrap events
        await stream_bootstrap_events(
            transport=transport,
            info=info,
            bus=bus,
            timeout=timeout,
            log_prefix=log_prefix,
        )
    finally:
        await transport.close()


async def install_local_skyward(
    transport: SSHTransport,
    info: InstanceMetadata,
    log_prefix: str = "",
) -> None:
    """Install local skyward wheel.

    When skyward_source='local', the bootstrap script doesn't install skyward.
    This function builds the wheel locally, uploads it, and installs it.
    The Ray server is started by bootstrap, not here.

    Args:
        transport: Connected SSH transport.
        info: Instance info (for logging).
        log_prefix: Prefix for log messages.

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

    # Execute install script (as root since uv was installed by cloud-init as root)
    logger.info(f"{log_prefix}Running wheel install script on {info.id}...")
    _, stdout, stderr = await transport.run(
        "sudo bash /tmp/.install-wheel.sh",
        timeout=180.0,
    )
    logger.debug(f"{log_prefix}Install script output:\n{stdout}")
    if stderr:
        logger.debug(f"{log_prefix}Install script stderr:\n{stderr}")

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

    Event streaming is handled separately by the EventStreamer component,
    which provides unified streaming for all providers.

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

    logger.info(f"{log_prefix}Bootstrap started on {info.id} (streaming via EventStreamer)")


__all__ = [
    "wait_for_ssh",
    "run_bootstrap_via_ssh",
    "install_local_skyward",
]
