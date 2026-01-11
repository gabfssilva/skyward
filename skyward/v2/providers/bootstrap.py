"""Shared bootstrap streaming utilities for v2 providers.

Provides async helper functions for waiting on bootstrap completion
via JSONL event streaming.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from skyward.v2.bus import AsyncEventBus
from skyward.v2.events import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapFailed,
    BootstrapPhase,
    InstanceInfo,
    Log,
    Metric,
)
from skyward.v2.transport import (
    BootstrapError,
    RawBootstrapCommand,
    RawBootstrapConsole,
    RawBootstrapPhase,
    RawLogEvent,
    RawMetricEvent,
    SSHTransport,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Continuous Streaming
# =============================================================================


async def stream_instance_events(
    transport: SSHTransport,
    info: InstanceInfo,
    bus: AsyncEventBus,
    timeout: float = 600.0,
    log_prefix: str = "",
) -> None:
    """Stream all events from an instance indefinitely.

    Runs until cancelled or connection drops. Emits events to bus:
    - BootstrapPhase, BootstrapCommand, BootstrapConsole (during bootstrap)
    - Log (during execution)
    - Metric (throughout)

    Args:
        transport: Connected SSH transport.
        info: Instance info (for event enrichment).
        bus: Event bus to emit events to.
        timeout: Maximum time between events (adaptive).
        log_prefix: Prefix for log messages.
    """
    logger.info(f"{log_prefix}Starting event streaming for {info.id}")

    try:
        async for raw_event in transport.stream_events(timeout=timeout):
            match raw_event:
                case RawBootstrapConsole(content=content, stream=stream):
                    bus.emit(BootstrapConsole(
                        instance=info,
                        content=content,
                        stream=stream,
                    ))
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
                    if event == "started":
                        logger.info(f"{log_prefix}Phase '{phase}' started")
                    elif event == "completed":
                        elapsed_str = f" ({elapsed:.1f}s)" if elapsed else ""
                        logger.info(f"{log_prefix}Phase '{phase}' completed{elapsed_str}")
                    elif event == "failed":
                        logger.error(f"{log_prefix}Phase '{phase}' FAILED: {error}")
                        bus.emit(BootstrapFailed(
                            instance=info,
                            phase=phase,
                            error=error or "unknown",
                        ))

                case RawBootstrapCommand(command=command):
                    bus.emit(BootstrapCommand(
                        instance=info,
                        command=command,
                    ))
                    display = command[:80] + "..." if len(command) > 80 else command
                    logger.debug(f"{log_prefix}Running: {display}")

                case RawLogEvent(content=content, stream=stream):
                    bus.emit(Log(
                        instance=info,
                        line=content,
                        stream=stream,
                    ))

                case RawMetricEvent(name=name, value=value, ts=ts):
                    bus.emit(Metric(
                        instance=info,
                        name=name,
                        value=value,
                        timestamp=ts,
                    ))
                    logger.debug(f"{log_prefix}metric {name}={value}")

    except asyncio.CancelledError:
        logger.debug(f"{log_prefix}Streaming cancelled for {info.id}")
    except TimeoutError:
        logger.warning(f"{log_prefix}Streaming timeout for {info.id}")
    except Exception as e:
        logger.error(f"{log_prefix}Streaming error for {info.id}: {e}")
    finally:
        logger.debug(f"{log_prefix}Streaming ended for {info.id}")


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
    info: InstanceInfo,
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
    info: InstanceInfo,
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
    info: InstanceInfo,
    env: dict[str, str] | None = None,
    use_systemd: bool = True,
    rpyc_timeout: float = 30.0,
    log_prefix: str = "",
) -> None:
    """Install local skyward wheel and start RPyC server.

    When skyward_source='local', the bootstrap script doesn't install skyward.
    This function builds the wheel locally, uploads it, installs it, and starts
    the RPyC server.

    Args:
        transport: Connected SSH transport.
        info: Instance info (for logging).
        env: Environment variables to pass to the RPyC service.
        use_systemd: If True, use systemd (VMs). If False, use nohup (Docker).
        rpyc_timeout: Timeout for RPyC server to become ready.
        log_prefix: Prefix for log messages.

    Raises:
        RuntimeError: If wheel build or installation fails.
        TimeoutError: If RPyC server doesn't start in time.
    """
    from skyward.core.constants import RPYC_PORT
    from skyward.providers.common import build_wheel, _build_wheel_install_script

    # Build wheel locally
    logger.info(f"{log_prefix}Building local skyward wheel...")
    wheel_path = build_wheel()

    # Build install script
    install_script = _build_wheel_install_script(
        wheel_name=wheel_path.name,
        env=env,
        use_systemd=use_systemd,
    )

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

    # Wait for RPyC to be ready by connecting directly via SSH tunnel
    logger.info(f"{log_prefix}Waiting for RPyC server on {info.id}...")

    import socket

    import rpyc

    from skyward.v2.retry import retry

    # Find a free local port for the tunnel
    def find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    local_port = find_free_port()
    ssh_conn = transport._require_connection()

    # Create local port forwarding: local_port -> remote localhost:RPYC_PORT
    listener = await ssh_conn.forward_local_port("", local_port, "127.0.0.1", RPYC_PORT)
    logger.debug(f"{log_prefix}SSH tunnel: localhost:{local_port} -> remote:localhost:{RPYC_PORT}")

    try:
        @retry(max_attempts=int(rpyc_timeout * 2), base_delay=0.5)
        async def connect_rpyc() -> None:
            def try_connect() -> None:
                c = rpyc.connect(
                    "127.0.0.1",
                    port=local_port,
                    config={"allow_pickle": True, "sync_request_timeout": 5},
                )
                result = c.root.ping()
                c.close()
                if result != "pong":
                    raise ConnectionError("RPyC ping failed")

            await asyncio.get_event_loop().run_in_executor(None, try_connect)

        await connect_rpyc()
        logger.info(f"{log_prefix}RPyC server ready on {info.id}")

    except Exception as e:
        # Timeout - gather debug info
        _, ps_out, _ = await transport.run("ps aux | grep 'skyward.rpc' | grep -v grep || true")
        _, log_out, _ = await transport.run("tail -30 /var/log/skyward-rpyc.log 2>/dev/null || echo 'No log file'")

        if ps_out.strip():
            logger.error(f"{log_prefix}RPyC process found but not listening:\n{ps_out}")
            logger.error(f"{log_prefix}RPyC server log:\n{log_out}")
        else:
            logger.error(f"{log_prefix}RPyC process not running. Log output:\n{log_out}")

        raise TimeoutError(f"RPyC server not ready on {info.id} after {rpyc_timeout}s") from e

    finally:
        listener.close()
        await listener.wait_closed()


__all__ = [
    "wait_for_ssh",
    "stream_bootstrap_events",
    "wait_bootstrap_with_streaming",
    "install_local_skyward",
]
