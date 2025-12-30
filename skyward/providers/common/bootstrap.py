"""Shared bootstrap logic for all providers.

This module provides:
- Checkpoint definitions for tracking bootstrap progress
- Generic wait_for_bootstrap() that works with any command runner
- Base user data script generation
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
)

from skyward.callback import emit
from skyward.constants import (
    DEFAULT_PYTHON,
    RPYC_PORT,
    SKYWARD_DIR,
    UV_INSTALL_URL,
)
from skyward.events import BootstrapProgress


# =============================================================================
# Checkpoint Definitions
# =============================================================================


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Bootstrap checkpoint for progress tracking."""

    file: str  # e.g., ".step_uv"
    name: str  # e.g., "uv" (for display)


# Standard checkpoints in order of execution
# Providers can extend this with provider-specific checkpoints
CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(".step_uv", "uv"),
    Checkpoint(".step_apt", "apt"),
    Checkpoint(".step_pip", "pip deps"),
    Checkpoint(".step_wheel", "skyward"),
    Checkpoint(".step_server", "server"),
)

# Additional checkpoints for worker isolation
WORKER_CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(".step_partition", "partition"),
    Checkpoint(".step_cgroups", "cgroups"),
    Checkpoint(".step_workers", "workers"),
)


# =============================================================================
# Wait for Bootstrap
# =============================================================================


class BootstrapNotReadyError(Exception):
    """Raised when bootstrap check should be retried."""


class BootstrapFailedError(Exception):
    """Raised when bootstrap script failed with an error."""

    def __init__(self, error_msg: str):
        self.error_msg = error_msg
        super().__init__(error_msg)


# Protocol for running commands on remote instance
CommandRunner = Callable[[str], str]  # command -> stdout


def wait_for_bootstrap(
    run_command: CommandRunner,
    instance_id: str,
    timeout: int = 300,
    extra_checkpoints: tuple[Checkpoint, ...] = (),
) -> None:
    """Wait for instance bootstrap with progress tracking.

    This is provider-agnostic - the provider injects the command runner
    (SSM for AWS, SSH for DigitalOcean, etc.).

    Args:
        run_command: Function that runs a shell command and returns stdout.
                    Should raise on failure (non-zero exit, timeout, etc.).
        instance_id: Instance ID for logging and events.
        timeout: Maximum time to wait in seconds.
        extra_checkpoints: Additional provider-specific checkpoints to track.

    Raises:
        BootstrapFailedError: If bootstrap script failed (.error file exists).
        RuntimeError: If bootstrap times out.
    """
    all_checkpoints = CHECKPOINTS + extra_checkpoints

    # Build check command - checks all files at once
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

        # Check for error file first
        if "---ERROR---" in stdout:
            error_msg = stdout.split("---ERROR---", 1)[1].strip()
            raise BootstrapFailedError(error_msg)

        found_files = set(stdout.split("\n")) if stdout else set()

        # Emit events for completed checkpoints
        for checkpoint in all_checkpoints:
            if checkpoint.file in found_files and checkpoint.name not in completed_steps:
                emit(
                    BootstrapProgress(
                        instance_id=instance_id,
                        step=checkpoint.name,
                    )
                )
                completed_steps.add(checkpoint.name)

        # Check if bootstrap complete
        if ".ready" in found_files:
            return  # Success!

        # Not ready yet - retry
        raise BootstrapNotReadyError()

    try:
        _poll_bootstrap()
    except BootstrapFailedError as e:
        raise RuntimeError(f"Bootstrap failed on {instance_id}:\n{e.error_msg}") from None
    except (RetryError, BootstrapNotReadyError) as e:
        raise RuntimeError(f"Bootstrap timed out on {instance_id}") from e


# =============================================================================
# User Data Script Generation
# =============================================================================


def generate_base_script(
    python: str = DEFAULT_PYTHON,
    pip: tuple[str, ...] = (),
    apt: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    instance_timeout: int | None = None,
    preamble: str = "",
    postamble: str = "",
    pip_extra_index_url: str | None = None,
    worker_bootstrap: str = "",
) -> str:
    """Generate base bootstrap script for user_data.

    This creates a standardized bootstrap script that:
    1. Sets up error handling and logging
    2. Installs UV
    3. Installs apt packages
    4. Creates venv and installs pip packages
    5. Sets up systemd service for RPyC server (or multiple workers)

    Providers can inject custom sections via preamble/postamble.

    Args:
        python: Python version (e.g., "3.13").
        pip: Pip packages to install.
        apt: Apt packages to install.
        env: Environment variables to set.
        instance_timeout: Auto-terminate after N seconds (0 = disabled).
        preamble: Shell script to run BEFORE standard bootstrap.
        postamble: Shell script to run AFTER standard bootstrap (before .ready).
        pip_extra_index_url: Extra pip index URL.
        worker_bootstrap: Optional script for multi-worker setup. When provided,
            replaces the single RPyC server with multiple worker services.

    Returns:
        Complete bash script as string.
    """
    pip_packages = " ".join(pip) if pip else ""
    apt_packages = " ".join(apt) if apt else ""
    env_exports = "\n".join(f'export {k}="{v}"' for k, v in (env or {}).items())
    extra_index = f"--extra-index-url {pip_extra_index_url}" if pip_extra_index_url else ""

    script = f"""#!/bin/bash
set -e

mkdir -p {SKYWARD_DIR}

# Redirect all output to log file
exec > {SKYWARD_DIR}/bootstrap.log 2>&1

# On error, write to .error file
trap 'echo "Command failed: $BASH_COMMAND" > {SKYWARD_DIR}/.error; echo "Exit code: $?" >> {SKYWARD_DIR}/.error; echo "--- Output ---" >> {SKYWARD_DIR}/.error; tail -50 {SKYWARD_DIR}/bootstrap.log >> {SKYWARD_DIR}/.error' ERR

export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.local/bin:$PATH"
{env_exports}

# Auto-terminate after timeout
SKYWARD_INSTANCE_TIMEOUT="{instance_timeout or ''}"
if [ -n "$SKYWARD_INSTANCE_TIMEOUT" ] && [ "$SKYWARD_INSTANCE_TIMEOUT" -gt 0 ]; then
    (sleep $SKYWARD_INSTANCE_TIMEOUT && shutdown -h now) &
fi

{preamble}

# Install UV
if ! command -v uv &> /dev/null; then
    curl -LsSf {UV_INSTALL_URL} | sh
fi
touch {SKYWARD_DIR}/.step_uv

# Install apt packages
apt-get update -qq
apt-get install -y -qq python3 python3-venv curl ca-certificates
"""

    if apt_packages:
        script += f"apt-get install -y -qq {apt_packages}\n"

    script += f"""touch {SKYWARD_DIR}/.step_apt

# Create venv and install dependencies
cd {SKYWARD_DIR}
uv venv --python {python} venv
source venv/bin/activate

# Install base dependencies
uv pip install cloudpickle rpyc
"""

    if pip_packages:
        script += f"uv pip install {pip_packages} {extra_index}\n"

    script += f"""touch {SKYWARD_DIR}/.step_pip

# Skyward wheel installation (provider-specific in postamble)
touch {SKYWARD_DIR}/.step_wheel
"""

    # Either setup multiple workers or single RPyC server
    if worker_bootstrap:
        # Multi-worker setup: cgroups + systemd workers
        script += f"""
{postamble}

# Worker isolation bootstrap
{worker_bootstrap}

touch {SKYWARD_DIR}/.step_server
touch {SKYWARD_DIR}/.ready
"""
    else:
        # Single RPyC server (default)
        script += f"""
# Setup systemd service for RPyC server
cat > /etc/systemd/system/skyward-rpyc.service << 'SERVICEEOF'
[Unit]
Description=Skyward RPyC Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={SKYWARD_DIR}
Environment="PATH={SKYWARD_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart={SKYWARD_DIR}/venv/bin/python -m skyward.rpc
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl daemon-reload
systemctl enable skyward-rpyc

{postamble}

# Start RPyC server
systemctl start skyward-rpyc

# Wait for server to be ready
for i in $(seq 1 60); do
    if ss -tlnp 2>/dev/null | grep -q ':{RPYC_PORT} ' || netstat -tlnp 2>/dev/null | grep -q ':{RPYC_PORT} '; then
        break
    fi
    sleep 0.5
done
touch {SKYWARD_DIR}/.step_server

touch {SKYWARD_DIR}/.ready
"""
    return script


def generate_skyward_install_script(
    wheel_source: str,
    venv_path: str = f"{SKYWARD_DIR}/venv",
) -> str:
    """Generate script to install skyward wheel.

    Args:
        wheel_source: Path or URL to wheel file.
        venv_path: Path to virtualenv.

    Returns:
        Shell script fragment.
    """
    return f"""
# Install skyward wheel
{venv_path}/bin/pip install {wheel_source}
touch {SKYWARD_DIR}/.step_wheel
"""
