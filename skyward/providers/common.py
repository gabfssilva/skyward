"""Common utilities shared between providers (wheel building, etc.)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger

from skyward.constants import RPYC_PORT, SKYWARD_DIR


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
    "build_wheel",
    "_build_wheel_install_script",
]
