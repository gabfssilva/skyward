"""Unified bootstrap script generator.

Single entry point for generating bootstrap scripts across all providers.
Customization happens via passing operations, not configuration flags.
"""

from __future__ import annotations

from skyward.constants import DEFAULT_PYTHON, RPYC_PORT, SKYWARD_DIR

from .compose import Op, bootstrap
from .ops import (
    apt,
    checkpoint,
    env_export,
    instance_timeout,
    pip,
    systemd,
    uv,
    install_uv,
    wait_for_port,
)
from .worker import rpyc_service_unit


def skyward_bootstrap(
    python: str = DEFAULT_PYTHON,
    pip_packages: tuple[str, ...] = (),
    apt_packages: tuple[str, ...] = (),
    env: frozenset[tuple[str, str]] = frozenset(),
    instance_timeout_secs: int | None = None,
    # Functional customization points
    preamble: tuple[Op, ...] = (),
    pip_ops: tuple[Op, ...] | None = None,
    wheel_ops: tuple[Op, ...] | None = None,
    server_ops: tuple[Op, ...] | None = None,
    postamble: tuple[Op, ...] = (),
) -> str:
    """Generate bootstrap script with functional customization.

    This is the single entry point for all providers. Customization
    happens by passing operations directly, not by setting flags.

    Args:
        python: Python version to use.
        pip_packages: Pip packages to install (used if pip_ops is None).
        apt_packages: Additional apt packages to install.
        env: Environment variables to export.
        instance_timeout_secs: Auto-shutdown timeout in seconds.
        preamble: Operations to run before everything (e.g., ssm_restart).
        pip_ops: Custom pip operations. Default: pip install packages.
        wheel_ops: Custom wheel operations. Default: SCP placeholder.
        server_ops: Custom server operations. Default: single RPyC.
        postamble: Operations to run at the end.

    Returns:
        Complete shell script string.

    Examples:
        # SSH providers (DigitalOcean, Verda) - use defaults
        script = skyward_bootstrap(
            python="3.12",
            pip_packages=("torch", "transformers"),
            apt_packages=("git",),
        )

        # AWS - customize with S3 operations
        script = skyward_bootstrap(
            python="3.12",
            apt_packages=("git",),
            preamble=(ssm_restart(),),
            pip_ops=(s3_pip_install(bucket, hash), checkpoint(".step_pip")),
            wheel_ops=(s3_wheel(bucket, key), checkpoint(".step_wheel")),
        )

        # With workers
        script = skyward_bootstrap(
            python="3.12",
            pip_packages=("torch",),
            server_ops=worker_server_ops(configs, limits, partition_script),
        )
    """
    # Default pip operations: install base + user packages
    if pip_ops is None:
        pip_ops = (
            pip("cloudpickle", "rpyc", *pip_packages),
            checkpoint(".step_pip"),
        )

    # Default wheel operations: placeholder for SCP
    if wheel_ops is None:
        wheel_ops = (
            "# Skyward wheel installed via SCP",
            checkpoint(".step_wheel"),
        )

    # Default server operations: single RPyC server
    if server_ops is None:
        server_ops = (
            systemd("skyward-rpyc", rpyc_service_unit()),
            wait_for_port(RPYC_PORT),
            checkpoint(".step_server"),
        )

    # Build operations list
    ops: list[Op] = []

    # Preamble (provider-specific, e.g., SSM restart)
    ops.extend(preamble)

    # Environment
    if env:
        ops.append(env_export(**dict(env)))
    if instance_timeout_secs:
        ops.append(instance_timeout(instance_timeout_secs))

    # UV installation
    ops.append(install_uv())
    ops.append(checkpoint(".step_uv"))

    # APT packages
    base_apt = ("python3", "python3-venv", "curl", "ca-certificates")
    ops.append(apt(*base_apt, *apt_packages))
    ops.append(checkpoint(".step_apt"))

    # Virtual environment
    ops.append(uv(python, f"{SKYWARD_DIR}/.venv"))
    ops.append(f"source {SKYWARD_DIR}/.venv/bin/activate")

    # Pip packages (customizable)
    ops.extend(pip_ops)

    # Wheel installation (customizable)
    ops.extend(wheel_ops)

    # Server setup (customizable)
    ops.extend(server_ops)

    # Postamble
    ops.extend(postamble)

    # Final checkpoint
    ops.append(checkpoint(".ready"))

    return bootstrap(*ops)
