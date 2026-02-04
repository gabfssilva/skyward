"""Unified bootstrap script generator.

Single entry point for generating bootstrap scripts across all providers.
Customization happens via passing operations, not configuration flags.
"""

from __future__ import annotations

from ..constants import DEFAULT_PYTHON, SKYWARD_DIR

from .compose import Op, bootstrap
from .ops import (
    apt,
    checkpoint,
    env_export,
    install_uv,
    instance_timeout,
    pip,
    uv,
)


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
        preamble: Operations to run before everything (e.g., grid_driver, inject_ssh_key).
        pip_ops: Custom pip operations. Default: pip install packages.
        wheel_ops: Custom wheel operations. Default: SCP placeholder.
        server_ops: Custom server operations. Default: Ray head (node 0).
        postamble: Operations to run at the end.

    Returns:
        Complete shell script string.

    Examples:
        # SSH providers (DigitalOcean, Verda, AWS) - use defaults
        script = skyward_bootstrap(
            python="3.12",
            pip_packages=("torch", "transformers"),
            apt_packages=("git",),
        )

        # AWS - customize with S3 operations
        script = skyward_bootstrap(
            python="3.12",
            apt_packages=("git",),
            pip_ops=(s3_pip_install(bucket, hash), checkpoint(".step_pip")),
            wheel_ops=(s3_wheel(bucket, key), checkpoint(".step_wheel")),
        )

        # With Ray (requires passing server_ops from provider)
        from skyward.bootstrap.ray import server_ops as ray_server_ops
        script = skyward_bootstrap(
            python="3.12",
            pip_packages=("torch",),
            server_ops=ray_server_ops(node_id=0, head_ip=None, num_gpus=1),
        )
    """
    # Default pip operations: install base + user packages
    # Note: Ray is installed via server_ops, not here
    if pip_ops is None:
        pip_ops = (
            pip("cloudpickle", *pip_packages),
            checkpoint(".step_pip"),
        )

    # Default wheel operations: placeholder for SCP
    if wheel_ops is None:
        wheel_ops = (
            "# Skyward wheel installed via SCP",
            checkpoint(".step_wheel"),
        )

    # Default server operations: Ray head/worker
    # NOTE: Provider must pass server_ops with correct node_id and head_ip
    # This default only works for single-node clusters (head only)
    if server_ops is None:
        from .ray import server_ops as ray_server_ops
        server_ops = ray_server_ops(
            node_id=0,
            head_ip=None,
            num_gpus=1,
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
