"""Declarative bootstrap script DSL.

This module provides a functional DSL for generating shell scripts
used in cloud-init user_data and startup scripts.

Example:
    >>> from skyward.bootstrap import bootstrap, apt, pip, checkpoint, when
    >>>
    >>> script = bootstrap(
    ...     apt("python3", "curl"),
    ...     checkpoint(".step_apt"),
    ...     when("command -v nvidia-smi",
    ...         "nvidia-smi --query-gpu=name",
    ...     ),
    ...     pip("torch", "transformers"),
    ...     checkpoint(".ready"),
    ... )
"""

from __future__ import annotations

# Core types and composition
from .compose import (
    Op,
    bootstrap,
    resolve,
)

# Control flow
from .control import (
    and_then,
    capture,
    for_each,
    group,
    or_else,
    subshell,
    unless,
    var,
    when,
)

# Core operations
from .ops import (
    activate,
    apt,
    cd,
    checkpoint,
    emit_bootstrap_complete,
    env_export,
    file,
    grid_driver,
    inject_ssh_key,
    install_uv,
    instance_timeout,
    mkdir,
    nohup_service,
    phase,
    phase_simple,
    pip,
    s3_pip_install,
    s3_wheel,
    shell,
    shell_vars,
    start_metrics,
    stop_metrics,
    systemd,
    systemd_template,
    uv,
    uv_add,
    uv_init,
    wait_for_port,
)

# Unified generator
from .unified import skyward_bootstrap

# Worker operations (legacy, for MIG support)
from .worker import (
    cgroups,
    mig_setup,
    start_workers,
    wait_for_workers,
    worker_envs,
    worker_server_ops,
    worker_service_template,
    worker_service_unit,
)

# Ray operations
from .ray import (
    ray_head_start,
    ray_install,
    ray_service,
    ray_worker_start,
    server_ops as ray_server_ops,
)

__all__ = [
    # Core types
    "Op",
    "bootstrap",
    "resolve",
    # Core operations
    "activate",
    "apt",
    "pip",
    "uv",
    "install_uv",
    "uv_add",
    "uv_init",
    "checkpoint",
    "emit_bootstrap_complete",
    "mkdir",
    "file",
    "phase",
    "phase_simple",
    "systemd",
    "systemd_template",
    "nohup_service",
    "shell",
    "cd",
    "wait_for_port",
    "env_export",
    "instance_timeout",
    "inject_ssh_key",
    "shell_vars",
    "start_metrics",
    "stop_metrics",
    # AWS operations
    "grid_driver",
    "s3_pip_install",
    "s3_wheel",
    # Control flow
    "capture",
    "var",
    "when",
    "unless",
    "for_each",
    "and_then",
    "or_else",
    "group",
    "subshell",
    # Worker operations (legacy, for MIG support)
    "cgroups",
    "worker_envs",
    "mig_setup",
    "worker_service_unit",
    "worker_service_template",
    "start_workers",
    "wait_for_workers",
    "worker_server_ops",
    # Ray operations
    "ray_install",
    "ray_head_start",
    "ray_worker_start",
    "ray_service",
    "ray_server_ops",
    # Unified generator
    "skyward_bootstrap",
]
