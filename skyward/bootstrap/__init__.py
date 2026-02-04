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
    apt,
    cd,
    checkpoint,
    emit_bootstrap_complete,
    env_export,
    file,
    grid_driver,
    install_uv,
    instance_timeout,
    mkdir,
    phase,
    phase_simple,
    pip,
    shell,
    shell_vars,
    start_metrics,
    stop_metrics,
    uv,
    uv_add,
    uv_init,
    wait_for_port,
)

# Ray operations
from .ray import (
    ray_install,
    ray_service,
    server_ops as ray_server_ops,
)

__all__ = [
    # Core types
    "Op",
    "bootstrap",
    "resolve",
    # Core operations
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
    "shell",
    "cd",
    "wait_for_port",
    "env_export",
    "instance_timeout",
    "shell_vars",
    "start_metrics",
    "stop_metrics",
    # AWS operations
    "grid_driver",
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
    # Ray operations
    "ray_install",
    "ray_service",
    "ray_server_ops",
]
