"""Declarative bootstrap script DSL.

This module provides a functional DSL for generating shell scripts
used in cloud-init user_data and startup scripts.

Example:
    >>> from skyward.bootstrap import bootstrap, apt, pip, checkpoint
    >>>
    >>> script = bootstrap(
    ...     apt("python3", "curl"),
    ...     checkpoint(".step_apt"),
    ...     pip("torch", "transformers"),
    ...     checkpoint(".ready"),
    ... )
"""

from __future__ import annotations

# Core types and composition
from .compose import (
    EMIT_SH_PATH,
    Op,
    bootstrap,
    resolve,
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

# Casty operations
from .casty import (
    casty_install,
    casty_service,
    server_ops as casty_server_ops,
)

__all__ = [
    # Core types
    "EMIT_SH_PATH",
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
    # Casty operations
    "casty_install",
    "casty_service",
    "casty_server_ops",
]
