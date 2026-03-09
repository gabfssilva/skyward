"""Specification dataclasses for pool and image configuration.

Re-exports all types from ``skyward.api.spec`` and provides
the ``generate_bootstrap()`` standalone function.
"""

from __future__ import annotations

from skyward.api.spec import DEFAULT_IMAGE as DEFAULT_IMAGE
from skyward.api.spec import AllocationStrategy as AllocationStrategy
from skyward.api.spec import Architecture as Architecture
from skyward.api.spec import Image as Image
from skyward.api.spec import Options as Options
from skyward.api.spec import PipIndex as PipIndex
from skyward.api.spec import PoolSpec as PoolSpec
from skyward.api.spec import PoolState as PoolState
from skyward.api.spec import SelectionStrategy as SelectionStrategy
from skyward.api.spec import SkywardSource as SkywardSource
from skyward.api.spec import Spec as Spec
from skyward.api.spec import SpecKwargs as SpecKwargs
from skyward.api.spec import Volume as Volume
from skyward.api.spec import Worker as Worker
from skyward.api.spec import WorkerExecutor as WorkerExecutor
from skyward.api.spec import _detect_skyward_source as _detect_skyward_source
from skyward.api.spec import _SpecRequired as _SpecRequired
from skyward.providers.bootstrap import (
    Op,
    apt,
    bootstrap,
    emit_bootstrap_complete,
    env_export,
    install_uv,
    instance_timeout,
    phase,
    phase_simple,
    shell_vars,
    start_metrics,
    uv_add,
    uv_configure_indexes,
    uv_init,
)


def generate_bootstrap(
    image: Image,
    *,
    ttl: int = 0,
    shutdown_command: str = "shutdown -h now",
    preamble: Op | None = None,
    postamble: Op | None = None,
) -> str:
    """Generate bootstrap script for cloud-init/user_data."""
    ops: list[Op | None] = [
        instance_timeout(ttl, shutdown_command=shutdown_command) if ttl else None,
        preamble,
    ]

    if image.shell_vars or image.env:
        env_ops: list[Op] = []
        if image.shell_vars:
            env_ops.append(shell_vars(**image.shell_vars))
        if image.env:
            env_ops.append(env_export(**image.env))
        ops.append(phase_simple("env", *env_ops))

    ops.extend([
        phase("apt", apt("curl", "git", *image.apt)),
        phase_simple(
            "uv",
            install_uv(),
            uv_init(image.python, name="skyward-bootstrap"),
            uv_configure_indexes(image.pip_indexes),
        ),
        phase("deps", uv_add("cloudpickle", "lz4", *image.pip)),
    ])

    match image.skyward_source:
        case "github":
            ops.append(phase("skyward", uv_add("git+https://github.com/gabfssilva/skyward.git")))
        case "pypi":
            ops.append(phase("skyward", uv_add("skyward")))

    ops.append(postamble)
    ops.append(emit_bootstrap_complete())
    ops.append(start_metrics())

    return bootstrap(*ops, metrics=image.metrics)
