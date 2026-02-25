"""NVIDIA MPS plugin — Multi-Process Service for concurrent GPU sharing."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from skyward.plugins.plugin import Plugin
from skyward.providers.bootstrap import phase_simple

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.spec import Image
    from skyward.providers.bootstrap import Op


def mps(
    active_thread_percentage: int | None = None,
    pinned_memory_limit: str | None = None,
) -> Plugin:
    """NVIDIA MPS plugin for concurrent GPU sharing.

    Starts the MPS daemon during bootstrap and configures environment
    variables so multiple CUDA processes share a single GPU context
    with reduced overhead.

    Parameters
    ----------
    active_thread_percentage
        Maximum percentage of GPU compute a single client can use (1-100).
    pinned_memory_limit
        Per-device pinned memory limit (e.g. ``"0=2G"`` for 2 GB on device 0).
    """

    def transform(image: Image, cluster: Cluster[Any]) -> Image:
        env = {
            **image.env,
            "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
            "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-mps-log",
        }
        if active_thread_percentage is not None:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(active_thread_percentage)
        if pinned_memory_limit is not None:
            env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = pinned_memory_limit
        return replace(image, env=env)

    def bootstrap_factory(cluster: Cluster[Any]) -> tuple[Op, ...]:
        return (
            phase_simple(
                "nvidia-mps",
                "mkdir -p /tmp/nvidia-mps /tmp/nvidia-mps-log",
                "nvidia-cuda-mps-control -d",
            ),
        )

    return (
        Plugin.create("mps")
        .with_image_transform(transform)
        .with_bootstrap(bootstrap_factory)
    )
