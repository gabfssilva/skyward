"""NVIDIA MIG plugin — Multi-Instance GPU partitioning."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from skyward.plugins.plugin import Plugin
from skyward.providers.bootstrap import phase_simple

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.runtime import InstanceInfo
    from skyward.api.spec import Image
    from skyward.providers.bootstrap import Op


def mig(profile: str) -> Plugin:
    """NVIDIA MIG plugin for GPU partitioning.

    Enables MIG mode, creates partitions during bootstrap, and assigns
    each subprocess its own MIG device via CUDA_VISIBLE_DEVICES.

    Parameters
    ----------
    profile
        MIG profile name (e.g. ``"3g.40gb"``, ``"1g.10gb"``).
    """

    def transform(image: Image, cluster: Cluster[Any]) -> Image:
        return replace(image, env={**image.env, "NVIDIA_VISIBLE_DEVICES": "all"})

    def bootstrap_factory(cluster: Cluster[Any]) -> tuple[Op, ...]:
        concurrency = cluster.spec.worker.concurrency
        cgi_commands = "\n".join(
            f"nvidia-smi mig -cgi {profile} -C" for _ in range(concurrency)
        )
        return (
            phase_simple(
                "nvidia-mig",
                "nvidia-smi -mig 1",
                cgi_commands,
            ),
        )

    @contextmanager
    def around(info: InstanceInfo) -> Iterator[None]:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
        mig_uuids = re.findall(r"(MIG-[0-9a-fA-F-]+)", result.stdout)
        os.environ["CUDA_VISIBLE_DEVICES"] = mig_uuids[info.worker]
        yield

    return (
        Plugin.create("mig")
        .with_image_transform(transform)
        .with_bootstrap(bootstrap_factory)
        .with_around_process(around)
    )
