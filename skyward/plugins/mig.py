"""NVIDIA MIG plugin — Multi-Instance GPU partitioning.

Standard ``nvidia-smi --query-gpu`` reports metrics for the **physical GPU**,
not individual MIG slices.  ``nvidia-smi -i MIG-<uuid>`` is also unsupported
(outputs *"No devices were found"*).

This plugin replaces the default GPU metrics with collectors that try two
strategies in order:

1. **DCGM** (``dcgmi dmon``) — per-GPU-Instance metrics.  Requires the
   ``datacenter-gpu-manager`` package and ``nv-hostengine`` running.
   Installed best-effort during bootstrap.
2. **nvidia-smi** plain-text parsing — falls back to scraping the MIG
   devices table from ``nvidia-smi`` output for per-slice memory.

Temperature always comes from ``nvidia-smi`` (one die, shared across slices).

On Ampere (A100/A30) per-slice GPU utilization is only available through DCGM
field 1002 (SM Active).  If DCGM is not installed, utilization falls back to
the physical GPU ``utilization.gpu`` field.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from skyward.observability.metrics import Metric
from skyward.plugins.plugin import Plugin
from skyward.providers.bootstrap import phase_simple

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.runtime import InstanceInfo
    from skyward.api.spec import Image
    from skyward.providers.bootstrap import Op

# nvidia-smi field names used by the default GPU metrics.
_GPU_METRIC_NAMES = frozenset({"gpu_util", "gpu_mem_mb", "gpu_mem_total_mb", "gpu_temp"})

# DCGM field IDs.
_DCGM_SM_ACTIVE = 1002  # DCGM_FI_PROF_SM_ACTIVE — compute utilization (%)
_DCGM_FB_USED = 252  # DCGM_FI_DEV_FB_USED — framebuffer memory used (MiB)
_DCGM_FB_TOTAL = 250  # DCGM_FI_DEV_FB_TOTAL — framebuffer memory total (MiB)

# Helper: discover GPU-Instance entity IDs from ``dcgmi discovery -c``.
_MIG_IDS = (
    "dcgmi discovery -c 2>/dev/null "
    "| grep -oP 'GPU Instance \\(EntityID: \\K\\d+' "
    "| sed 's/^/i:/' | paste -sd,"
)


def _dcgm_query(field_id: int) -> str:
    """One-shot DCGM query for *field_id* across all GPU-Instances.

    Discovers GPU-Instance entity IDs dynamically, then runs
    ``dcgmi dmon -e <field> -i i:0,i:1,... -c 1`` and parses
    only ``GPU-I`` rows.
    """
    return (
        f"ids=$({_MIG_IDS}); "
        f"[ -n \"$ids\" ] && dcgmi dmon -e {field_id} -i $ids -c 1 -d 100 2>/dev/null "
        "| awk '/GPU-I/{print $NF}' "
        "| grep -E '^[0-9]'"
    )


# nvidia-smi plain-text fallbacks — parse the MIG devices table.
# The table has two memory lines per instance (Shared + BAR1); we pick
# odd lines (Shared = actual VRAM).
_SMI_MIG_MEM_USED = (
    "nvidia-smi 2>/dev/null "
    "| sed -n '/MIG devices/,/Processes/p' "
    "| grep -oP '\\d+MiB\\s*/\\s*\\d+MiB' "
    "| awk 'NR%2==1{split($0,a,/MiB/); gsub(/ /,\"\",a[1]); print a[1]}'"
)

_SMI_MIG_MEM_TOTAL = (
    "nvidia-smi 2>/dev/null "
    "| sed -n '/MIG devices/,/Processes/p' "
    "| grep -oP '\\d+MiB\\s*/\\s*\\d+MiB' "
    "| awk 'NR%2==1{split($0,a,\"/ \"); gsub(/MiB.*/,\"\",a[2]); gsub(/ /,\"\",a[2]); print a[2]}'"
)

_SMI_GPU_UTIL = (
    "nvidia-smi --query-gpu=utilization.gpu "
    "--format=csv,noheader,nounits 2>/dev/null "
    "| tr -d ' ' | grep -E '^[0-9]'"
)


def _with_fallback(dcgm_cmd: str, smi_cmd: str) -> str:
    """Try DCGM; if empty output, fall back to nvidia-smi."""
    return (
        f"_r=$({dcgm_cmd}); "
        "if [ -n \"$_r\" ]; then echo \"$_r\"; "
        f"else {smi_cmd}; fi"
    )


def _mig_metrics(gpu_interval: float = 3) -> tuple[Metric, ...]:
    """GPU metrics with DCGM -> nvidia-smi fallback per metric."""
    return (
        Metric(
            name="gpu_util",
            command=_with_fallback(
                _dcgm_query(_DCGM_SM_ACTIVE),
                _SMI_GPU_UTIL,
            ),
            interval=gpu_interval,
            multi=True,
        ),
        Metric(
            name="gpu_mem_mb",
            command=_with_fallback(
                _dcgm_query(_DCGM_FB_USED),
                _SMI_MIG_MEM_USED,
            ),
            interval=gpu_interval,
            multi=True,
        ),
        Metric(
            name="gpu_mem_total_mb",
            command=_with_fallback(
                _dcgm_query(_DCGM_FB_TOTAL),
                _SMI_MIG_MEM_TOTAL,
            ),
            interval=60.0,
            multi=True,
        ),
        Metric(
            name="gpu_temp",
            command=(
                "nvidia-smi --query-gpu=temperature.gpu "
                "--format=csv,noheader,nounits 2>/dev/null | tr -d ' '"
            ),
            interval=gpu_interval,
            multi=True,
        ),
    )


_DCGM_INSTALL = """\
if ! command -v dcgmi >/dev/null 2>&1; then
    apt-get install -y -qq datacenter-gpu-manager 2>/dev/null || true
fi
if command -v nv-hostengine >/dev/null 2>&1 && ! pgrep -x nv-hostengine >/dev/null 2>&1; then
    nv-hostengine 2>/dev/null || true
fi"""


def mig(profile: str) -> Plugin:
    """NVIDIA MIG plugin for GPU partitioning.

    Enables MIG mode, creates partitions during bootstrap, attempts to install
    DCGM for per-slice metrics (with nvidia-smi fallback), and assigns each
    subprocess its own MIG device via CUDA_VISIBLE_DEVICES.

    Parameters
    ----------
    profile
        MIG profile name (e.g. ``"3g.40gb"``, ``"1g.10gb"``).
    """

    def transform(image: Image, cluster: Cluster[Any]) -> Image:
        existing = image.metrics or ()
        non_gpu = tuple(m for m in existing if m.name not in _GPU_METRIC_NAMES)
        gpu_interval = next(
            (m.interval for m in existing if m.name in _GPU_METRIC_NAMES), 3.0,
        )
        return replace(
            image,
            env={**image.env, "NVIDIA_VISIBLE_DEVICES": "all"},
            metrics=(*non_gpu, *_mig_metrics(gpu_interval)),
        )

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
            phase_simple("nvidia-dcgm", _DCGM_INSTALL),
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
