"""The MPS plugin, checked without a GPU: the env it sets and how it travels.

The daemon start is the one side effect that needs hardware, so it is stubbed to a
no-op; everything else is the worker environment ``setup`` leaves behind.
"""

import os
import subprocess

from skyward.worker.plugins.mps import Mps
from skyward.shared.schemas import PluginRef
from skyward.worker.api import Info

INFO = Info(node="n", compute="c", rank=0, peers=("host",))


def test_it_points_the_env_at_the_pipe_and_log_directories(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)

    with Mps().setup(INFO):
        assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/nvidia-mps"
        assert os.environ["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/nvidia-mps-log"


def test_it_exports_the_limits_only_when_they_are_set(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.delenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", raising=False)
    monkeypatch.delenv("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", raising=False)

    plugin = Mps(active_thread_percentage=40, pinned_memory_limit="0=2G")
    with plugin.setup(INFO):
        assert os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "40"
        assert os.environ["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=2G"


def test_it_leaves_the_limits_unset_by_default(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.delenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", raising=False)
    monkeypatch.delenv("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", raising=False)

    with Mps().setup(INFO):
        assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in os.environ
        assert "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" not in os.environ


def test_a_missing_control_binary_does_not_crash_the_worker(monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("nvidia-cuda-mps-control")

    monkeypatch.setattr(subprocess, "run", absent)

    try:
        with Mps().setup(INFO):
            pass
    except FileNotFoundError:
        raise AssertionError("setup must swallow a missing control binary")


def test_it_travels_as_its_kind_and_its_fields():
    ref = Mps(active_thread_percentage=50).ref()

    assert ref == PluginRef(
        kind="mps",
        params={"active_thread_percentage": 50, "pinned_memory_limit": None},
    )
