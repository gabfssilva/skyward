"""Provider sanity tests — full-stack e2e with real cloud GPUs.

Each test provisions 2 nodes with a cheap GPU, runs a simple ML workload
(matrix multiply on CUDA), verifies all nodes respond, and tears down.

Run individually:
    uv run pytest -m aws --no-header -q
    uv run pytest -m runpod --no-header -q
    uv run pytest -m vastai --no-header -q
    uv run pytest -m verda --no-header -q

Run all sanity tests:
    uv run pytest -m sanity --no-header -q
"""

from __future__ import annotations

import pytest

import skyward as sky

NODES = 2
TIMEOUT = 600


@sky.compute
def gpu_matmul(size: int = 512) -> dict:
    """Multiply two matrices on GPU and return diagnostics."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    c = a @ b

    info = sky.instance_info()

    return {
        "node": info.node if info else -1,
        "total_nodes": info.total_nodes if info else -1,
        "shape": list(c.shape),
        "device": torch.cuda.get_device_name(0),
        "sum": c.sum().item(),
    }


def _assert_single(result: dict) -> None:
    assert result["shape"] == [512, 512]
    assert result["device"]
    assert isinstance(result["sum"], float)


def _assert_broadcast(results: list[dict]) -> None:
    assert len(results) == NODES
    nodes = sorted(r["node"] for r in results)
    assert nodes == list(range(NODES))
    for r in results:
        _assert_single(r)


# ---------------------------------------------------------------------------
# AWS
# ---------------------------------------------------------------------------


@pytest.mark.sanity
@pytest.mark.aws
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.xdist_group("aws")
class TestAWSSanity:
    @pytest.fixture(scope="class")
    def pool(self):
        with sky.App(console=False), sky.ComputePool(
            provider=sky.AWS(),
            accelerator=sky.accelerators.T4(),
            nodes=NODES,
            image=sky.Image(pip=["torch"]),
            allocation="spot-if-available",
        ) as p:
            yield p

    def test_single_dispatch(self, pool):
        result = gpu_matmul() >> pool
        _assert_single(result)

    def test_broadcast(self, pool):
        results = gpu_matmul() @ pool
        _assert_broadcast(results)


# ---------------------------------------------------------------------------
# RunPod
# ---------------------------------------------------------------------------


@pytest.mark.sanity
@pytest.mark.runpod
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.xdist_group("runpod")
class TestRunPodSanity:
    @pytest.fixture(scope="class")
    def pool(self):
        with sky.App(console=False), sky.ComputePool(
            provider=sky.RunPod(),
            accelerator=sky.accelerators.RTX_4090(),
            nodes=NODES,
            image=sky.Image(pip=["torch"]),
        ) as p:
            yield p

    def test_single_dispatch(self, pool):
        result = gpu_matmul() >> pool
        _assert_single(result)

    def test_broadcast(self, pool):
        results = gpu_matmul() @ pool
        _assert_broadcast(results)


# ---------------------------------------------------------------------------
# VastAI
# ---------------------------------------------------------------------------


@pytest.mark.sanity
@pytest.mark.vastai
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.xdist_group("vastai")
class TestVastAISanity:
    @pytest.fixture(scope="class")
    def pool(self):
        with sky.App(console=False), sky.ComputePool(
            provider=sky.VastAI(),
            accelerator=sky.accelerators.RTX_3090(),
            nodes=NODES,
            image=sky.Image(pip=["torch"]),
        ) as p:
            yield p

    def test_single_dispatch(self, pool):
        result = gpu_matmul() >> pool
        _assert_single(result)

    def test_broadcast(self, pool):
        results = gpu_matmul() @ pool
        _assert_broadcast(results)


# ---------------------------------------------------------------------------
# Verda
# ---------------------------------------------------------------------------


@pytest.mark.sanity
@pytest.mark.verda
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.xdist_group("verda")
class TestVerdaSanity:
    @pytest.fixture(scope="class")
    def pool(self):
        with sky.App(console=False), sky.ComputePool(
            provider=sky.Verda(),
            accelerator=sky.accelerators.A100(),
            nodes=NODES,
            image=sky.Image(pip=["torch"]),
        ) as p:
            yield p

    def test_single_dispatch(self, pool):
        result = gpu_matmul() >> pool
        _assert_single(result)

    def test_broadcast(self, pool):
        results = gpu_matmul() @ pool
        _assert_broadcast(results)
