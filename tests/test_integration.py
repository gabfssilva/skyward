import pytest

import skyward as sky

# =============================================================================
# Compute functions
# =============================================================================


@sky.compute
def hello() -> str:
    return "Hello from Skyward!"


@sky.compute
def check_cuda() -> dict:
    import torch  # type: ignore[reportMissingImports]

    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@sky.compute
def check_jax() -> dict:
    import jax

    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kind": devices[0].device_kind if devices else None,
    }


@sky.compute
def check_env() -> str:
    import os

    return os.environ.get("MY_CUSTOM_VAR", "NOT_SET")


@sky.compute
def distributed_sum(data: list[int]) -> dict:
    info = sky.instance_info()
    assert info is not None
    local = sky.shard(data)
    return {"node": info.node, "partial_sum": sum(local)}


# =============================================================================
# Tests - CPU Only
# =============================================================================


@pytest.mark.integration
def test_hello_world(cpu_provider):
    provider, _ = cpu_provider

    with sky.pool(provider=provider, nodes=1, panel=False):
        result = hello() >> sky

    assert result == "Hello from Skyward!"


# =============================================================================
# Tests - GPU Single Node
# =============================================================================


@pytest.mark.integration
def test_custom_lib_cuda(gpu_provider):
    provider, accelerator = gpu_provider

    with sky.pool(
        provider=provider,
        accelerator=accelerator,
        nodes=1,
        panel=False,
        image=sky.Image(pip=["torch"]),
    ):
        result = check_cuda() >> sky

    assert result["cuda_available"] is True
    assert result["device_count"] >= 1


@pytest.mark.integration
def test_custom_lib_jax(gpu_provider):
    provider, accelerator = gpu_provider

    with sky.pool(
        provider=provider,
        accelerator=accelerator,
        nodes=1,
        panel=False,
        image=sky.Image(pip=["jax[cuda12]"]),
    ):
        result = check_jax() >> sky

    assert result["backend"] == "gpu"
    assert result["device_count"] >= 1


@pytest.mark.integration
def test_custom_env(gpu_provider):
    provider, accelerator = gpu_provider

    with sky.pool(
        provider=provider,
        accelerator=accelerator,
        nodes=1,
        panel=False,
        image=sky.Image(env={"MY_CUSTOM_VAR": "skyward_test_123"}),
    ):
        result = check_env() >> sky

    assert result == "skyward_test_123"


# =============================================================================
# Tests - Cluster (distributed)
# =============================================================================


@pytest.mark.integration
def test_distributed_training(cluster_provider):
    provider, accelerator = cluster_provider
    data = list(range(100))

    with sky.pool(provider=provider, accelerator=accelerator, nodes=2, panel=False):
        results = distributed_sum(data) @ sky

    assert len(results) == 2
    assert {r["node"] for r in results} == {0, 1}
    assert sum(r["partial_sum"] for r in results) == sum(data)
