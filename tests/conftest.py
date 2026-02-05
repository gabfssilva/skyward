import pytest
import skyward as sky
from skyward.accelerators import T4, A100, RTX_4090


PROVIDERS = {
    "vastai": (sky.VastAI(bid_multiplier=2, min_reliability=0.99), RTX_4090()),
    "aws": (sky.AWS(), T4()),
    "verda": (sky.Verda(), A100(memory="80GB")),
    "runpod": (sky.RunPod(), RTX_4090()),
}

SINGLE_NODE = ["vastai", "aws", "verda", "runpod"]
CLUSTER = ["vastai", "aws", "verda"]


@pytest.fixture(params=SINGLE_NODE, ids=SINGLE_NODE)
def cpu_provider(request):
    provider, _ = PROVIDERS[request.param]
    return provider, None


@pytest.fixture(params=SINGLE_NODE, ids=SINGLE_NODE)
def gpu_provider(request):
    return PROVIDERS[request.param]


@pytest.fixture(params=CLUSTER, ids=CLUSTER)
def cluster_provider(request):
    return PROVIDERS[request.param]
