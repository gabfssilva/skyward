"""Each provider, for real: a GPU machine that comes up, runs code, and goes away.

These cost money. They never run by themselves — the default pytest run excludes
the ``sanity`` marker, and even an explicit ``-m sanity`` is inert unless
``SKYWARD_SANITY=1`` is in the environment. The intended invocation is
``SKYWARD_SANITY=1 task test:sanity:runpod``, by a human, on purpose.

Credentials come from the environment *here*, never in an adapter: a provider
whose variables are missing is skipped, not failed.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import pytest

import skyward as sky

pytestmark = [
    pytest.mark.sanity,
    pytest.mark.timeout(1800),
    pytest.mark.skipif(os.environ.get("SKYWARD_SANITY") != "1", reason="costs money: set SKYWARD_SANITY=1 to run"),
]

cloudpickle.register_pickle_by_value(sys.modules[__name__])

SANITY = Path(__file__).parent / ".sanity"


@dataclass(frozen=True, slots=True)
class Target:
    kind: str
    provider: sky.Provider
    accelerator: sky.Accelerator
    env: tuple[str, ...]
    cluster: bool = False


def targets() -> tuple[Target, ...]:
    return (
        Target("aws", sky.AWS(region="us-east-1"), sky.accelerators.T4(), ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")),
        Target("gcp", sky.GCP(), sky.accelerators.T4(), ("GCP_SERVICE_ACCOUNT_JSON",)),
        Target("hyperstack", sky.Hyperstack(), sky.accelerators.A100(), ("HYPERSTACK_API_KEY",)),
        Target("jarvislabs", sky.JarvisLabs(), sky.accelerators.RTX_5000ADA(), ("JL_API_KEY",)),
        Target("lambda_cloud", sky.Lambda(), sky.accelerators.A10(), ("LAMBDA_API_KEY",)),
        Target("massed_compute", sky.MassedCompute(), sky.accelerators.RTX_A4000(), ("MASSED_API_KEY",)),
        Target("novita", sky.Novita(), sky.accelerators.RTX_4090(), ("NOVITA_API_KEY",)),
        Target("runpod", sky.RunPod(container_image='nvidia/cuda:12.8.2-runtime-ubuntu24.04'), sky.accelerators.A40(), ("RUNPOD_API_KEY",)),
        Target("salad", sky.Salad(), sky.accelerators.RTX_4090(), ("SALAD_API_KEY", "SALAD_ORGANIZATION", "SALAD_PROJECT")),
        Target("scaleway", sky.Scaleway(), sky.accelerators.L4(), ("SCW_SECRET_KEY",)),
        Target("tensordock", sky.TensorDock(), sky.accelerators.RTX_4090(), ("TENSORDOCK_API_TOKEN",)),
        Target("vastai", sky.VastAI(), sky.accelerators.RTX_4090(), ("VAST_API_KEY",), cluster=True),
        Target("verda", sky.Verda(), sky.accelerators.A100(), ("VERDA_CLIENT_ID", "VERDA_CLIENT_SECRET")),
        Target("vultr", sky.Vultr(), sky.accelerators.L40S(), ("VULTR_API_KEY",)),
    )


def cases(cluster: bool = False) -> list[object]:
    return [
        pytest.param(target, marks=getattr(pytest.mark, target.kind), id=target.kind)
        for target in targets()
        if target.cluster or not cluster
    ]


def database(target: Target, flavor: str) -> Path:
    """One SQLite per provider and level, kept under ``tests/.sanity`` for a post-mortem."""
    SANITY.mkdir(exist_ok=True)
    return SANITY / f"{target.kind}-{flavor}.sqlite"


def ready(target: Target) -> Target:
    if missing := [var for var in target.env if not os.environ.get(var)]:
        pytest.skip(f"{target.kind}: no credential in {', '.join(missing)}")
    return target


@sky.function
def gpu_name() -> str:
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return probe.stdout.strip().lower()


@sky.function
def whoami() -> tuple[int, int]:
    info = sky.instance_info()
    return info.rank, info.nodes


@sky.function
def tally() -> int:
    counter = sky.counter("sanity-tally")
    counter.add()
    sky.barrier("sanity-counted", parties=sky.instance_info().nodes).wait()
    return counter.get()


def describe_one_gpu_node() -> None:
    @pytest.mark.parametrize("target", cases())
    def it_boots_runs_and_answers(target: Target) -> None:
        target = ready(target)

        with sky.Compute(provider=target.provider, accelerator=target.accelerator, nodes=1, database=database(target, "single")) as pool:
            name = gpu_name() >> pool
            rank, nodes = whoami() >> pool

        assert target.accelerator.name.split("-")[-1] in name.replace(" ", "-"), f"asked {target.accelerator.name}, machine says {name}"
        assert (rank, nodes) == (0, 1)


def describe_a_cluster_of_two() -> None:
    @pytest.mark.parametrize("target", cases(cluster=True))
    def they_see_each_other(target: Target) -> None:
        target = ready(target)

        with sky.Compute(provider=target.provider, accelerator=target.accelerator, nodes=2, database=database(target, "cluster")) as pool:
            assert sorted(whoami() @ pool) == [(0, 2), (1, 2)]
            assert tally() @ pool == [2, 2], "both nodes bumped the same counter"

            shards = my_share(list(range(5))) @ pool
            assert sorted(item for shard in shards for item in shard) == list(range(5))


@sky.function
def my_share(data: list[int]) -> list[int]:
    return list(sky.shard(data))
