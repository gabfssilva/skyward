from __future__ import annotations

import pytest

import skyward as sky
from skyward import ComputePool, Image


@pytest.fixture(scope="session")
def pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-default'),
        nodes=2,
        concurrency=5,
        vcpus=1,
        memory_gb=1,
    ) as p:
        yield p

@pytest.fixture(scope="session")
def pip_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-pip'),
        nodes=1,
        concurrency=5,
        image=Image(pip=["requests"]),
        vcpus=1,
        memory_gb=1,
    ) as p:
        yield p


@pytest.fixture(scope="session")
def apt_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-apt'),
        nodes=1,
        concurrency=2,
        image=Image(apt=["jq"]),
        memory_gb=1,
        vcpus=1,
    ) as p:
        yield p


@pytest.fixture(scope="session")
def env_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-env'),
        nodes=1,
        image=Image(env={"MY_TEST_VAR": "hello123"}),
        vcpus=0.5,
        memory_gb=0.5,
    ) as p:
        yield p


@pytest.fixture(scope="session")
def torch_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-torch'),
        concurrency=2,
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(
            pip=["torch"],
            pip_extra_index_url='https://download.pytorch.org/whl/cpu'
        ),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def jax_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-jax'),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(pip=["jax"]),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def keras_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-keras'),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(
            pip=["keras", "torch"],
            pip_extra_index_url='https://download.pytorch.org/whl/cpu',
            env={"KERAS_BACKEND": "torch"},
        ),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def parallel_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-parallel'),
        nodes=2,
        vcpus=1,
        memory_gb=1,
        concurrency=3,
        image=Image(pip=["joblib", "scikit-learn"]),
    ) as p:
        yield p
