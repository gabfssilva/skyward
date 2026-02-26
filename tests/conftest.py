from __future__ import annotations

from dataclasses import replace

import pytest

import skyward as sky
from skyward import ComputePool, Image, PipIndex, Worker


@pytest.fixture(scope="session")
def pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-default'),
        nodes=2,
        worker=Worker(concurrency=5),
        vcpus=1,
        memory_gb=1,
    ) as p:
        yield p

@pytest.fixture(scope="session")
def pip_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-pip'),
        nodes=1,
        worker=Worker(concurrency=5),
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
        worker=Worker(concurrency=2),
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
        worker=Worker(concurrency=2),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(
            pip=["torch"],
            pip_indexes=[PipIndex(url="https://download.pytorch.org/whl/cpu", packages=["torch"])],
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
            pip_indexes=[PipIndex(url="https://download.pytorch.org/whl/cpu", packages=["torch"])],
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
        worker=Worker(concurrency=3),
        image=Image(pip=["joblib", "scikit-learn"]),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def torch_plugin_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-torch-plugin'),
        worker=Worker(concurrency=2),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        plugins=[sky.plugins.torch(backend="gloo")],
    ) as p:
        yield p


@pytest.fixture(scope="session")
def jax_plugin_pool():
    # Strip image transform (adds CUDA deps) — containers are CPU-only.
    # Keep around_app which initializes jax.distributed.
    jax_plugin = replace(sky.plugins.jax(), transform=None)
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-jax-plugin'),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(pip=["jax"]),
        plugins=[jax_plugin],
    ) as p:
        yield p


@pytest.fixture(scope="session")
def keras_plugin_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-keras-plugin'),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(
            pip=["torch"],
            pip_indexes=[PipIndex(url="https://download.pytorch.org/whl/cpu", packages=["torch"])],
        ),
        plugins=[sky.plugins.keras(backend="torch")],
    ) as p:
        yield p


@pytest.fixture(scope="session")
def joblib_plugin_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(network="skyward", container_prefix='skyward-joblib-plugin'),
        nodes=2,
        vcpus=1,
        memory_gb=1,
        worker=Worker(concurrency=3),
        image=Image(pip=["scikit-learn"]),
        plugins=[sky.plugins.joblib()],
    ) as p:
        yield p
