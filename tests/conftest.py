from __future__ import annotations

from dataclasses import replace

import pytest

import skyward as sky
from skyward import Image, Options, PipIndex, Worker


@pytest.fixture(scope="session")
def pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-default'),
        nodes=2,
        vcpus=1,
        memory_gb=1,
        options=Options(worker=Worker(concurrency=5), console=False, logging=True),
    ) as p:
        yield p

@pytest.fixture(scope="session")
def pip_pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-pip'),
        nodes=1,
        image=Image(pip=["requests"]),
        vcpus=1,
        memory_gb=1,
        options=Options(worker=Worker(concurrency=5), console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def apt_pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-apt'),
        nodes=1,
        image=Image(apt=["jq"]),
        memory_gb=1,
        vcpus=1,
        options=Options(worker=Worker(concurrency=2), console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def env_pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-env'),
        nodes=1,
        image=Image(env={"MY_TEST_VAR": "hello123"}),
        vcpus=0.5,
        memory_gb=0.5,
        options=Options(console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def torch_pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-torch'),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(
            pip=["torch"],
            pip_indexes=[PipIndex(url="https://download.pytorch.org/whl/cpu", packages=["torch"])],
        ),
        options=Options(worker=Worker(concurrency=2), console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def jax_pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-jax'),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(pip=["jax"]),
        options=Options(console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def keras_pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-keras'),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(
            pip=["keras", "torch"],
            pip_indexes=[PipIndex(url="https://download.pytorch.org/whl/cpu", packages=["torch"])],
            env={"KERAS_BACKEND": "torch"},
        ),
        options=Options(console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def parallel_pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-parallel'),
        nodes=2,
        vcpus=1,
        memory_gb=1,
        image=Image(pip=["joblib", "scikit-learn"]),
        options=Options(worker=Worker(concurrency=3), console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def torch_plugin_pool():
    # Thread executor: dist.init_process_group is per-process state.
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-torch-plugin'),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        plugins=[sky.plugins.torch(backend="gloo")],
        options=Options(worker=Worker(concurrency=2, executor="thread"), console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def jax_plugin_pool():
    # Strip image transform (adds CUDA deps) — containers are CPU-only.
    # Keep around_app which initializes jax.distributed.
    # Thread executor: jax.distributed.initialize() is per-process state.
    jax_plugin = replace(sky.plugins.jax(), transform=None)
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-jax-plugin'),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(pip=["jax"]),
        plugins=[jax_plugin],
        options=Options(worker=Worker(executor="thread"), console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def keras_plugin_pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-keras-plugin'),
        nodes=2,
        vcpus=2,
        memory_gb=2,
        image=Image(
            pip=["torch"],
            pip_indexes=[PipIndex(url="https://download.pytorch.org/whl/cpu", packages=["torch"])],
        ),
        plugins=[sky.plugins.keras(backend="torch")],
        options=Options(console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def pandas_pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-pandas'),
        nodes=1,
        image=Image(pip=["pandas==2.3.3"]),
        vcpus=1,
        memory_gb=1,
        options=Options(worker=Worker(concurrency=2), console=False),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def joblib_plugin_pool():
    with sky.Compute(
        provider=sky.Container(network="skyward", container_prefix='skyward-joblib-plugin'),
        nodes=2,
        vcpus=1,
        memory_gb=1,
        image=Image(pip=["scikit-learn"]),
        plugins=[sky.plugins.joblib()],
        options=Options(worker=Worker(concurrency=3), console=False),
    ) as p:
        yield p
