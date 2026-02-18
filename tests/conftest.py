from __future__ import annotations

import pytest

import skyward as sky
from skyward import ComputePool, Image


@pytest.fixture(scope="session")
def pool():
    with ComputePool(
        provider=sky.Container(),
        nodes=2
    ) as p:
        yield p


@pytest.fixture(scope="session")
def pip_pool():
    with ComputePool(
        provider=sky.Container(),
        nodes=1,
        image=Image(pip=["requests"]),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def apt_pool():
    with ComputePool(
        provider=sky.Container(),
        nodes=1,
        image=Image(apt=["jq"]),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def env_pool():
    with ComputePool(
        provider=sky.Container(),
        nodes=1,
        image=Image(env={"MY_TEST_VAR": "hello123"}),
    ) as p:
        yield p


@pytest.fixture(scope="session")
def torch_pool():
    with ComputePool(
        provider=sky.Container(),
        nodes=2,
        vcpus=3,
        memory_gb=3,
        image=Image(
            pip=["torch"],
            pip_extra_index_url='https://download.pytorch.org/whl/cpu'
        ),
    ) as p:
        yield p
