"""The compute outlives the block that asked for it.

Which is the whole reason the control plane is a daemon and not a library. The
second pool here creates nothing: it finds the machines that are already running,
adopts the workers that are already on them, and gets an answer out of them.
"""

import sys
from pathlib import Path

import cloudpickle
import pytest

import skyward2 as skyward
from skyward2 import SkywardError

pytestmark = pytest.mark.e2e

IMAGE = skyward.Image(python="3.13", skyward="local")

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@skyward.function
def double(x: int) -> int:
    return x * 2


def test_a_second_pool_attaches_to_the_compute_the_first_one_left(tmp_path: Path):
    database = tmp_path / "skyward.sqlite"

    with skyward.Compute(
        provider=skyward.Container(),
        nodes=1,
        cpus=1,
        memory_gb=1,
        image=IMAGE,
        name="overnight",
        database=database,
        delete_on_exit=False,
    ) as pool:
        assert double(1) >> pool == 2
        left = pool.id

    with skyward.Compute.attached("overnight", database=database, delete_on_exit=True) as again:
        assert again.id == left, "the same compute, found by the name it was given"
        assert double(21) >> again == 42, "the worker that was already running took the task"


def test_attaching_to_nothing_says_so(tmp_path: Path):
    with pytest.raises(SkywardError) as raised, skyward.Compute.attached("nobody", database=tmp_path / "skyward.sqlite"):
        pass

    assert raised.value.code == "not_found"


def test_a_spec_and_an_attachment_describe_different_computes():
    with pytest.raises(ValueError, match="already exists"):
        skyward.Compute(provider=skyward.Container(), attach="overnight")
