"""The SDK, from the outside, against real containers.

Nothing here knows there is a control plane. The daemon runs in this process and
is reached over HTTP that never touches a socket, which is precisely what the
test is for: the code below is what a user writes, and it is the same code they
would write against a daemon on another machine.
"""

import sys
from pathlib import Path

import cloudpickle
import pytest

import skyward2 as skyward
from skyward2 import Compute, TaskFailedError

pytestmark = pytest.mark.e2e

IMAGE = skyward.Image(python="3.13", skyward="local")

cloudpickle.register_pickle_by_value(sys.modules[__name__])
"""Ship this module's functions by value.

cloudpickle sends anything importable by reference, and a test module is not
importable on a node. Real code hits the same wall the moment a function lives
in a package rather than in `__main__` — the SDK will have to say something
about that, and this line is where it will be said.
"""


@skyward.function
def double(x: int) -> int:
    return x * 2


@skyward.function
def blow_up() -> int:
    raise ValueError("the function said no")


@skyward.function
def whoami() -> str:
    import os

    return os.environ["SKYWARD_NODE"]


@pytest.fixture
def pool(tmp_path: Path):
    with skyward.Compute(
        provider=skyward.Container(),
        nodes=2,
        cpus=1,
        memory_gb=1,
        image=IMAGE,
        database=tmp_path / "skyward.sqlite",
    ) as pool:
        yield pool


def test_a_call_runs_on_a_node_and_the_value_comes_back(pool: Compute):
    assert double(21) >> pool == 42


def test_a_broadcast_runs_on_every_node(pool: Compute):
    everyone = whoami() @ pool

    assert len(everyone) == 2
    assert len(set(everyone)) == 2, "two nodes, two answers"


def test_a_future_does_not_block_and_a_group_overlaps(pool: Compute):
    future = double(5) > pool
    assert future.result() == 10

    assert (double(1) & double(2) & double(3)) >> pool == [2, 4, 6]


def test_gather_groups_calls_the_way_the_ampersand_does(pool: Compute):
    assert skyward.gather(double(1), double(2)) >> pool == [2, 4]


def test_map_keeps_the_order_it_was_asked_in(pool: Compute):
    assert pool.map(double, range(6)) == [0, 2, 4, 6, 8, 10]


def test_a_function_that_raises_arrives_as_an_exception_with_its_traceback(pool: Compute):
    with pytest.raises(TaskFailedError) as raised:
        blow_up() >> pool

    assert "the function said no" in raised.value.message
    assert "ValueError" in (raised.value.details.get("traceback") or ""), "the remote traceback survives the trip"
