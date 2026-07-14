"""State the nodes share, against two real machines.

There is no way to fake this one. A counter that two processes agree about is the
whole claim, and a single node would prove none of it.
"""

import sys
from pathlib import Path

import cloudpickle
import pytest

import skyward2 as skyward
from skyward2 import Compute

pytestmark = pytest.mark.e2e

IMAGE = skyward.Image(python="3.13", skyward="local")

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@skyward.function
def tally() -> int:
    """Every node counts itself, and nobody reads until all of them have."""
    info = skyward.instance_info()

    skyward.dict("ranks")[info.node] = info.rank
    skyward.counter("tally").add()
    skyward.barrier("counted", info.nodes).wait(60)

    return skyward.counter("tally").get()


@skyward.function
def ranks() -> list[int]:
    return sorted(rank for _, rank in skyward.dict("ranks").items())


@skyward.function
def guarded(times: int) -> None:
    """A read-modify-write that is only safe because of the lock around it."""
    for _ in range(times):
        with skyward.lock("balance"):
            balance = skyward.dict("bank")
            balance["total"] = balance.get("total", 0) + 1


@skyward.function
def balance() -> int:
    return skyward.dict("bank")["total"]


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


def test_the_nodes_agree_about_a_counter_and_a_dict(pool: Compute):
    assert tally() @ pool == [2, 2], "each node reads the other's increment, because the barrier held it"
    assert ranks() >> pool == [0, 1], "and both wrote to the same map"


def test_a_lock_makes_a_read_modify_write_safe_across_nodes(pool: Compute):
    guarded(20) @ pool

    assert balance() >> pool == 40, "forty increments, none of them lost"
