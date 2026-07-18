"""The subprocess executors, against two real nodes.

A ``process`` or ``loky`` task runs off the worker's process, so the collections it
reaches for have to come home over the IPC bridge. The only way to prove they do is to
run one on real nodes and read what it wrote — a counter both nodes agree on, and a
lock that still serialises them, with the work happening in a subprocess the whole time.
"""

import sys
from pathlib import Path

import cloudpickle
import pytest

import skyward as sky
from skyward import Compute, Executor

pytestmark = pytest.mark.e2e

IMAGE = sky.Image(python="3.13", skyward="local")

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@sky.function
def tally() -> int:
    """Every node counts itself, and nobody reads until all of them have."""
    info = sky.instance_info()
    sky.counter("tally").add()
    sky.barrier("counted", info.nodes).wait(60)
    return sky.counter("tally").get()


@sky.function
def guarded(times: int) -> None:
    """A read-modify-write that is only safe because of the lock around it."""
    for _ in range(times):
        with sky.lock("balance"):
            bank = sky.dict("bank")
            bank["total"] = bank.get("total", 0) + 1


@sky.function
def balance() -> int:
    return sky.dict("bank")["total"]


@pytest.fixture(
    params=[Executor("process", True), Executor("process", False), Executor("loky", True)],
    ids=["process-reused", "process-fresh", "loky"],
)
def pool(request: pytest.FixtureRequest, tmp_path: Path):
    with sky.Compute(
        provider=sky.Container(),
        nodes=2,
        cpus=1,
        memory_gb=1,
        image=IMAGE,
        database=tmp_path / "skyward.sqlite",
        executor=request.param,
    ) as pool:
        yield pool


def test_a_counter_survives_the_trip_to_a_subprocess(pool: Compute):
    assert tally() @ pool == [2, 2], "each node read the other's increment, from a subprocess"


def test_a_lock_still_serialises_the_nodes_off_process(pool: Compute):
    guarded(20) @ pool
    assert balance() >> pool == 40, "forty increments, none of them lost across the bridge"
