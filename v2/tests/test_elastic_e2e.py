"""The pool sizes itself to the work.

Which is the whole of what the reconciler does, and everything else in it exists to
make this safe: the row before the machine, so a pass that runs while one is booting
does not buy another; the queue that is really a queue, so the demand is visible; the
idle count, so a burst that ended does not cost the machines it just paid to boot.
"""

import sys
import time
from pathlib import Path

import cloudpickle
import pytest

import skyward2 as skyward
from skyward2 import Compute

pytestmark = pytest.mark.e2e

IMAGE = skyward.Image(python="3.13", skyward="local")

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@skyward.function
def occupy(seconds: float) -> str:
    import os
    import time

    time.sleep(seconds)
    return os.environ["SKYWARD_NODE"]


def until(pool: Compute, nodes: int, seconds: float) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if (current := pool.current_nodes()) == nodes:
            return current
        time.sleep(1.0)
    return pool.current_nodes()


def test_a_queue_grows_the_pool_and_an_empty_one_gives_it_back(tmp_path: Path):
    with skyward.Compute(
        provider=skyward.Container(),
        nodes=(1, 3),
        cpus=1,
        memory_gb=1,
        image=IMAGE,
        database=tmp_path / "skyward.sqlite",
    ) as pool:
        assert pool.current_nodes() == 1, "nothing has been asked of it, so it is at its floor"

        held = [occupy(25.0) > pool for _ in range(3)]

        assert until(pool, nodes=3, seconds=120) == 3, "three calls, one slot each — the queue is the pressure"

        nodes = {future.result() for future in held}
        assert len(nodes) > 1, "the machines it bought took the work that bought them"

        assert until(pool, nodes=1, seconds=120) == 1, "the burst ended and the pool gave the machines back"
