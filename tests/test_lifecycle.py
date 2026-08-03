"""The pool as a thing with a life of its own.

The machines belong to the daemon, not to the process that asked for them, which
is the whole point of these: one process leaves, another one picks the compute up,
and the work carries on.
"""

import sys
import time

import cloudpickle
import pytest

import skyward as sky
from tests.conftest import Build

pytestmark = [pytest.mark.compute, pytest.mark.xdist_group("lifecycle")]

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@sky.function
def double(x: int) -> int:
    return x * 2


def describe_a_compute_that_outlives_the_process_that_made_it() -> None:
    def it_is_picked_up_by_name_and_goes_on_working(compute: Build, daemon: str) -> None:
        with compute(name="picked-up", delete_on_exit=False) as pool:
            assert double(2) >> pool == 4
            left_behind = pool.id

        with sky.Compute.attached("picked-up", url=daemon, delete_on_exit=True) as rejoined:
            assert rejoined.id == left_behind, "the same compute, not a new one"
            assert double(3) >> rejoined == 6, "and its machines were still there"


def describe_a_pool_that_may_start_before_it_is_whole() -> None:
    @pytest.mark.timeout(600)
    def it_takes_work_at_its_floor_and_grows_to_what_was_asked(compute: Build) -> None:
        with compute(nodes=sky.Nodes(desired=2, min=1)) as pool:
            assert pool.current_nodes() >= 1, "the block returned as soon as the floor was ready"
            assert double(4) >> pool == 8

            deadline = time.monotonic() + 240
            while pool.current_nodes() < 2 and time.monotonic() < deadline:
                time.sleep(2)

            assert pool.current_nodes() == 2, "and the second machine arrived on its own"
