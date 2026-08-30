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
from tests.conftest import Build, cli, rows

pytestmark = [pytest.mark.compute, pytest.mark.xdist_group("lifecycle")]

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@sky.function
def double(x: int) -> int:
    return x * 2


@sky.function
def world() -> int:
    return sky.instance_info().nodes


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
        with compute(nodes=sky.Nodes(initial=2, min=1)) as pool:
            assert pool.current_nodes() >= 1, "the block returned as soon as the floor was ready"
            assert double(4) >> pool == 8

            deadline = time.monotonic() + 240
            while pool.current_nodes() < 2 and time.monotonic() < deadline:
                time.sleep(2)

            assert pool.current_nodes() == 2, "and the second machine arrived on its own"


def describe_a_compute_asked_for_a_different_size_after_it_is_up() -> None:
    @pytest.mark.timeout(600)
    def it_grows_where_it_stands_and_answers_the_shell_asking_the_same(compute: Build, daemon: str) -> None:
        with compute(nodes=1) as pool:
            pool.resize(2)

            deadline = time.monotonic() + 240
            while pool.current_nodes() < 2 and time.monotonic() < deadline:
                time.sleep(2)

            assert pool.current_nodes() == 2, "the machine asked for after the fact was bought"
            assert double(5) @ pool == [10, 10], "and it takes work like the one that was there before it"
            assert world() @ pool == [2, 2], "and the node that was here first knows the world grew under it"

            before = int(rows("compute", "get", pool.id, "--url", daemon)[0]["generation"])
            scaled = rows("compute", "scale", pool.id, "--nodes", "1", "--url", daemon)[0]

            assert int(scaled["generation"]) == before + 1, "the shell writes the intent the SDK writes"


def describe_a_pool_that_never_becomes_ready() -> None:
    @pytest.mark.timeout(600)
    def it_takes_down_what_it_had_already_bought(compute: Build, daemon: str) -> None:
        pool = compute(options=sky.Options(ready_timeout=3))

        with pytest.raises(TimeoutError), pool:
            pytest.fail("the block cannot open on a pool that never came up")

        assert rows("compute", "get", pool.id, "--url", daemon)[0]["state"] == "deleted", "what it had already bought was given back"

    @pytest.mark.timeout(600)
    def it_leaves_standing_a_compute_asked_to_outlive_the_process(compute: Build, daemon: str) -> None:
        pool = compute(options=sky.Options(ready_timeout=3), delete_on_exit=False)

        with pytest.raises(TimeoutError), pool:
            pytest.fail("the block cannot open on a pool that never came up")

        state = rows("compute", "get", pool.id, "--url", daemon)[0]["state"]
        cli("compute", "delete", pool.id, "--url", daemon)

        assert state not in {"deleting", "deleted"}, "the caller said the machines outlive the process"


def describe_a_compute_the_shell_created_and_nobody_holds() -> None:
    @pytest.mark.timeout(900)
    def it_goes_on_being_reconciled_after_it_is_ownerless(daemon: str) -> None:
        created = rows("compute", "create", "--provider", "container", "--nodes", "1", "--cpus", "1", "--memory", "1", "--name", "unheld", "--url", daemon)[0]
        born = time.monotonic()

        try:
            assert _settles(created["id"], daemon, nodes=1) == "ready"

            time.sleep(max(0.0, born + 70 - time.monotonic()))
            rows("compute", "scale", created["id"], "--nodes", "2", "--url", daemon)

            assert _settles(created["id"], daemon, nodes=2) == "ready", "a lease it never had is not a reason to stop reconciling it"
        finally:
            cli("compute", "delete", created["id"], "--url", daemon)


def _settles(ref: str, daemon: str, nodes: int) -> str:
    """The compute's state once it stands on ``nodes`` machines, or whatever it settled as."""
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        compute = rows("compute", "get", ref, "--url", daemon)[0]
        if compute["state"] in {"failed", "degraded"} or int(compute["ready"]) == nodes:
            return compute["state"]
        time.sleep(3)
    return "timed out"
