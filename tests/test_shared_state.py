"""State two nodes agree about, on two real machines.

There is no way to fake this one: a counter that two processes agree about is the
whole claim, and a single node would prove none of it.
"""

import sys

import cloudpickle
import pytest

import skyward as sky

pytestmark = [pytest.mark.compute, pytest.mark.xdist_group("pool")]

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@sky.function
def tally() -> int:
    """Every node counts itself, and nobody reads until all of them have."""
    info = sky.instance_info()

    sky.dict("ranks")[info.node] = info.rank
    sky.counter("tally").add()
    sky.barrier("counted", info.nodes).wait(60)

    return sky.counter("tally").get()


@sky.function
def ranks() -> list[int]:
    return sorted(rank for _, rank in sky.dict("ranks").items())


@sky.function
def guarded(times: int) -> None:
    """A read-modify-write that is only safe because of the lock around it."""
    for _ in range(times):
        with sky.lock("balance"):
            bank = sky.dict("bank")
            bank["total"] = (bank.get("total", 0) or 0) + 1


@sky.function
def balance() -> int:
    return sky.dict("bank")["total"]


@sky.function
def fill(items: int) -> None:
    work = sky.queue("work")
    for item in range(items):
        work.offer(item)


@sky.function
def drain() -> list[int]:
    work = sky.queue("work")
    taken: list[int] = []
    while (item := work.poll()) is not None:
        taken.append(item)
    return taken


def describe_a_counter_and_a_map_shared_by_the_nodes() -> None:
    def they_read_each_others_writes(pool: sky.Compute) -> None:
        assert tally() @ pool == [2, 2], "each node read the other's increment, because the barrier held it"
        assert ranks() >> pool == [0, 1], "and both wrote to the same map"


def describe_a_lock_held_across_nodes() -> None:
    def it_makes_a_read_modify_write_safe(pool: sky.Compute) -> None:
        guarded(20) @ pool

        assert balance() >> pool == 40, "forty increments, none of them lost"


def describe_a_queue_shared_by_the_nodes() -> None:
    def what_one_node_puts_in_another_takes_out(pool: sky.Compute) -> None:
        fill(6) >> pool

        assert sorted(item for shard in drain() @ pool for item in shard) == [0, 1, 2, 3, 4, 5]
