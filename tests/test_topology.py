"""What a node knows about itself and the others, and what it does with it.

``shard`` is the reason the topology is exposed at all: a function that splits its
own data by rank only works if every node agrees about how many there are and which
one it is.
"""

import sys

import cloudpickle
import pytest

import skyward as sky
from tests.conftest import Build

pytestmark = [pytest.mark.compute, pytest.mark.xdist_group("pool")]

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@sky.function
def placement() -> tuple[int, int, bool, int]:
    info = sky.instance_info()
    return info.rank, info.nodes, info.is_head, len(info.peers)


@sky.function
def my_share(data: list[int]) -> list[int]:
    return list(sky.shard(data))


@sky.function
def my_pairs(features: list[int], labels: list[str]) -> list[tuple[int, str]]:
    mine, theirs = sky.shard(features, labels, shuffle=True, seed=7)
    return list(zip(mine, theirs, strict=True))


@sky.function
def worker_slot() -> tuple[int, int]:
    sky.barrier("slots", parties=2).wait()
    info = sky.instance_info()
    return info.global_worker_index, info.total_workers


def describe_a_node_among_the_others() -> None:
    def it_knows_its_rank_its_peers_and_whether_it_leads(pool: sky.Compute) -> None:
        assert sorted(placement() @ pool) == [(0, 2, True, 2), (1, 2, False, 2)]


def describe_sharding_data_by_rank() -> None:
    def it_partitions_the_data_and_loses_none_of_it(pool: sky.Compute) -> None:
        shards = my_share(list(range(7))) @ pool

        assert sorted(item for shard in shards for item in shard) == list(range(7))
        assert sorted(len(shard) for shard in shards) == [3, 4], "as evenly as seven splits in two"

    def it_keeps_paired_sequences_aligned_even_when_shuffled(pool: sky.Compute) -> None:
        pairs = [pair for shard in my_pairs([1, 2, 3, 4], ["a", "b", "c", "d"]) @ pool for pair in shard]

        assert sorted(pairs) == [(1, "a"), (2, "b"), (3, "c"), (4, "d")], "every row kept its own label"


def describe_running_more_than_one_task_per_node() -> None:
    def it_gives_every_worker_a_slot_of_its_own(compute: Build) -> None:
        with compute(nodes=1, executor=sky.Executor(type="process", concurrency=2)) as pool:
            slots = sky.gather(worker_slot(), worker_slot()) >> pool

        assert {total for _, total in slots} == {2}, "two workers on the one node"
        assert sorted(index for index, _ in slots) == [0, 1], "and each of them its own index"
