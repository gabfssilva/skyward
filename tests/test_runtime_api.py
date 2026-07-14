"""What a function can ask about the node it is running on.

The environment is the whole input, so the tests set it and nothing else — there is
no node here, and there does not need to be one.
"""

import pytest

from skyward.runtime.api import (
    NotOnANodeError,
    Policy,
    instance_info,
    is_head,
    shard,
    silent,
    stdout,
)


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch):
    def place(rank: int, peers: int = 4):
        monkeypatch.setenv("SKYWARD_NODE", f"nod_{rank}")
        monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_1")
        monkeypatch.setenv("SKYWARD_RANK", str(rank))
        monkeypatch.setenv("SKYWARD_PEERS", ",".join(f"10.0.0.{index}" for index in range(peers)))

    return place


def test_a_node_knows_where_it_is_among_the_others(node):
    node(rank=2)
    info = instance_info()

    assert (info.rank, info.nodes) == (2, 4)
    assert info.host == "10.0.0.2", "a node finds itself at its own rank"
    assert info.head == "10.0.0.0"
    assert not info.is_head and not is_head()


def test_asking_off_a_node_says_so_rather_than_guessing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SKYWARD_NODE", raising=False)

    with pytest.raises(NotOnANodeError):
        instance_info()


def test_the_shards_of_one_dataset_partition_it(node):
    data = list(range(10))
    shards = []
    for rank in range(4):
        node(rank=rank)
        shards.append(list(shard(data)))

    assert [len(part) for part in shards] == [2, 3, 2, 3], "ten does not divide by four; nothing is dropped"
    assert sorted(item for part in shards for item in part) == data


def test_a_shard_of_a_tuple_is_a_tuple(node):
    node(rank=0)

    assert shard((1, 2, 3, 4)) == (1,)


def test_drop_last_gives_every_node_the_same_count(node):
    counts = []
    for rank in range(4):
        node(rank=rank)
        counts.append(len(shard(list(range(10)), drop_last=True)))

    assert counts == [2, 2, 2, 2], "the two that would not go round are dropped"


def test_shuffled_shards_still_partition_and_agree_across_nodes(node):
    data = list(range(12))
    shards = []
    for rank in range(4):
        node(rank=rank)
        shards.append(list(shard(data, shuffle=True, seed=7)))

    seen = [item for part in shards for item in part]
    assert sorted(seen) == data, "every element lands on exactly one node"
    assert seen != data, "and not in the order it arrived in"


def test_a_policy_narrows_by_stream_and_by_rank():
    everything = Policy()
    assert everything.allows("stdout", rank=3) and everything.allows("stderr", rank=3)

    quiet = Policy(streams=frozenset({"stdout"}), ranks=frozenset({0}))
    assert quiet.allows("stdout", rank=0)
    assert not quiet.allows("stderr", rank=0)
    assert not quiet.allows("stdout", rank=1)


def test_the_decorators_hold_their_policy_only_while_the_function_runs():
    from skyward.runtime.api import policy

    @stdout(only=0)
    def loud() -> Policy:
        return policy.get()

    @silent
    def quiet() -> Policy:
        return policy.get()

    assert loud() == Policy(streams=frozenset({"stdout"}), ranks=frozenset({0}))
    assert quiet() == Policy(streams=frozenset())
    assert policy.get() == Policy(), "and give it back afterwards"
