"""What a function can ask about the node it is running on.

The environment is the whole input, so the tests set it and nothing else — there is
no node here, and there does not need to be one.
"""

import pytest

from skyward.worker.api import (
    Info,
    NotOnANodeError,
    Policy,
    instance_info,
    is_head,
    redirect_output,
    shard,
    silent,
    stdout,
)


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch):
    def place(rank: int, peers: int = 4, slots: int = 1):
        monkeypatch.setenv("SKYWARD_NODE", f"nod_{rank}")
        monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_1")
        monkeypatch.setenv("SKYWARD_RANK", str(rank))
        monkeypatch.setenv("SKYWARD_PEERS", ",".join(f"10.0.0.{index}" for index in range(peers)))
        monkeypatch.setenv("SKYWARD_SLOTS", str(slots))

    return place


def test_a_node_knows_where_it_is_among_the_others(node):
    node(rank=2)
    info = instance_info()

    assert (info.rank, info.nodes) == (2, 4)
    assert info.host == "10.0.0.2", "a node finds itself at its own rank"
    assert info.head == "10.0.0.0"
    assert not info.is_head and not is_head()


def test_the_head_is_rank_zero_at_the_worker_port(node):
    from skyward.worker.worker import PORT

    node(rank=1)
    info = instance_info()

    assert info.head_addr == "10.0.0.0" == info.head
    assert info.head_port == PORT
    assert info.job_id == "cmp_1" == info.compute


def test_slots_give_every_node_a_block_of_global_worker_indices(node):
    node(rank=0, slots=2)
    assert instance_info().workers_per_node == 2

    seen = []
    for rank in range(4):
        for worker in range(2):
            node(rank=rank, slots=2)
            info = Info(node=f"nod_{rank}", compute="cmp_1", rank=rank, peers=info_peers(4), worker=worker, workers_per_node=2)
            assert info.total_workers == 8
            seen.append(info.global_worker_index)

    assert seen == list(range(8)), "the ranks' slot blocks tile the range without a gap"


def test_workers_per_node_defaults_to_one_when_unset(monkeypatch: pytest.MonkeyPatch, node):
    node(rank=0)
    monkeypatch.delenv("SKYWARD_SLOTS", raising=False)

    info = instance_info()
    assert info.workers_per_node == 1
    assert info.total_workers == 4
    assert info.global_worker_index == 0


def info_peers(nodes: int) -> tuple[str, ...]:
    return tuple(f"10.0.0.{index}" for index in range(nodes))


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


def _info(rank: int, nodes: int = 4) -> Info:
    return Info(
        node=f"nod_{rank}",
        compute="cmp_1",
        rank=rank,
        peers=tuple(f"10.0.0.{index}" for index in range(nodes)),
    )


def test_a_policy_narrows_by_stream_and_by_rank():
    everything = Policy()
    assert everything.allows("stdout", _info(rank=3)) and everything.allows("stderr", _info(rank=3))

    quiet = Policy(streams=frozenset({"stdout"}), ranks=frozenset({0}))
    assert quiet.allows("stdout", _info(rank=0))
    assert not quiet.allows("stderr", _info(rank=0))
    assert not quiet.allows("stdout", _info(rank=1))


def test_a_policy_can_narrow_by_a_predicate_on_the_node():
    only_even = Policy(where=lambda info: info.rank % 2 == 0)
    assert only_even.allows("stdout", _info(rank=2))
    assert not only_even.allows("stdout", _info(rank=3))


def test_the_decorators_hold_their_policy_only_while_the_function_runs():
    from skyward.worker.api import policy

    @stdout(only=0)
    def loud() -> Policy:
        return policy.get()

    @silent
    def quiet() -> Policy:
        return policy.get()

    assert loud() == Policy(streams=frozenset({"stdout"}), ranks=frozenset({0}))
    assert quiet() == Policy(streams=frozenset())
    assert policy.get() == Policy(), "and give it back afterwards"


def test_paired_sequences_shard_with_the_same_split(node):
    node(rank=1)
    x = list(range(10))
    y = [value * 10 for value in range(10)]

    xs, ys = shard(x, y)

    assert list(xs) == [2, 3, 4], "rank one's contiguous slice of ten over four"
    assert [value * 10 for value in xs] == list(ys), "row i of x lines up with row i of y"


def test_a_shuffle_keeps_paired_sequences_aligned(node):
    x = list(range(12))
    y = [value * 10 for value in range(12)]

    for rank in range(4):
        node(rank=rank)
        xs, ys = shard(x, y, shuffle=True, seed=3)
        assert [value * 10 for value in xs] == list(ys), "the same permutation across both"


def test_overrides_shard_without_a_node(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SKYWARD_NODE", raising=False)
    data = list(range(10))

    assert list(shard(data, node=1, total_nodes=4)) == [2, 3, 4], "the split needs no node when told the topology"


def test_stdout_only_head_keeps_rank_zero():
    from skyward.worker.api import policy

    @stdout(only="head")
    def loud() -> Policy:
        return policy.get()

    seen = loud()
    assert seen.streams == frozenset({"stdout"})
    assert seen.ranks == frozenset({0})
    assert seen.where is None


def test_stdout_only_a_predicate_travels_in_the_policy():
    from skyward.worker.api import policy

    def keep(info: Info) -> bool:
        return info.rank in (0, 2)

    @stdout(only=keep)
    def loud() -> Policy:
        return policy.get()

    seen = loud()
    assert seen.where is keep
    assert seen.allows("stdout", _info(rank=2))
    assert not seen.allows("stdout", _info(rank=1))


def test_redirect_output_captures_stdout_and_stderr():
    import sys

    chunks: list[str] = []
    with redirect_output(chunks.append):
        print("out")
        print("err", file=sys.stderr)

    assert "".join(chunks) == "out\nerr\n"
