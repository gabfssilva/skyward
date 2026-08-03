"""What a function may ask while it is running on a node.

The only part of skyward the user's code imports on the far side. It answers from
the environment the node was launched with and from nothing else — no client, no
store, no call back to the daemon. A function that wants to know its rank is a
function in the middle of a computation, and it is not going to wait on a round
trip to find out.

Importable with nothing but the standard library, which is the point: a node has
no httpx and no business acquiring one.
"""

from __future__ import annotations

import io
import os
import random
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal, Protocol, cast, overload, runtime_checkable

from skyward.worker import slot

type Stream = Literal["stdout", "stderr"]


class NotOnANodeError(RuntimeError):
    """Asked on the machine that dispatched the work rather than the one running it."""


@dataclass(frozen=True, slots=True)
class Info:
    """The node, as the code running on it sees it.

    Attributes
    ----------
    node : str
        The node's id, the same one the daemon and the store use for it.
    compute : str
        The compute it belongs to.
    rank : int
        Its position among the compute's nodes, from zero. Stable for as long as
        the node lives — it is the rank a broadcast froze when it was admitted.
    peers : tuple[str, ...]
        Every node's address, rank-ordered, this one included. The addresses peers
        reach each other on, not the ones the daemon dials: the tunnels are the
        daemon's problem and exist nowhere out here.
    worker : int
        The task's index among the worker's concurrent slots. Zero in the
        worker/thread process, distinct per subprocess under ``executor='process'``.
    workers_per_node : int
        How many concurrent slots the worker runs, the node's ``concurrency``. One
        unless the node was told to run several tasks at once.
    """

    node: str
    compute: str
    rank: int
    peers: tuple[str, ...]
    worker: int = 0
    workers_per_node: int = 1

    @property
    def nodes(self) -> int:
        return len(self.peers)

    @property
    def total_workers(self) -> int:
        """Slots across the whole compute, every node's ``workers_per_node`` summed.

        Uniform here — every node of a compute runs the same concurrency — so it is
        the node count times the per-node count.
        """
        return self.nodes * self.workers_per_node

    @property
    def global_worker_index(self) -> int:
        """This slot's position among all of the compute's slots, from zero.

        The rank's block of ``workers_per_node`` offset by the local slot, so the
        slots of rank zero come first, then rank one's, and so on without a gap.
        """
        return self.rank * self.workers_per_node + self.worker

    @property
    def host(self) -> str:
        return self.peers[self.rank]

    @property
    def head(self) -> str:
        """Rank zero's address.

        A compute has no head. This is one, by convention, for the libraries that
        insist on being told where the rendezvous is.
        """
        return self.peers[0]

    @property
    def head_addr(self) -> str:
        """Rank zero's address, under the name the training libraries expect."""
        return self.head

    @property
    def head_port(self) -> int:
        """The port rank zero answers on, the same one every node's worker listens on."""
        from skyward.worker.worker import PORT

        return PORT

    @property
    def job_id(self) -> str:
        """The compute's id, under the name the libraries that want a job id expect."""
        return self.compute

    @property
    def is_head(self) -> bool:
        return self.rank == 0


def instance_info() -> Info:
    node = os.environ.get("SKYWARD_NODE")
    if node is None:
        raise NotOnANodeError("instance_info() is only answerable inside a function running on a node")

    return Info(
        node=node,
        compute=os.environ["SKYWARD_COMPUTE"],
        rank=int(os.environ["SKYWARD_RANK"]),
        peers=tuple(peer for peer in os.environ["SKYWARD_PEERS"].split(",") if peer),
        worker=slot.get(),
        workers_per_node=int(os.environ.get("SKYWARD_SLOTS", "1")),
    )


def is_head() -> bool:
    return instance_info().is_head


@runtime_checkable
class _Fancy[T](Protocol):
    """An array that takes a list of positions and stays itself.

    ``ndim`` is the tell: a numpy array or a torch tensor carries it and indexes by a
    list of positions, where a plain sequence does neither — which is what lets a
    shuffled shard of an array come back an array rather than a list.
    """

    @property
    def ndim(self) -> int: ...

    def __getitem__(self, index: list[int], /) -> Sequence[T]: ...


def _fancy_take[T](array: _Fancy[T], order: list[int]) -> Sequence[T]:
    return array[order]


def _take[T](data: Sequence[T], order: Sequence[int]) -> Sequence[T]:
    """The elements at ``order``, keeping the container's type where indexing can."""
    if isinstance(data, _Fancy):
        return _fancy_take(data, list(order))
    picked = [data[index] for index in order]
    return tuple(picked) if isinstance(data, tuple) else picked


def _shard[T](
    data: Sequence[T],
    rank: int,
    nodes: int,
    shuffle: bool,
    seed: int | str,
    drop_last: bool,
) -> Sequence[T]:
    total = len(data)
    if drop_last:
        total -= total % nodes

    start = total * rank // nodes
    stop = total * (rank + 1) // nodes

    if not shuffle:
        return data[start:stop]

    order = list(range(total))
    random.Random(seed).shuffle(order)
    return _take(data, order[start:stop])


@overload
def shard[T](
    data: Sequence[T],
    /,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> Sequence[T]: ...


@overload
def shard[T1, T2](
    data1: Sequence[T1],
    data2: Sequence[T2],
    /,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> tuple[Sequence[T1], Sequence[T2]]: ...


@overload
def shard[T1, T2, T3](
    data1: Sequence[T1],
    data2: Sequence[T2],
    data3: Sequence[T3],
    /,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> tuple[Sequence[T1], Sequence[T2], Sequence[T3]]: ...


def shard(
    *data: Sequence[object],
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> Sequence[object] | tuple[Sequence[object], ...]:
    """This node's slice of the data, without the data ever being split anywhere else.

    Every node is handed the whole sequence and keeps the part that is its own,
    which is the cheap way round when the data is already on the machines — a
    dataset mounted from S3, a file baked into the image — and the only way round
    that does not put the daemon in the path of every byte.

    The split is by rank, contiguous, and identical on every node: they agree
    because they each compute the same thing, not because anybody told them. Passing
    several sequences shards each the same way — the same rank split and, under
    ``shuffle``, the same permutation — so row ``i`` of one lines up with row ``i``
    of the next.

    Parameters
    ----------
    *data : Sequence
        One or more sequences. Anything with a length and a slice: a list stays a
        list, a tuple a tuple, a numpy array or torch tensor its own type — the
        contiguous slice and the fancy index both preserve it.
    shuffle : bool
        Permute before splitting. Deterministic given ``seed``, which is what makes
        the nodes' shards disjoint rather than merely random. Equal-length sequences
        get the same permutation, so their shards stay aligned.
    seed : int | None
        The permutation. ``None`` seeds from the compute, so that every node of one
        compute shuffles the same way and two computes do not.
    drop_last : bool
        Truncate to a multiple of the node count, so every node gets the same
        number of elements. What a training step usually wants, and what an
        unbalanced last batch usually breaks.
    node : int | None
        Override this node's rank instead of reading it off the node. For testing,
        or for sharding against a topology the node is not itself part of.
    total_nodes : int | None
        Override the node count. Given together with ``node``, the split needs no
        node to run on at all.

    Returns
    -------
    Sequence | tuple[Sequence, ...]
        This rank's elements — one sequence for one argument, a tuple of them for
        several. Possibly empty: eight nodes and three items is three nodes with
        work and five without, not an error.
    """
    if not data:
        raise ValueError("shard() needs at least one sequence")

    if node is None or total_nodes is None:
        info = instance_info()
        node = info.rank if node is None else node
        total_nodes = info.nodes if total_nodes is None else total_nodes
        default_seed: int | str = info.compute
    else:
        default_seed = 0

    key = seed if seed is not None else default_seed
    shards = tuple(_shard(one, node, total_nodes, shuffle, key, drop_last) for one in data)
    return shards[0] if len(shards) == 1 else shards


type Predicate = Callable[[Info], bool]
"""Given the node's own view of itself, whether its output makes the trip back."""

type OutputSpec = int | Sequence[int] | Literal["head"] | Predicate
"""Which nodes an output decorator keeps: ranks, ``"head"`` for rank zero, or a test."""


@dataclass(frozen=True, slots=True)
class Policy:
    """Which of a task's output makes the trip back.

    Everything, by default. The decorators below narrow it, and the journal on the
    node reads it — the filtering is done where the output is written, so what is
    silenced costs nothing rather than being shipped and then dropped.

    A ``where`` predicate travels inside the pickled task and is asked on the node,
    against the node's own :class:`Info`; ``ranks`` is the cheap case that needs no
    more than the rank the node already knows.
    """

    streams: frozenset[Stream] = frozenset({"stdout", "stderr"})
    ranks: frozenset[int] | None = None
    where: Predicate | None = None

    def allows(self, stream: Stream, info: Info) -> bool:
        if stream not in self.streams:
            return False
        if self.where is not None:
            return self.where(info)
        return self.ranks is None or info.rank in self.ranks


EVERYTHING = Policy()

policy: ContextVar[Policy] = ContextVar("policy", default=EVERYTHING)


def silent[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Say nothing. The result still comes back; the printing does not."""
    return _under(fn, Policy(streams=frozenset()))


@overload
def stdout[**P, T](fn: Callable[P, T]) -> Callable[P, T]: ...


@overload
def stdout[**P, T](*, only: OutputSpec) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def stdout[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    only: OutputSpec | None = None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Forward stdout and drop stderr, optionally from ``only`` some of the nodes.

    ``only`` is a rank, a sequence of ranks, ``"head"`` for rank zero, or a predicate
    on the node's own :class:`Info`. ``only="head"`` is what turns a broadcast onto
    sixty-four nodes back into something a human can read.
    """
    return _stream("stdout", fn, only)


@overload
def stderr[**P, T](fn: Callable[P, T]) -> Callable[P, T]: ...


@overload
def stderr[**P, T](*, only: OutputSpec) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def stderr[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    only: OutputSpec | None = None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Forward stderr and drop stdout, optionally from ``only`` some of the nodes."""
    return _stream("stderr", fn, only)


def _narrow(stream: Stream, only: OutputSpec | None) -> Policy:
    streams: frozenset[Stream] = frozenset({stream})
    match only:
        case None:
            return Policy(streams=streams)
        case "head":
            return Policy(streams=streams, ranks=frozenset({0}))
        case int():
            return Policy(streams=streams, ranks=frozenset({only}))
        case [*ranks]:
            return Policy(streams=streams, ranks=frozenset(ranks))
        case _:
            return Policy(streams=streams, where=only)


def _stream[**P, T](
    stream: Stream,
    fn: Callable[P, T] | None,
    only: OutputSpec | None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    wanted = _narrow(stream, only)

    def decorate(target: Callable[P, T]) -> Callable[P, T]:
        return _under(target, wanted)

    return decorate(fn) if fn else decorate


class _Under:
    """A callable that runs its function under a policy.

    A class rather than a closure so that pickling it never touches the module's
    ``policy`` ContextVar: instances carry only the function and the policy, and
    the code is found again by reference on the node.
    """

    def __init__(self, fn: Callable[..., Any], wanted: Policy) -> None:
        self._fn = fn
        self._wanted = wanted
        wraps(fn)(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        token = policy.set(self._wanted)
        try:
            return self._fn(*args, **kwargs)
        finally:
            policy.reset(token)


def _under[**P, T](fn: Callable[P, T], wanted: Policy) -> Callable[P, T]:
    return cast(Callable[P, T], _Under(fn, wanted))


class CallbackWriter(io.TextIOBase):
    """A writable stream that hands every write to a callback and keeps nothing."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        self._callback = callback

    def write(self, s: str, /) -> int:
        self._callback(s)
        return len(s)

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False


@contextmanager
def redirect_output(callback: Callable[[str], None]) -> Generator[tuple[CallbackWriter, CallbackWriter]]:
    """Send the running function's stdout and stderr to ``callback`` for the block.

    Both streams are redirected into a :class:`CallbackWriter`, so ``print`` and
    anything writing to ``sys.stderr`` reach ``callback`` a write at a time. Runs on
    the node, inside the task; the redirection ends with the block.

    Parameters
    ----------
    callback : Callable[[str], None]
        Called with each string written to either stream.

    Yields
    ------
    tuple[CallbackWriter, CallbackWriter]
        The ``(stdout, stderr)`` writers now standing in for the two streams.
    """
    out = CallbackWriter(callback)
    err = CallbackWriter(callback)
    with redirect_stdout(out), redirect_stderr(err):
        yield out, err
