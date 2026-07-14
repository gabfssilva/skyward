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

import os
import random
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Literal, overload

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
    """

    node: str
    compute: str
    rank: int
    peers: tuple[str, ...]

    @property
    def nodes(self) -> int:
        return len(self.peers)

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
    )


def is_head() -> bool:
    return instance_info().is_head


def shard[T](
    data: Sequence[T],
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
) -> Sequence[T]:
    """This node's slice of the data, without the data ever being split anywhere else.

    Every node is handed the whole sequence and keeps the part that is its own,
    which is the cheap way round when the data is already on the machines — a
    dataset mounted from S3, a file baked into the image — and the only way round
    that does not put the daemon in the path of every byte.

    The split is by rank, contiguous, and identical on every node: they agree
    because they each compute the same thing, not because anybody told them.

    Parameters
    ----------
    data : Sequence[T]
        Anything with a length and a slice. Contiguous slicing is what preserves
        the type — a list stays a list, an array stays an array — so ``shuffle``
        is the one case that falls back to a list, since the elements it wants are
        no longer next to each other.
    shuffle : bool
        Permute before splitting. Deterministic given ``seed``, which is what makes
        the nodes' shards disjoint rather than merely random.
    seed : int | None
        The permutation. ``None`` seeds from the compute, so that every node of one
        compute shuffles the same way and two computes do not.
    drop_last : bool
        Truncate to a multiple of the node count, so every node gets the same
        number of elements. What a training step usually wants, and what an
        unbalanced last batch usually breaks.

    Returns
    -------
    Sequence[T]
        This rank's elements. Possibly empty: eight nodes and three items is three
        nodes with work and five without, not an error.
    """
    info = instance_info()
    total = len(data)

    if drop_last:
        total -= total % info.nodes

    if not shuffle:
        start = total * info.rank // info.nodes
        stop = total * (info.rank + 1) // info.nodes
        return data[start:stop]

    order = list(range(total))
    random.Random(seed if seed is not None else info.compute).shuffle(order)
    start = total * info.rank // info.nodes
    stop = total * (info.rank + 1) // info.nodes
    return [data[index] for index in order[start:stop]]


@dataclass(frozen=True, slots=True)
class Policy:
    """Which of a task's output makes the trip back.

    Everything, by default. The decorators below narrow it, and the journal on the
    node reads it — the filtering is done where the output is written, so what is
    silenced costs nothing rather than being shipped and then dropped.
    """

    streams: frozenset[Stream] = frozenset({"stdout", "stderr"})
    ranks: frozenset[int] | None = None

    def allows(self, stream: Stream, rank: int) -> bool:
        return stream in self.streams and (self.ranks is None or rank in self.ranks)


EVERYTHING = Policy()

policy: ContextVar[Policy] = ContextVar("policy", default=EVERYTHING)


def rank() -> int:
    """The node's rank, as the journal needs it: cheap, and answerable off a node."""
    return int(os.environ.get("SKYWARD_RANK", "0"))


def silent[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Say nothing. The result still comes back; the printing does not."""
    return _under(fn, Policy(streams=frozenset()))


@overload
def stdout[**P, T](fn: Callable[P, T]) -> Callable[P, T]: ...


@overload
def stdout[**P, T](*, only: int | tuple[int, ...]) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def stdout[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    only: int | tuple[int, ...] | None = None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Forward stdout and drop stderr, optionally from ``only`` these ranks.

    ``only=0`` is what turns a broadcast onto sixty-four nodes back into something
    a human can read.
    """
    return _stream("stdout", fn, only)


@overload
def stderr[**P, T](fn: Callable[P, T]) -> Callable[P, T]: ...


@overload
def stderr[**P, T](*, only: int | tuple[int, ...]) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def stderr[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    only: int | tuple[int, ...] | None = None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Forward stderr and drop stdout, optionally from ``only`` these ranks."""
    return _stream("stderr", fn, only)


def _stream[**P, T](
    stream: Stream,
    fn: Callable[P, T] | None,
    only: int | tuple[int, ...] | None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    ranks = None if only is None else frozenset(only if isinstance(only, tuple) else (only,))
    wanted = Policy(streams=frozenset({stream}), ranks=ranks)

    def decorate(target: Callable[P, T]) -> Callable[P, T]:
        return _under(target, wanted)

    return decorate(fn) if fn else decorate


def _under[**P, T](fn: Callable[P, T], wanted: Policy) -> Callable[P, T]:
    @wraps(fn)
    def run(*args: P.args, **kwargs: P.kwargs) -> T:
        token = policy.set(wanted)
        try:
            return fn(*args, **kwargs)
        finally:
            policy.reset(token)

    return run
