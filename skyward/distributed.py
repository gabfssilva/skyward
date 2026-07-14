"""State the nodes share, for as long as the compute does.

    counter = sky.counter("processed")
    counter.add(len(batch))

The compute is already a casty cluster — the workers found each other to be
callable, and a cluster that can route a call can hold a map. So these are casty's
collections, replicated across the nodes and quorum-acknowledged, and skyward's
part is the two things the user needs and casty does not provide: a synchronous
face, and values that are not msgpack.

Synchronous because the code holding them is: a training loop is a training loop.
The calls are handed to the worker's event loop from the thread the task runs on,
which is where they were always going to end up.

Values are pickled, keys are not. A value is a user's object — an array, a model,
a dataclass they wrote this morning — and there is no wire format for that but
theirs. A key has to route to a shard, so it stays something casty can hash: a
string, a number, a tuple of them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Hashable
from dataclasses import dataclass
from types import TracebackType

import casty

from skyward.protocol.codec import dumps, loads
from skyward.runtime.api import NotOnANodeError, instance_info

REPLICAS = 3
"""What a compute replicates to, when it is big enough to.

Three is the smallest number that survives losing one and still has a majority.
A quorum bigger than the cluster fences every write, so the actual figure is
``min(REPLICAS, nodes)`` — a one-node compute replicates to itself, and the
collections work there too.
"""


@dataclass(frozen=True, slots=True)
class Cluster:
    system: casty.ActorSystem
    loop: asyncio.AbstractEventLoop
    replicas: int


_cluster: Cluster | None = None


def bind(system: casty.ActorSystem, loop: asyncio.AbstractEventLoop) -> None:
    """Hand the worker's cluster to the user's code. Called once, by the worker."""
    global _cluster
    _cluster = Cluster(system, loop, replicas=min(REPLICAS, instance_info().nodes))


def unbind() -> None:
    global _cluster
    _cluster = None


def cluster() -> Cluster:
    if _cluster is None:
        raise NotOnANodeError("distributed collections live in the compute, and are only reachable from inside one")
    return _cluster


def run[T](coro: Coroutine[object, object, T]) -> T:
    """Await, from the thread the user's function is running on.

    The collections are asynchronous and the caller is not. The loop that owns the
    cluster is the worker's, in this process, and this is the only door into it.
    """
    return asyncio.run_coroutine_threadsafe(coro, cluster().loop).result()


class Dict[K: Hashable, V]:
    """A map every node sees, and that survives any one of them dying."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def _map(self) -> casty.Map[K, bytes]:
        current = cluster()
        return current.system.map(self._name, replicas=current.replicas)

    def __setitem__(self, key: K, value: V) -> None:
        run(self._map.put(key, dumps(value)))

    def __getitem__(self, key: K) -> V:
        raw = run(self._map.get(key))
        if raw is None:
            raise KeyError(key)
        return loads(raw)

    def __contains__(self, key: K) -> bool:
        return run(self._map.contains(key))

    def __len__(self) -> int:
        return run(self._map.size())

    def get(self, key: K, default: V | None = None) -> V | None:
        raw = run(self._map.get(key))
        return default if raw is None else loads(raw)

    def pop(self, key: K) -> bool:
        return run(self._map.remove(key))

    def items(self) -> list[tuple[K, V]]:
        return [(key, loads(raw)) for key, raw in run(self._map.items())]

    def clear(self) -> None:
        run(self._map.clear())


class Set[T]:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def _set(self) -> casty.Set[bytes]:
        current = cluster()
        return current.system.set(self._name, replicas=current.replicas)

    def add(self, item: T) -> bool:
        return run(self._set.add(dumps(item)))

    def remove(self, item: T) -> bool:
        return run(self._set.remove(dumps(item)))

    def __contains__(self, item: T) -> bool:
        return run(self._set.contains(dumps(item)))

    def __len__(self) -> int:
        return run(self._set.size())

    def items(self) -> list[T]:
        return [loads(raw) for raw in run(self._set.items())]

    def clear(self) -> None:
        run(self._set.clear())


class Counter:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def _counter(self) -> casty.Counter:
        current = cluster()
        return current.system.counter(self._name, replicas=current.replicas)

    def add(self, delta: int = 1) -> None:
        run(self._counter.add(delta))

    def get(self) -> int:
        return run(self._counter.get())

    def reset(self) -> None:
        run(self._counter.reset())


class Queue[T]:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def _queue(self) -> casty.Queue[bytes]:
        current = cluster()
        return current.system.queue(self._name, replicas=current.replicas)

    def offer(self, item: T) -> None:
        run(self._queue.offer(dumps(item)))

    def poll(self) -> T | None:
        """The next item, or nothing. It does not block: an empty queue is an answer."""
        raw = run(self._queue.poll())
        return None if raw is None else loads(raw)

    def __len__(self) -> int:
        return run(self._queue.size())

    def clear(self) -> None:
        run(self._queue.clear())


class Barrier:
    """Everybody waits until ``parties`` of them are here."""

    def __init__(self, name: str, parties: int) -> None:
        self._name = name
        self._parties = parties

    def wait(self, timeout: float | None = None) -> None:
        current = cluster()
        barrier = current.system.barrier(self._name, parties=self._parties, replicas=current.replicas)
        run(barrier.wait(timeout))


class Lock:
    """One holder at a time, across the compute.

        with sky.lock("checkpoint"):
            save(model)

    The lease has a time to live, so a node that dies holding the lock hands it
    back — there is no other way to get it back from a machine that is gone.
    """

    def __init__(self, name: str, ttl: float = 30.0, timeout: float | None = None) -> None:
        self._name = name
        self._ttl = ttl
        self._timeout = timeout
        self._lease: casty.Lease | None = None

    def __enter__(self) -> Lock:
        current = cluster()
        lock = current.system.lock(self._name, ttl=self._ttl, timeout=self._timeout, replicas=current.replicas)
        self._lease = run(lock.acquire())
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if self._lease:
            run(self._lease.release())
            self._lease = None


def dict[K: Hashable, V](name: str) -> Dict[K, V]:  # noqa: A001
    return Dict(name)


def set[T](name: str) -> Set[T]:  # noqa: A001
    return Set(name)


def counter(name: str) -> Counter:
    return Counter(name)


def queue[T](name: str) -> Queue[T]:
    return Queue(name)


def barrier(name: str, parties: int) -> Barrier:
    return Barrier(name, parties)


def lock(name: str, ttl: float = 30.0, timeout: float | None = None) -> Lock:
    return Lock(name, ttl, timeout)
