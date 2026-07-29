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
import itertools
from collections.abc import Callable, Hashable, Mapping, MutableMapping
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, TypedDict

import casty

from skyward.shared.codec import dumps, loads
from skyward.worker.api import NotOnANodeError, instance_info

type Consistency = Literal["strong", "eventual"]
"""How hard a write is acknowledged before the call returns.

- ``"strong"`` — the default: a majority of replicas must ack, so a value read
  back after a write is the value written, and it survives losing a minority.
- ``"eventual"`` — one replica acks and reads take the nearest copy. Cheaper, and
  enough when a collection is a scratchpad rather than a source of truth.
"""

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


type Params = Mapping[str, object]
type Backend = Callable[[str, str, str, tuple[object, ...], Params], object]
"""One call, addressed by (collection kind, name, method), its args and its build params.

The seam the subprocess executors bend: in the worker's own process the call reaches
casty directly; from a task running in a subprocess it rides a pipe back here first.
The collection classes below are the same on both sides — only what ``invoke`` resolves
to changes.
"""

_leases: MutableMapping[str, casty.Lease] = {}
"""Locks held on behalf of a subprocess, by token. The lease is not serialisable, so it
stays here and the far side holds only the token that names it."""
_tokens = itertools.count()


class _Regime(TypedDict, total=False):
    write: casty.Consistency
    read: casty.Consistency


def _regime(params: Params) -> _Regime:
    """The casty write/read acks for this call's consistency.

    ``"strong"`` is the empty mapping — casty's own defaults (majority write, one
    read), so a strong call is byte-for-byte the call it always was.
    """
    match params.get("consistency", "strong"):
        case "eventual":
            return {"write": casty.ONE, "read": casty.ONE}
        case _:
            return {}


async def _apply(
    current: Cluster,
    kind: str,
    name: str,
    method: str,
    args: tuple[object, ...],
    params: Params,
) -> object:
    match kind:
        case "map":
            return await getattr(current.system.map(name, replicas=current.replicas, **_regime(params)), method)(*args)
        case "set":
            return await getattr(current.system.set(name, replicas=current.replicas, **_regime(params)), method)(*args)
        case "counter":
            return await getattr(current.system.counter(name, replicas=current.replicas, **_regime(params)), method)(*args)
        case "queue":
            return await getattr(current.system.queue(name, replicas=current.replicas), method)(*args)
        case "barrier":
            parties = params["parties"]
            (timeout,) = args
            assert isinstance(parties, int)
            assert timeout is None or isinstance(timeout, int | float)
            return await current.system.barrier(name, parties=parties, replicas=current.replicas).wait(timeout)
        case "lock":
            return await _acquire_or_release(current, name, method, args, params)
        case _:
            raise ValueError(f"unknown collection {kind!r}")


async def _acquire_or_release(
    current: Cluster,
    name: str,
    method: str,
    args: tuple[object, ...],
    params: Params,
) -> object:
    match method:
        case "acquire":
            ttl, timeout = params["ttl"], params["timeout"]
            assert isinstance(ttl, int | float)
            assert timeout is None or isinstance(timeout, int | float)
            lock = current.system.lock(name, ttl=ttl, timeout=timeout, replicas=current.replicas)
            token = str(next(_tokens))
            _leases[token] = await lock.acquire()
            return token
        case "release":
            (token,) = args
            assert isinstance(token, str)
            lease = _leases.pop(token, None)
            if lease is not None:
                await lease.release()
            return None
        case _:
            raise ValueError(f"unknown lock method {method!r}")


def _local(kind: str, name: str, method: str, args: tuple[object, ...], params: Params) -> object:
    current = cluster()
    return asyncio.run_coroutine_threadsafe(_apply(current, kind, name, method, args, params), current.loop).result()


_backend: Backend = _local


def install(backend: Backend) -> None:
    """Route the collections somewhere other than this process's cluster.

    Called by the IPC bridge inside a subprocess executor, where ``cluster()`` is empty
    and every call has to reach back to the worker that spawned it.
    """
    global _backend
    _backend = backend


def invoke(kind: str, name: str, method: str, args: tuple[object, ...] = (), params: Params | None = None) -> object:
    return _backend(kind, name, method, args, params or {})


def _blob(value: object) -> bytes | None:
    if value is None or isinstance(value, bytes):
        return value
    raise TypeError(f"the collection bridge returned {type(value).__name__}, not bytes")


def _flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"the collection bridge returned {type(value).__name__}, not a bool")


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"the collection bridge returned {type(value).__name__}, not an int")
    return value


def _blobs(value: object) -> list[bytes]:
    if isinstance(value, list):
        return value
    raise TypeError(f"the collection bridge returned {type(value).__name__}, not a list")


def _pairs[K](value: object) -> list[tuple[K, bytes]]:
    if isinstance(value, list):
        return value
    raise TypeError(f"the collection bridge returned {type(value).__name__}, not a list")


class Dict[K: Hashable, V]:
    """A map every node sees, and that survives any one of them dying."""

    def __init__(self, name: str, consistency: Consistency = "strong") -> None:
        self._name = name
        self._params: Params = {"consistency": consistency}

    def __setitem__(self, key: K, value: V) -> None:
        invoke("map", self._name, "put", (key, dumps(value)), self._params)

    def __getitem__(self, key: K) -> V:
        raw = _blob(invoke("map", self._name, "get", (key,), self._params))
        if raw is None:
            raise KeyError(key)
        return loads(raw)

    def __contains__(self, key: K) -> bool:
        return _flag(invoke("map", self._name, "contains", (key,), self._params))

    def __len__(self) -> int:
        return _count(invoke("map", self._name, "size", (), self._params))

    def get(self, key: K, default: V | None = None) -> V | None:
        raw = _blob(invoke("map", self._name, "get", (key,), self._params))
        return default if raw is None else loads(raw)

    def pop(self, key: K) -> bool:
        return _flag(invoke("map", self._name, "remove", (key,), self._params))

    def items(self) -> list[tuple[K, V]]:
        return [(key, loads(raw)) for key, raw in _pairs(invoke("map", self._name, "items", (), self._params))]

    def clear(self) -> None:
        invoke("map", self._name, "clear", (), self._params)


class Set[T]:
    def __init__(self, name: str, consistency: Consistency = "strong") -> None:
        self._name = name
        self._params: Params = {"consistency": consistency}

    def add(self, item: T) -> bool:
        return _flag(invoke("set", self._name, "add", (dumps(item),), self._params))

    def remove(self, item: T) -> bool:
        return _flag(invoke("set", self._name, "remove", (dumps(item),), self._params))

    def __contains__(self, item: T) -> bool:
        return _flag(invoke("set", self._name, "contains", (dumps(item),), self._params))

    def __len__(self) -> int:
        return _count(invoke("set", self._name, "size", (), self._params))

    def items(self) -> list[T]:
        return [loads(raw) for raw in _blobs(invoke("set", self._name, "items", (), self._params))]

    def clear(self) -> None:
        invoke("set", self._name, "clear", (), self._params)


class Counter:
    def __init__(self, name: str, consistency: Consistency = "strong") -> None:
        self._name = name
        self._params: Params = {"consistency": consistency}

    def add(self, delta: int = 1) -> None:
        invoke("counter", self._name, "add", (delta,), self._params)

    def get(self) -> int:
        return _count(invoke("counter", self._name, "get", (), self._params))

    def reset(self) -> None:
        invoke("counter", self._name, "reset", (), self._params)


class Queue[T]:
    def __init__(self, name: str) -> None:
        self._name = name

    def offer(self, item: T) -> None:
        invoke("queue", self._name, "offer", (dumps(item),))

    def poll(self) -> T | None:
        """The next item, or nothing. It does not block: an empty queue is an answer."""
        raw = _blob(invoke("queue", self._name, "poll"))
        return None if raw is None else loads(raw)

    def __len__(self) -> int:
        return _count(invoke("queue", self._name, "size"))

    def clear(self) -> None:
        invoke("queue", self._name, "clear")


class Barrier:
    """Everybody waits until ``parties`` of them are here."""

    def __init__(self, name: str, parties: int) -> None:
        self._name = name
        self._parties = parties

    def wait(self, timeout: float | None = None) -> None:
        invoke("barrier", self._name, "wait", (timeout,), {"parties": self._parties})


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
        self._token: str | None = None

    def __enter__(self) -> Lock:
        held = invoke("lock", self._name, "acquire", (), {"ttl": self._ttl, "timeout": self._timeout})
        assert isinstance(held, str)
        self._token = held
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if self._token is not None:
            invoke("lock", self._name, "release", (self._token,))
            self._token = None


class DistributedRegistry[K: Hashable, V]:
    """A named place to leave objects for the other nodes to find.

        models = sky.registry("checkpoints")
        models.register(step, model)          # on the node that trained it
        model = models.lookup(latest_step)    # on any node that wants it

    A map with a directory's vocabulary — ``register`` a value under a name,
    ``lookup`` it from anywhere, ``list`` the names on offer. It replicates and
    survives a dead node the same way the other collections do.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def register(self, key: K, value: V) -> None:
        invoke("map", self._name, "put", (key, dumps(value)))

    def lookup(self, key: K) -> V | None:
        raw = _blob(invoke("map", self._name, "get", (key,)))
        return None if raw is None else loads(raw)

    def unregister(self, key: K) -> bool:
        return _flag(invoke("map", self._name, "remove", (key,)))

    def list(self) -> list[K]:
        return [key for key, _ in _pairs(invoke("map", self._name, "items"))]


def dict[K: Hashable, V](name: str, consistency: Consistency = "strong") -> Dict[K, V]:  # noqa: A001
    return Dict(name, consistency)


def set[T](name: str, consistency: Consistency = "strong") -> Set[T]:  # noqa: A001
    return Set(name, consistency)


def counter(name: str, consistency: Consistency = "strong") -> Counter:
    return Counter(name, consistency)


def registry[K: Hashable, V](name: str) -> DistributedRegistry[K, V]:
    return DistributedRegistry(name)


def queue[T](name: str) -> Queue[T]:
    return Queue(name)


def barrier(name: str, parties: int) -> Barrier:
    return Barrier(name, parties)


def lock(name: str, ttl: float = 30.0, timeout: float | None = None) -> Lock:
    return Lock(name, ttl, timeout)
