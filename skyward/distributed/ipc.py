"""IPC bridge for distributed collections in subprocess executors.

When Worker(executor="process") runs tasks via ProcessPoolExecutor,
subprocesses cannot access the parent's ClusteredActorSystem. This module
provides:

- IPCRegistry / IPC proxy classes — subprocess side, sends requests over a Pipe
- start_bridge() — parent side, dispatches requests to the real registry
- ipc_initializer() — ProcessPoolExecutor initializer that wires up each subprocess

Architecture:

    Subprocess                         Parent Worker
    ┌──────────────┐                  ┌─────────────────────┐
    │ sky.dict("x") │                  │ DistributedRegistry  │
    │  → IPCRegistry │                  │  (real Casty proxies)│
    │    → IPCDictProxy                │                     │
    │      → conn.send(req) ──Pipe──→ │ Bridge Thread        │
    │      ← conn.recv(resp) ←─Pipe── │   dispatch to proxy  │
    └──────────────┘                  └─────────────────────┘
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any, cast

from skyward.observability.logger import logger

from .types import Consistency

log = logger.bind(component="ipc-bridge")


# ── Protocol messages ────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class IPCRequest:
    collection_type: str
    collection_name: str
    method: str
    args: tuple[Any, ...] = ()
    create_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IPCResponse:
    value: Any = None
    error: str | None = None
    traceback: str | None = None


# ── Subprocess helpers ───────────────────────────────────────────────

def _ipc_call(
    conn: Connection,
    collection_type: str,
    collection_name: str,
    method: str,
    args: tuple[Any, ...] = (),
    create_kwargs: dict[str, Any] | None = None,
) -> Any:
    request = IPCRequest(
        collection_type=collection_type,
        collection_name=collection_name,
        method=method,
        args=args,
        create_kwargs=create_kwargs or {},
    )
    conn.send(request)
    response: IPCResponse = conn.recv()
    if response.error is not None:
        raise RuntimeError(
            f"IPC bridge error ({collection_type}.{method}): {response.error}"
            + (f"\nRemote traceback:\n{response.traceback}" if response.traceback else "")
        )
    return response.value


# ── IPC Proxy classes (subprocess side) ──────────────────────────────

class IPCCounterProxy:
    __slots__ = ("_conn", "_name", "_create_kwargs")

    def __init__(self, conn: Connection, name: str, consistency: Consistency | None) -> None:
        self._conn = conn
        self._name = name
        self._create_kwargs = {"consistency": consistency}

    @property
    def value(self) -> int:
        return _ipc_call(
            self._conn, "counter", self._name, "value",
            create_kwargs=self._create_kwargs,
        )

    def increment(self, n: int = 1) -> None:
        _ipc_call(
            self._conn, "counter", self._name, "increment", (n,),
            create_kwargs=self._create_kwargs,
        )

    def decrement(self, n: int = 1) -> None:
        _ipc_call(
            self._conn, "counter", self._name, "decrement", (n,),
            create_kwargs=self._create_kwargs,
        )

    def reset(self, value: int = 0) -> None:
        _ipc_call(
            self._conn, "counter", self._name, "reset", (value,),
            create_kwargs=self._create_kwargs,
        )

    def __int__(self) -> int:
        return self.value


class IPCDictProxy:
    __slots__ = ("_conn", "_name", "_create_kwargs")

    def __init__(self, conn: Connection, name: str, consistency: Consistency | None) -> None:
        self._conn = conn
        self._name = name
        self._create_kwargs = {"consistency": consistency}

    def __getitem__(self, key: str) -> Any:
        return _ipc_call(
            self._conn, "dict", self._name, "__getitem__", (key,),
            create_kwargs=self._create_kwargs,
        )

    def __setitem__(self, key: str, value: Any) -> None:
        _ipc_call(
            self._conn, "dict", self._name, "__setitem__", (key, value),
            create_kwargs=self._create_kwargs,
        )

    def __delitem__(self, key: str) -> None:
        _ipc_call(
            self._conn, "dict", self._name, "__delitem__", (key,),
            create_kwargs=self._create_kwargs,
        )

    def __contains__(self, key: str) -> bool:
        return _ipc_call(
            self._conn, "dict", self._name, "__contains__", (key,),
            create_kwargs=self._create_kwargs,
        )

    def get(self, key: str, default: Any = None) -> Any:
        return _ipc_call(
            self._conn, "dict", self._name, "get", (key, default),
            create_kwargs=self._create_kwargs,
        )

    def update(self, items: dict[str, Any]) -> None:
        _ipc_call(
            self._conn, "dict", self._name, "update", (items,),
            create_kwargs=self._create_kwargs,
        )

    def pop(self, key: str, default: Any = None) -> Any:
        return _ipc_call(
            self._conn, "dict", self._name, "pop", (key, default),
            create_kwargs=self._create_kwargs,
        )


class IPCSetProxy:
    __slots__ = ("_conn", "_name", "_create_kwargs")

    def __init__(self, conn: Connection, name: str, consistency: Consistency | None) -> None:
        self._conn = conn
        self._name = name
        self._create_kwargs = {"consistency": consistency}

    def __contains__(self, value: Any) -> bool:
        return _ipc_call(
            self._conn, "set", self._name, "__contains__", (value,),
            create_kwargs=self._create_kwargs,
        )

    def __len__(self) -> int:
        return _ipc_call(
            self._conn, "set", self._name, "__len__",
            create_kwargs=self._create_kwargs,
        )

    def add(self, value: Any) -> None:
        _ipc_call(
            self._conn, "set", self._name, "add", (value,),
            create_kwargs=self._create_kwargs,
        )

    def discard(self, value: Any) -> None:
        _ipc_call(
            self._conn, "set", self._name, "discard", (value,),
            create_kwargs=self._create_kwargs,
        )


class IPCQueueProxy:
    __slots__ = ("_conn", "_name")

    def __init__(self, conn: Connection, name: str) -> None:
        self._conn = conn
        self._name = name

    def __len__(self) -> int:
        return _ipc_call(self._conn, "queue", self._name, "__len__")

    def put(self, value: Any) -> None:
        _ipc_call(self._conn, "queue", self._name, "put", (value,))

    def get(self, timeout: float | None = None) -> Any:
        return _ipc_call(self._conn, "queue", self._name, "get", (timeout,))

    def empty(self) -> bool:
        return _ipc_call(self._conn, "queue", self._name, "empty")


class IPCBarrierProxy:
    __slots__ = ("_conn", "_name", "_create_kwargs")

    def __init__(self, conn: Connection, name: str, n: int) -> None:
        self._conn = conn
        self._name = name
        self._create_kwargs = {"n": n}

    def wait(self) -> None:
        _ipc_call(
            self._conn, "barrier", self._name, "wait",
            create_kwargs=self._create_kwargs,
        )

    def reset(self) -> None:
        _ipc_call(
            self._conn, "barrier", self._name, "reset",
            create_kwargs=self._create_kwargs,
        )


class IPCLockProxy:
    __slots__ = ("_conn", "_name")

    def __init__(self, conn: Connection, name: str) -> None:
        self._conn = conn
        self._name = name

    def acquire(self) -> bool:
        return _ipc_call(self._conn, "lock", self._name, "acquire")

    def release(self) -> None:
        _ipc_call(self._conn, "lock", self._name, "release")

    def __enter__(self) -> IPCLockProxy:
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


# ── IPCRegistry (subprocess side) ────────────────────────────────────

class IPCRegistry:
    """Registry that proxies all operations over a multiprocessing Pipe."""

    __slots__ = ("_conn",)

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def dict(self, name: str, *, consistency: Consistency | None = None) -> IPCDictProxy:
        return IPCDictProxy(self._conn, name, consistency)

    def set(self, name: str, *, consistency: Consistency | None = None) -> IPCSetProxy:
        return IPCSetProxy(self._conn, name, consistency)

    def counter(self, name: str, *, consistency: Consistency | None = None) -> IPCCounterProxy:
        return IPCCounterProxy(self._conn, name, consistency)

    def queue(self, name: str) -> IPCQueueProxy:
        return IPCQueueProxy(self._conn, name)

    def barrier(self, name: str, n: int) -> IPCBarrierProxy:
        return IPCBarrierProxy(self._conn, name, n)

    def lock(self, name: str) -> IPCLockProxy:
        return IPCLockProxy(self._conn, name)

    def cleanup(self) -> None:
        pass


# ── Bridge (parent side) ─────────────────────────────────────────────

def _dispatch(
    request: IPCRequest,
    registry: Any,
    proxy_cache: dict[tuple[str, str], Any],
) -> IPCResponse:
    try:
        cache_key = (request.collection_type, request.collection_name)

        if cache_key not in proxy_cache:
            factory = getattr(registry, request.collection_type)
            match request.collection_type:
                case "barrier":
                    n = request.create_kwargs.get("n", 1)
                    proxy_cache[cache_key] = factory(request.collection_name, n)
                case "dict" | "set" | "counter":
                    consistency = request.create_kwargs.get("consistency")
                    proxy_cache[cache_key] = factory(
                        request.collection_name, consistency=consistency,
                    )
                case _:
                    proxy_cache[cache_key] = factory(request.collection_name)

        proxy = proxy_cache[cache_key]

        match request.method:
            case "value" if request.collection_type == "counter":
                result = proxy.value
            case _:
                method = getattr(proxy, request.method)
                result = method(*request.args)

        return IPCResponse(value=result)
    except Exception as e:
        return IPCResponse(error=str(e), traceback=traceback.format_exc())


def _bridge_loop(
    registration_queue: Any,
    registry: Any,
    stop_event: threading.Event,
) -> None:
    proxy_cache: dict[tuple[str, str], Any] = {}
    cache_lock = threading.Lock()
    dispatch_pool = ThreadPoolExecutor(max_workers=4)
    alive: set[Connection] = set()

    def _handle(conn: Connection, request: IPCRequest) -> None:
        with cache_lock:
            response = _dispatch(request, registry, proxy_cache)
        try:
            conn.send(response)
        except (BrokenPipeError, EOFError, OSError):
            alive.discard(conn)

    try:
        while not stop_event.is_set():
            while not registration_queue.empty():
                try:
                    parent_conn: Connection = registration_queue.get_nowait()
                    alive.add(parent_conn)
                    log.debug("Registered new IPC pipe (total={n})", n=len(alive))
                except Exception:
                    break

            if not alive:
                stop_event.wait(timeout=0.5)
                continue

            ready = wait(list(alive), timeout=1.0)
            for obj in ready:
                c = cast(Connection, obj)
                try:
                    request: IPCRequest = c.recv()
                    dispatch_pool.submit(_handle, c, request)
                except (EOFError, OSError):
                    alive.discard(c)
    finally:
        dispatch_pool.shutdown(wait=False)


def start_bridge(
    registration_queue: Any,
    registry: Any,
) -> threading.Event:
    """Start the IPC bridge daemon thread.

    Parameters
    ----------
    registration_queue
        Queue where subprocess initializers register parent-side Pipe connections.
        Each new worker (including loky respawns) creates a Pipe and puts the
        parent end here for the bridge to monitor.
    registry
        The real DistributedRegistry backed by Casty.

    Returns
    -------
    threading.Event
        Set this event to signal the bridge thread to stop.
    """
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_bridge_loop,
        args=(registration_queue, registry, stop_event),
        daemon=True,
        name="ipc-bridge",
    )
    thread.start()
    log.info("IPC bridge started")
    return stop_event


# ── ProcessPoolExecutor initializer ──────────────────────────────────

_subprocess_conn: Connection | None = None


def ipc_initializer(registration_queue: Any) -> None:
    """Loky worker initializer — creates a Pipe and registers with the bridge.

    Called once per worker process (including respawns). Each worker creates
    its own Pipe, sends the parent end to the bridge via the registration queue,
    and keeps the child end for IPC.
    """
    import multiprocessing
    import sys

    global _subprocess_conn

    from skyward.distributed import _set_active_registry

    parent_conn, child_conn = multiprocessing.Pipe()
    registration_queue.put(parent_conn)
    _subprocess_conn = child_conn
    _set_active_registry(IPCRegistry(child_conn))

    # Loky subprocesses inherit fd 1/2 pointing to casty.log (via shell redirect),
    # but Python defaults to block buffering for non-TTY fds (~8KB buffer).
    # Without line buffering, print() output stays in the buffer and never reaches
    # casty.log → tail → events.jsonl. Reconfigure to match the parent's behavior.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
