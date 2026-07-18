"""The bridge that lets a task in a subprocess reach the compute's state.

A ``thread`` executor runs the user's function in the worker's own process, where
``sky.counter`` and its kin reach casty directly. A ``process`` or ``loky`` executor
runs it somewhere else — a subprocess with no cluster and no event loop — so the
collections have to come home.

They come home over a pipe. Each subprocess makes one, hands the worker its end
through a queue the pool shares, and keeps the other; a collection call in the
subprocess is a request down that pipe, answered by the worker running the very same
call against the cluster it holds. The subprocess never learns casty exists.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
import traceback
from collections.abc import Iterator, Mapping
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue

from skyward import distributed
from skyward.protocol.schemas import Executor as Kind

type Registrations = Queue[Connection]


@dataclass(frozen=True, slots=True)
class Call:
    """One collection call, as it crosses the pipe from a subprocess to the worker."""

    kind: str
    name: str
    method: str
    args: tuple[object, ...]
    params: Mapping[str, object]


def _install(registrations: Registrations) -> None:
    """Wire this subprocess to the worker. The pool's initializer, run once per child.

    The pipe is made here, in the child, and its far end shipped back up through the
    registration queue — the one primitive the pool already shares with every worker.
    """
    parent, child = multiprocessing.get_context("spawn").Pipe()
    registrations.put(parent)
    lock = threading.Lock()

    def backend(kind: str, name: str, method: str, args: tuple[object, ...], params: Mapping[str, object]) -> object:
        with lock:
            child.send(Call(kind, name, method, args, params))
            reply = child.recv()
        assert isinstance(reply, tuple)
        ok, payload = reply
        if ok:
            return payload
        raise RuntimeError(payload)

    distributed.install(backend)


class Bridge:
    """The worker's end: it answers every subprocess's collection calls.

    One thread waits on all the children's pipes at once and hands each request to a
    small pool, because a collection call runs on the worker's loop and blocks the
    thread that made it — one blocked answer must not hold up the next child's.
    """

    def __init__(self, registrations: Registrations) -> None:
        self._registrations = registrations
        self._stop = threading.Event()
        self._serving = threading.Thread(target=self._serve, name="skyward-ipc", daemon=True)
        self._dispatch = ThreadPoolExecutor(max_workers=8, thread_name_prefix="skyward-ipc-call")

    def start(self) -> None:
        self._serving.start()

    def close(self) -> None:
        self._stop.set()
        self._serving.join(timeout=5)
        self._dispatch.shutdown(wait=False)

    def _serve(self) -> None:
        live: set[Connection] = set()
        while not self._stop.is_set():
            while True:
                try:
                    live.add(self._registrations.get_nowait())
                except queue.Empty:
                    break
            if not live:
                self._stop.wait(0.1)
                continue
            for ready in wait(list(live), timeout=0.5):
                assert isinstance(ready, Connection)
                try:
                    request = ready.recv()
                except (EOFError, OSError):
                    live.discard(ready)
                    continue
                assert isinstance(request, Call)
                self._dispatch.submit(self._answer, ready, request)

    def _answer(self, conn: Connection, call: Call) -> None:
        try:
            value = distributed.invoke(call.kind, call.name, call.method, call.args, call.params)
            conn.send((True, value))
        except Exception as exc:
            conn.send((False, f"{exc}\n{traceback.format_exc()}"))


@contextmanager
def executor(kind: Kind, reuse: bool, workers: int) -> Iterator[Executor]:
    """The pool the tasks run on, and the bridge behind it if they run off-process.

    ``workers`` is the executor's width — how many tasks run at once. It is the
    ``concurrency`` the pool was asked for; the buffer lives above it, in how many
    calls casty admits, not in how many the pool runs.
    """
    if kind == "thread":
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            yield pool
        finally:
            pool.shutdown(wait=False)
        return

    spawn = multiprocessing.get_context("spawn")
    registrations: Registrations = spawn.Queue()
    bridge = Bridge(registrations)
    bridge.start()
    pool = _pool(kind, reuse, workers, spawn, registrations)
    try:
        yield pool
    finally:
        pool.shutdown(wait=False)
        bridge.close()


def _pool(kind: Kind, reuse: bool, workers: int, spawn: BaseContext, registrations: Registrations) -> Executor:
    match kind:
        case "loky":
            from loky import get_reusable_executor

            return get_reusable_executor(
                max_workers=workers,
                initializer=_install,
                initargs=(registrations,),
                reuse="auto",
            )
        case _ if reuse:
            return ProcessPoolExecutor(
                max_workers=workers,
                mp_context=spawn,
                initializer=_install,
                initargs=(registrations,),
            )
        case _:
            return ProcessPoolExecutor(
                max_workers=workers,
                mp_context=spawn,
                initializer=_install,
                initargs=(registrations,),
                max_tasks_per_child=1,
            )
