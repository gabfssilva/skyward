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

import asyncio
import multiprocessing
import os
import queue
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from concurrent.futures import BrokenExecutor, Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import Connection, wait
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
from typing import Literal

from skyward.worker import distributed, slot, stopping

type Kind = Literal["process", "loky"]

type Registrations = Queue[Connection]
type Slots = Queue[int]

SLOT_TIMEOUT = 2.0
"""Seconds a child waits for its index before starting without a distinct one."""

LOKY_IDLE_SECONDS = 24 * 60 * 60
"""How long an idle loky worker waits for work before it exits; loky's own default is ten seconds."""


@dataclass(frozen=True, slots=True)
class Call:
    """One collection call, as it crosses the pipe from a subprocess to the worker."""

    kind: str
    name: str
    method: str
    args: tuple[object, ...]
    params: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Running:
    """A subprocess saying which attempt it has just started, or with ``pid`` ``None``, just finished."""

    id: str
    pid: int | None


type Tell = Callable[[Running], bool]

_tell: Tell | None = None
"""This subprocess's way to tell the worker what it is running. Set by :func:`_install`."""


@contextmanager
def attempt(id: str) -> Iterator[None]:
    """Run the block as attempt ``id`` in this subprocess, where the worker can stop it.

    The worker is told which process holds the attempt before it starts, and told
    again once it is over, so a stop is only ever sent to the process running what it
    is meant for. An attempt the worker was already asked to stop is answered for
    here, before any of the user's code runs.
    """
    with stopping.current(id):
        try:
            if _tell is not None and _tell(Running(id, os.getpid())):
                raise stopping.Stop
            yield
        finally:
            if _tell is not None:
                _tell(Running(id, None))


def _install(registrations: Registrations, slots: Slots | None = None) -> None:
    """Wire this subprocess to the worker. The pool's initializer, run once per child.

    The pipe is made here, in the child, and its far end shipped back up through the
    registration queue — the one primitive the pool already shares with every worker.

    ``slots`` carries this child's index among the pool's workers, one per child, and
    is only handed over by the reused pools: there each child is one of a fixed
    ``workers`` set and takes a distinct index that outlives its every task. Under
    ``reuse=False`` the children are transient and unbounded — a fixed range would
    drain — so ``slots`` is ``None`` and the index stays at its default zero.

    The read waits, briefly. The queue is filled by the parent before the pool is
    built, but a multiprocessing queue is written by a feeder thread, so a child
    that spawns quickly can reach an empty queue that is about to have its index in
    it. Not waiting meant that child silently keeping index zero — two workers on
    the same share of the machine, which is the one thing the index exists to
    prevent. What must not happen is waiting forever: a child the pool replaces
    later has no index left to take, and falls back to zero rather than never
    starting.
    """
    if slots is not None:
        with suppress(queue.Empty):
            slot.set(slots.get(timeout=SLOT_TIMEOUT))

    global _tell
    stopping.arm()
    parent, child = multiprocessing.get_context("spawn").Pipe()
    registrations.put(parent)
    lock = threading.Lock()

    def exchange(request: Call | Running) -> tuple[bool, object]:
        with stopping.deferred(), lock:
            child.send(request)
            reply = child.recv()
        assert isinstance(reply, tuple)
        return reply

    def backend(kind: str, name: str, method: str, args: tuple[object, ...], params: Mapping[str, object]) -> object:
        ok, payload = exchange(Call(kind, name, method, args, params))
        if ok:
            return payload
        raise RuntimeError(payload)

    def tell(running: Running) -> bool:
        _, stop = exchange(running)
        return stop is True

    _tell = tell
    distributed.install(backend)


class Bridge:
    """The worker's end: it answers every subprocess's collection calls.

    One thread waits on all the children's pipes at once and hands each request to a
    small pool, because a collection call runs on the worker's loop and blocks the
    thread that made it — one blocked answer must not hold up the next child's.

    A registration is read from the child that made it — the queue carries a handle,
    and the descriptor behind it is fetched from the process that owns it — so a
    child that died between registering and being read has nothing to hand over.
    That is a child with no calls to answer, not a reason for the thread that
    answers every other child to stop.

    It is also where the worker learns which child runs which attempt: ``pids`` is
    what a stop is addressed by, and a child starting an attempt the worker was
    already asked to stop hears so in the reply.
    """

    def __init__(self, registrations: Registrations) -> None:
        self.pids: dict[str, int] = {}
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
                except (OSError, EOFError):
                    continue
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
                match request:
                    case Running(id=id, pid=None):
                        self.pids.pop(id, None)
                        ready.send((True, False))
                    case Running(id=id, pid=int(pid)):
                        self.pids[id] = pid
                        ready.send((True, id in stopping.asked))
                    case Call():
                        self._dispatch.submit(self._answer, ready, request)

    def _answer(self, conn: Connection, call: Call) -> None:
        try:
            value = distributed.invoke(call.kind, call.name, call.method, call.args, call.params)
            conn.send((True, value))
        except Exception as exc:
            conn.send((False, f"{exc}\n{traceback.format_exc()}"))


class Pool:
    """The subprocess pool the tasks run on, and the means to build it again.

    A child that dies abruptly — killed for memory, crashed in a native extension —
    breaks the executor it belonged to for good: every task after it fails on
    arrival, in milliseconds, with the pool's own error and not the user's. A
    worker that kept such a pool would be a node that is up, answers its pings,
    and fails everything it is given, which is the worst thing a node can be. So
    the pool is disposable and the worker is not: the broken one is dropped, the
    next task gets a fresh one, and only the tasks that were in the dead child
    are lost.

    Loky's workers are kept while idle: its default retires them after ten seconds,
    and the next task would pay for a new process and every import again.
    """

    def __init__(self, kind: Kind, reuse: bool, workers: int, spawn: BaseContext, registrations: Registrations, pids: MutableMapping[str, int]) -> None:
        self._kind = kind
        self._pids = pids
        self._reuse = reuse
        self._workers = workers
        self._spawn = spawn
        self._registrations = registrations
        self._slots: Slots = spawn.Queue()
        self._executor = self._build()

    async def run[**P, T](self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run ``fn`` in a child, and let ``BrokenExecutor`` through to the caller.

        The caller sees the same exception the executor raised — that is its verdict
        on the task in flight — but by the time it does, the pool behind this object
        is already a new one. Two tasks that die together both raise and both ask
        for the rebuild; the second finds it done.
        """
        executor = self._executor
        try:
            return await asyncio.get_running_loop().run_in_executor(executor, partial(fn, *args, **kwargs))
        except BrokenExecutor:
            if executor is self._executor:
                executor.shutdown(wait=False)
                self._executor = self._build()
            raise

    def stop(self, id: str) -> bool:
        """Signal the child running ``id`` to stop it. ``False`` if no child is running it."""
        pid = self._pids.get(id)
        if pid is None:
            return False
        with suppress(ProcessLookupError):
            os.kill(pid, stopping.SIGNAL)
        return True

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def _build(self) -> Executor:
        for index in range(self._workers):
            self._slots.put(index)
        match self._kind:
            case "loky":
                from loky import get_reusable_executor

                return get_reusable_executor(
                    max_workers=self._workers,
                    initializer=_install,
                    initargs=(self._registrations, self._slots),
                    reuse="auto",
                    timeout=LOKY_IDLE_SECONDS,
                )
            case "process" if self._reuse:
                return ProcessPoolExecutor(
                    max_workers=self._workers,
                    mp_context=self._spawn,
                    initializer=_install,
                    initargs=(self._registrations, self._slots),
                )
            case _:
                return ProcessPoolExecutor(
                    max_workers=self._workers,
                    mp_context=self._spawn,
                    initializer=_install,
                    initargs=(self._registrations,),
                    max_tasks_per_child=1,
                )


@contextmanager
def pool(kind: Kind, reuse: bool, workers: int) -> Iterator[Pool]:
    """The pool the tasks run on, and the bridge behind it.

    ``workers`` is the pool's width — how many tasks run at once. It is the
    ``concurrency`` the pool was asked for; the buffer lives above it, in how many
    calls casty admits, not in how many the pool runs.
    """
    spawn = multiprocessing.get_context("spawn")
    registrations: Registrations = spawn.Queue()
    bridge = Bridge(registrations)
    bridge.start()
    subprocesses = Pool(kind, reuse, workers, spawn, registrations, bridge.pids)
    try:
        yield subprocesses
    finally:
        subprocesses.close()
        bridge.close()
