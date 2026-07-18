"""The process that runs the user's functions, on the machine.

Started by the node once the bootstrap has left a venv behind. It joins the
compute's casty cluster and serves two things: the tasks, and the questions about
the tasks.

Nothing the user wrote crosses casty's wire as an object. Payloads go in and come
out as opaque bytes, and only the two ends unpickle them — so the classes that
travel are the user's own, which exist on both sides by construction, and never
the worker's, which do not.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import sys
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from typing import Literal

import casty
import msgspec

from skyward import distributed, plugins
from skyward.plugins import Plugin
from skyward.protocol import codec
from skyward.protocol.frames import Chunk, Done, End, Failed, Lookup, Outcome, Pending, Step, Unknown
from skyward.protocol.schemas import Executor as ExecutorKind
from skyward.protocol.schemas import PluginRef
from skyward.runtime import ipc
from skyward.runtime.api import instance_info
from skyward.runtime.journal import Journal, Phase, emit, task

PORT = 25520
SEED_TIMEOUT = 180.0
CONCURRENCY = int(os.environ.get("SKYWARD_SLOTS", "1"))
"""How many tasks the executor runs at once — the width of the pool."""
BUFFER = int(os.environ.get("SKYWARD_BUFFER", "0"))
"""How many more casty admits, to keep the pool fed. See :class:`Worker`."""
REUSE = os.environ.get("SKYWARD_REUSE", "1") == "1"


def _mode() -> ExecutorKind:
    match os.environ.get("SKYWARD_EXECUTOR", "thread"):
        case "process":
            return "process"
        case "loky":
            return "loky"
        case _:
            return "thread"


MODE = _mode()


type Arguments = tuple[tuple[object, ...], dict[str, object]]

DONE = object()
"""``StopIteration`` does not survive a thread hop; this does."""

encode = msgspec.msgpack.encode
function: codec.Codec[Callable[..., object]] = codec.Pickle()
generator: codec.Codec[Callable[..., Iterator[object]]] = codec.Pickle()
arguments: codec.Codec[Arguments] = codec.Pickle()
outcomes: dict[str, Outcome | None] = {}
generators: dict[str, Iterator[object]] = {}
"""Streams in flight, by execution. Alive only as long as somebody is pulling on them."""

installed: tuple[Plugin, ...] = ()
"""The compute's plugins, rebuilt on this machine from what the spec said they were."""

task_pool: Executor
"""Where a task runs: the thread pool, or a subprocess pool. Set by :func:`main`."""
thread_pool: Executor
"""Where a generator always runs, whatever the task pool is: pulling an item blocks,
and a subprocess cannot hold the far end of a stream the caller is pacing. It is the
task pool itself under ``thread``, a pool of its own otherwise."""


@casty.service(name="skyward.Worker", concurrency=CONCURRENCY + BUFFER)
class Worker:
    """The tasks.

    The service admits ``concurrency + buffer`` calls; the executor runs
    ``concurrency`` of them. The gap is the buffer: those calls arrive and their
    payloads are unpickled, then wait at the executor's door, so a slot that frees
    finds the next task in hand. A call that finds even the buffer full sits in the
    mailbox, and that is the backpressure the daemon reads when it decides whether
    the compute needs another node.
    """

    async def run(self, id: str, code: bytes, args: bytes) -> bytes:
        outcomes[id] = None
        outcome = await execute(id, code, args)
        outcomes[id] = outcome
        return encode(outcome)

    async def open(self, id: str, code: bytes, args: bytes) -> None:
        """Build the generator. Nothing of the user's code has run yet.

        Calling a generator function runs none of its body, which is what makes the
        opening separable from the pulling — and the pulling is what the caller
        paces.
        """
        fn = await generator.decode(code)
        positional, keyword = await arguments.decode(args)
        generators[id] = plugins.chain(installed, partial(fn, *positional, **keyword), instance_info())()

    async def step(self, id: str) -> bytes:
        """One item, because somebody asked for one.

        The stream is a pull and not a push: the daemon calls this once per item it
        has somewhere to put, and the request reading it is what asks. So a consumer
        that stops consuming stops the generator, and the node is never asked to hold
        what nobody has asked for yet.

        The code and the arguments are not sent again. They were sent once, to
        ``open`` — a generator that yields a million items would otherwise ship the
        user's function a million times.
        """
        return encode(await advance(id))

    async def close(self, id: str) -> None:
        """Let go of a generator whose caller went away.

        There is nothing to keep. A stream cannot be resumed — the items already sent
        are gone from here — so a caller that has stopped reading is a caller that
        will not be back for this one.
        """
        generators.pop(id, None)


@casty.service(name="skyward.Control")
class Control:
    """The questions, on a channel that cannot queue behind the answers.

    If these lived on :class:`Worker` they would wait for a free slot, and a node
    with every slot busy would be indistinguishable from a node that had died —
    the health of the work arriving only after the work it was supposed to be
    watching over.
    """

    async def ping(self) -> str:
        return os.environ["SKYWARD_NODE"]

    async def result(self, id: str) -> bytes:
        """What the daemon asks after coming back up and finding a task in flight."""
        lookup: Lookup
        match outcomes.get(id, Unknown()):
            case Unknown():
                lookup = Unknown()
            case None:
                lookup = Pending()
            case outcome:
                lookup = outcome
        return encode(lookup)


async def execute(id: str, code: bytes, args: bytes) -> Outcome:
    """Run one task, off the event loop, start to finish.

    The code and the arguments arrive as two blobs and are unpickled here, on the
    machine that has the user's libraries. The daemon never opens either: it has
    no torch, no pandas and no copy of the module the function came from, and a
    control plane that had to import the user's world in order to dispatch to it
    would be a control plane that dies of somebody else's dependency.

    Nothing here touches the loop, either. The codec threads both directions and
    the function gets a thread of its own — this loop is the one `Control.ping`
    answers on, and a node holding it is a node that reads as dead. The plugins wrap
    the call inside that thread with it: a plugin that grabs a lock or sets a thread
    local is talking about the thread the user's code runs on, and would be talking
    about the event loop's if it were wrapped anywhere else.

    A failure to unpickle is therefore a failed task, with a traceback, rather
    than an exception thrown inside a casty handler. It is also the most common
    failure there is: it is what a version of pandas that differs between the two
    ends looks like from here.
    """
    def call(fn: Callable[..., object], positional: tuple[object, ...], keyword: dict[str, object]) -> object:
        """Flush in the thread that wrote, and while the task's output policy still holds.

        Both are context: the journal reads the policy the user's decorator set, and
        that lives in the thread's copy of the context, not in the loop's. A trailing
        line flushed anywhere else is a line flushed under the wrong policy, by a task
        that is no longer the current one.
        """
        try:
            return fn(*positional, **keyword)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

    loop = asyncio.get_running_loop()
    token = task.set(id)
    try:
        if MODE == "thread":
            fn = await function.decode(code)
            positional, keyword = await arguments.decode(args)
            wrapped = plugins.chain(installed, partial(call, fn, positional, keyword), instance_info())
            value = await loop.run_in_executor(thread_pool, contextvars.copy_context().run, wrapped)
            return Done(value=await codec.payload.encode(value))

        ok, payload = await loop.run_in_executor(task_pool, _run_in_process, code, args)
        if ok:
            assert isinstance(payload, bytes)
            return Done(value=payload)
        assert isinstance(payload, tuple)
        error, trace = payload
        return Failed(error=error, traceback=trace)
    except Exception as exc:
        return Failed(error=str(exc), traceback=traceback.format_exc())
    finally:
        task.reset(token)


async def advance(id: str) -> Step:
    """Pull one item, off the event loop, and say what came of it.

    The generator's body runs here and nowhere else, so every item costs a thread.
    That is the right price: a step that blocks is the normal case — a stream is
    usually a file being read or a model emitting tokens — and a step that blocked
    the loop would stop the node answering for as long as it took.
    """
    def pull(iterator: Iterator[object]) -> object:
        try:
            return next(iterator)
        except StopIteration:
            return DONE
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

    loop = asyncio.get_running_loop()
    token = task.set(id)
    try:
        item = await loop.run_in_executor(thread_pool, contextvars.copy_context().run, pull, generators[id])
        if item is DONE:
            generators.pop(id, None)
            return End()
        return Chunk(value=await codec.payload.encode(item))
    except Exception as exc:
        generators.pop(id, None)
        return Failed(error=str(exc), traceback=traceback.format_exc())
    finally:
        task.reset(token)


child_plugins: tuple[Plugin, ...] | None = None
"""The plugins, once a subprocess has rebuilt them. See :func:`_installed`."""


def _installed() -> tuple[Plugin, ...]:
    """The plugins for this subprocess, rebuilt once from what the spec carried.

    A subprocess is a fresh interpreter, so the worker's ``installed`` is empty here.
    The refs ride in the environment the pool inherited, and resolve to the same values
    the worker holds — the plugins decorate the call the same, wherever it runs.
    """
    global child_plugins
    if child_plugins is None:
        refs = msgspec.json.decode(os.environ.get("SKYWARD_PLUGINS", "[]"), type=tuple[PluginRef, ...])
        child_plugins = plugins.resolve(refs)
    return child_plugins


def _run_in_process(code: bytes, args: bytes) -> tuple[Literal[True], bytes] | tuple[Literal[False], tuple[str, str]]:
    """Run one task in a subprocess, and bring back an answer that survives the trip.

    The code and the arguments are unpickled here, where the user's libraries are,
    and the collections the task reaches for go home over the IPC bridge the pool's
    initializer left behind. The exception does not survive the trip: a user's error
    class may not exist here in a form the worker can unpickle, so a failure comes back
    as its message and traceback, already text, and the worker turns those into Failed.
    """
    try:
        fn: Callable[..., object] = codec.loads(code)
        decoded: Arguments = codec.loads(args)
        positional, keyword = decoded
        wrapped = plugins.chain(_installed(), partial(fn, *positional, **keyword), instance_info())
        try:
            value = wrapped()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
        return True, codec.dumps(value)
    except Exception as exc:
        return False, (str(exc), traceback.format_exc())


async def reachable(seed: str) -> None:
    """Wait for the seed to answer on its port before trying to join it.

    ``casty.start`` binds its own port before dialling the seed, and does not give
    it back when the join fails — so a worker that merely races the seed does not
    retry, it dies holding the port. A TCP connect is the cheap way to not race.
    """
    host, port = seed.rsplit(":", 1)
    async with asyncio.timeout(SEED_TIMEOUT):
        while True:
            try:
                _, writer = await asyncio.open_connection(host, int(port))
                writer.close()
                return
            except OSError:
                await asyncio.sleep(2.0)


async def main() -> None:
    """Join the cluster, let the plugins have the process, and wait for work.

    The plugins are set up after the cluster and before readiness is announced,
    which is the only window that works for the collective ones: ``init_process_group``
    blocks until every rank arrives, so a node that announced itself ready first
    would be handed a task it cannot start.

    And it blocks — that is what a collective does — so it is entered off the loop.
    A worker that sat in the loop waiting for the other ranks would stop answering
    casty's heartbeats while doing it, and be evicted from the cluster it was in the
    middle of joining.
    """
    global installed, task_pool, thread_pool

    seeds = [seed for seed in os.environ.get("SKYWARD_SEEDS", "").split(",") if seed]
    for seed in seeds:
        await reachable(seed)

    refs = msgspec.json.decode(os.environ.get("SKYWARD_PLUGINS", "[]"), type=tuple[PluginRef, ...])
    installed = plugins.resolve(refs)

    system = await casty.start(
        f"0.0.0.0:{PORT}",
        advertise=f"{os.environ['SKYWARD_PEER']}:{PORT}",
        seeds=seeds,
        cluster_name=os.environ["SKYWARD_COMPUTE"],
    )
    stack = ExitStack()
    try:
        distributed.bind(system, asyncio.get_running_loop())
        task_pool = stack.enter_context(ipc.executor(MODE, REUSE, CONCURRENCY))
        thread_pool = task_pool if MODE == "thread" else stack.enter_context(ThreadPoolExecutor(max_workers=CONCURRENCY))
        await asyncio.to_thread(setup, stack)

        emit(Phase(event="completed", phase="worker"))
        await asyncio.Event().wait()
    finally:
        distributed.unbind()
        await asyncio.to_thread(stack.close)
        await system.close()


def setup(stack: ExitStack) -> None:
    info = instance_info()
    for plugin in installed:
        stack.enter_context(plugin.setup(info))


def cli() -> None:
    """Run the worker, with its output going where the node is already looking.

    Readiness is announced from here, and only once the cluster has been joined.
    A tunnel that accepts TCP proves the machine's sshd is alive; it says nothing
    about a worker that died importing torch. What the node waits for is the node
    itself saying it got this far.
    """
    sys.stdout = Journal("stdout")
    sys.stderr = Journal("stderr")

    try:
        asyncio.run(main())
    except Exception as exc:
        emit(Phase(event="failed", phase="worker", error=str(exc)))
        raise


if __name__ == "__main__":
    cli()
