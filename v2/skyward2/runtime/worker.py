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
import os
import sys
import traceback
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial

import casty
import msgspec
from msgspec import Struct

from skyward2 import plugins
from skyward2.plugins import Plugin
from skyward2.protocol import codec
from skyward2.protocol.schemas import PluginRef
from skyward2.runtime.api import instance_info
from skyward2.runtime.journal import Journal, Phase, emit, task

PORT = 25520
SEED_TIMEOUT = 180.0


class Done(Struct, frozen=True, tag="done", tag_field="status"):
    value: bytes


class Failed(Struct, frozen=True, tag="failed", tag_field="status"):
    error: str
    traceback: str


class Pending(Struct, frozen=True, tag="pending", tag_field="status"):
    """The worker has the task and has not finished it."""


class Unknown(Struct, frozen=True, tag="unknown", tag_field="status"):
    """The worker has never heard of the task."""


type Outcome = Done | Failed
type Lookup = Done | Failed | Pending | Unknown
type Arguments = tuple[tuple[object, ...], dict[str, object]]

encode = msgspec.msgpack.encode
function: codec.Codec[Callable[..., object]] = codec.Pickle()
arguments: codec.Codec[Arguments] = codec.Pickle()
outcomes: dict[str, Outcome | None] = {}
installed: tuple[Plugin, ...] = ()
"""The compute's plugins, rebuilt on this machine from what the spec said they were."""


@casty.service(name="skyward.Worker", concurrency=int(os.environ.get("SKYWARD_SLOTS", "1")))
class Worker:
    """The tasks.

    ``concurrency`` is the number of slots, and the wait above it is not a
    detail: a call that finds every slot busy sits in the mailbox, and that is
    the backpressure the daemon is reading when it decides whether the compute
    needs another node.
    """

    async def run(self, id: str, code: bytes, args: bytes) -> bytes:
        outcomes[id] = None
        outcome = await execute(id, code, args)
        outcomes[id] = outcome
        return encode(outcome)


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

    token = task.set(id)
    try:
        fn = await function.decode(code)
        positional, keyword = await arguments.decode(args)
        wrapped = plugins.chain(installed, partial(call, fn, positional, keyword), instance_info())
        value = await asyncio.to_thread(wrapped)
        return Done(value=await codec.payload.encode(value))
    except Exception as exc:
        return Failed(error=str(exc), traceback=traceback.format_exc())
    finally:
        task.reset(token)


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
    global installed

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
        await asyncio.to_thread(setup, stack)

        emit(Phase(event="completed", phase="worker"))
        await asyncio.Event().wait()
    finally:
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
