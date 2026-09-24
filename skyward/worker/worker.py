"""The process that runs the user's functions, on the machine.

Started by the node once the bootstrap has left a venv behind, as ``python -m
skyward.worker``. It joins the compute's casty cluster and hosts two actors, pinned
to this machine: :func:`execution`, one key per attempt, and :func:`control`, one key
for the questions about the node.

Nothing the user wrote crosses casty's wire as an object. Payloads go in and come
out as opaque bytes, and only the two ends unpickle them — so the classes that
travel are the user's own, which exist on both sides by construction, and never
the worker's, which do not.

The actors are front desks. What they are asked to do takes as long as the user's
function does, and a key handles one message at a time — so every message is
answered at once or handed to a task of its own, and the body is back reading its
inbox before anything slow starts. That is also what keeps an ask the daemon gave up
on from reaching the work: casty cancels a body only while it is still on the
message, and no body here stays on one.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import sys
import traceback
from collections.abc import AsyncIterator, Callable, Coroutine, Generator, Iterator
from concurrent.futures import BrokenExecutor, ThreadPoolExecutor
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Literal, assert_never

import casty
import msgspec

from skyward.shared import codec, retry
from skyward.shared.frames import Chunk, Done, End, Failed, Lookup, Lost, Outcome, Pending, Step, Stopped, Unknown
from skyward.shared.observability import logger
from skyward.shared.schemas import Executor as ExecutorKind
from skyward.shared.schemas import PluginRef
from skyward.worker import distributed, ipc, plugins, stopping
from skyward.worker.api import Info, instance_info
from skyward.worker.journal import Health, Journal, Phase, emit, task
from skyward.worker.plugins import Plugin

PORT = 25520
SEED_TIMEOUT = 180.0
"""How long joining the compute's cluster may take before the worker gives up, and is started again by its supervisor.

casty keeps dialling the seeds until one answers, so the wait is for a machine that is still installing its
dependencies. Giving up ends the process, which says so in the journal, and the supervisor starts it again.
"""
MAX_MESSAGE_BYTES = 1024 * 1024 * 1024
"""The largest message or answer the compute's cluster carries; a task's arguments and its result each travel inside one."""
LIMITS = casty.Limits(message=MAX_MESSAGE_BYTES)
"""What crosses the compute's cluster. Every member and the daemon's client must agree on it, or the handshake refuses."""
COMPRESSION = casty.Compression(codecs=())
"""Compression is off: every payload is already an lz4 frame."""
THREADS = 2
"""The threads casty's transport runs on. The machine's cores are the user's; the wire needs few."""
TRANSFER = timedelta(hours=1)
"""How long an ask to an attempt may wait for its answer: the largest payload, over the slowest link a compute is reached by.

Not how long the function may run. Nothing waits on that in one ask: the attempt is handed over and answered for at
once, and its outcome is asked for again every :data:`HOLD` seconds. What this bounds is an answer that never comes
while the link stays up, which a link that drops does not need — casty fails those asks with ``Unavailable``.
"""
HOLD = 30.0
"""How long the worker holds a question about work that has not finished before answering that it has not."""
CONCURRENCY = int(os.environ.get("SKYWARD_SLOTS", "1"))
"""How many tasks the executor runs at once — the width of the pool."""
BUFFER = int(os.environ.get("SKYWARD_BUFFER", "0"))
"""How many more the worker admits, to keep the pool fed. See :func:`admit`."""
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


def material() -> casty.TLS | None:
    """What this machine shows the rest of the compute, and what it checks them against.

    The three files were written by the daemon over SSH, from the authority that
    belongs to this compute alone. Without them the port this process is about to
    open runs whatever anybody who can route to it sends — which is why the only
    compute that gets here empty-handed is one bound before there were certificates,
    whose daemon has no material to present either.
    """
    match os.environ.get("SKYWARD_TLS_CERT"), os.environ.get("SKYWARD_TLS_KEY"), os.environ.get("SKYWARD_TLS_CA"):
        case (str() as certificate, str() as key, str() as authority):
            return casty.TLS(cert=certificate, key=key, ca=authority, require_client_cert=True)
        case _:
            return None


type Arguments = tuple[tuple[object, ...], dict[str, object]]
type HealthCheck = Callable[[Info], bool | str]
type HealthChecks = tuple[AsyncIterator[tuple[bool, str | None]], int]

DONE = object()
"""``StopIteration`` does not survive a thread hop; this does."""

encode = msgspec.msgpack.encode
function: codec.Codec[Callable[..., object]] = codec.Pickle()
generator: codec.Codec[Callable[..., Iterator[object]]] = codec.Pickle()
arguments: codec.Codec[Arguments] = codec.Pickle()
outcomes: dict[str, asyncio.Future[Outcome]] = {}
"""Every attempt this worker was handed, by execution: its outcome, or the promise of one.

Kept until the daemon says it has recorded the outcome, or for :data:`KEEP_SECONDS`
after the attempt ended, whichever comes first."""
generators: dict[str, Iterator[object]] = {}
"""Streams in flight, by execution. Alive only as long as somebody is pulling on them."""
streams: dict[str, tuple[bytes, bytes]] = {}
"""Streams asked for and not opened yet, by execution: the generator's code and arguments, until the first pull opens it."""
steps: dict[str, asyncio.Task[Step]] = {}
"""The pull each stream's next item comes from, for as long as nobody has been handed that item."""
pulling: set[str] = set()
"""Streams whose next item is being pulled in a thread right now.

A close that lands mid-pull cannot close the generator — it is executing — so it
only lets go of it, and the pull closes it once it comes back."""
STOPPED = "the stream was stopped: it ran past its time"
"""What a stream's next pull says once :func:`stop` let go of it."""
running: set[asyncio.Task[object]] = set()
"""What the actors handed on and nobody awaits: the attempts, and the answers being held. Kept so none is collected half-way."""
KEEP_SECONDS = 3600.0
"""How long a settled outcome is kept for a daemon that has not said it recorded it.

The acknowledgement rides on the next attempt sent to this node, and a node may never
be sent another; without a bound, every payload it ever returned would stay here."""

installed: tuple[Plugin, ...] = ()
"""The compute's plugins, rebuilt on this machine from what the spec said they were."""

thread_pool: ThreadPoolExecutor
"""Where a generator always runs, and a task under ``thread``: pulling an item blocks,
and a subprocess cannot hold the far end of a stream the caller is pacing. Set by
:func:`main`."""
subprocesses: ipc.Pool | None = None
"""Where a task runs under ``process`` and ``loky``. Set by :func:`main`."""
admission: asyncio.Semaphore
"""How many attempts and pulls the worker takes on at once: ``concurrency + buffer``. Set by :func:`main`; see :func:`admit`."""


async def health(
    fn: HealthCheck,
    interval: float,
    timeout: float,
    initial_delay: float,
) -> AsyncIterator[tuple[bool, str | None]]:
    if initial_delay:
        await asyncio.sleep(initial_delay)
    loop = asyncio.get_running_loop()
    checks = ThreadPoolExecutor(max_workers=1, thread_name_prefix="skyward-health")
    running: asyncio.Future[bool | str] | None = None
    try:
        while True:
            if running is None:
                running = loop.run_in_executor(checks, fn, instance_info())
            try:
                async with asyncio.timeout(timeout):
                    result = await asyncio.shield(running)
            except TimeoutError:
                yield False, f"timeout after {timeout}s"
            except Exception as exc:
                running = None
                yield False, repr(exc)
            else:
                running = None
                match result:
                    case True:
                        yield True, None
                    case str() as reason if reason:
                        yield False, reason
                    case _:
                        yield False, None
            await asyncio.sleep(interval)
    finally:
        checks.shutdown(wait=False, cancel_futures=True)


async def warm(checks: AsyncIterator[tuple[bool, str | None]], consecutive_failures: int) -> None:
    failures = 0
    async for healthy, reason in checks:
        if healthy:
            return
        failures += 1
        if failures >= consecutive_failures:
            raise RuntimeError(f"health check failed {failures} times: {reason or 'unspecified'}")
    raise RuntimeError("health check ended before the node became ready")


async def unhealthy(checks: AsyncIterator[tuple[bool, str | None]], consecutive_failures: int) -> str:
    failures = 0
    async for healthy, reason in checks:
        if healthy:
            failures = 0
            continue
        failures += 1
        if failures >= consecutive_failures:
            return f"health check failed {failures} times: {reason or 'unspecified'}"
    raise RuntimeError("health check ended while the node was running")


def health_checks() -> HealthChecks | None:
    if not (path := os.environ.get("SKYWARD_HEALTH")):
        return None
    fn: HealthCheck = codec.loads(Path(path).read_bytes())
    return (
        health(
            fn,
            interval=float(os.environ["SKYWARD_HEALTH_INTERVAL"]),
            timeout=float(os.environ["SKYWARD_HEALTH_TIMEOUT"]),
            initial_delay=float(os.environ["SKYWARD_HEALTH_INITIAL_DELAY"]),
        ),
        int(os.environ["SKYWARD_HEALTH_FAILURES"]),
    )


async def start_health(configured: HealthChecks | None) -> asyncio.Task[str] | None:
    if configured is None:
        return None
    checks, consecutive_failures = configured
    await warm(checks, consecutive_failures)
    return asyncio.create_task(unhealthy(checks, consecutive_failures))


def bind_distributed(system: casty.ActorSystem) -> None:
    if os.environ.get("SKYWARD_CLUSTER", "1") == "1":
        distributed.bind(system, asyncio.get_running_loop())


@dataclass(frozen=True, slots=True)
class Run:
    """Take on an attempt, and answer once it is held: the outcome is asked for with :class:`Result`.

    ``decision`` is the task's retry decision, pickled, and ``attempt`` which attempt this is — asked together if the
    function raises. ``settled`` are executions whose outcome the daemon has recorded, and which this worker therefore
    no longer needs to keep.
    """

    reply_to: casty.Ref[None]
    code: bytes
    args: bytes
    decision: bytes = b""
    attempt: int = 1
    settled: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Result:
    """The attempt's outcome, as an encoded :data:`~skyward.shared.frames.Lookup`. See :func:`answer`."""

    reply_to: casty.Ref[bytes]


@dataclass(frozen=True, slots=True)
class Stop:
    """Stop an attempt that ran past its time, and say whether anything here was running it. See :func:`stop`."""

    reply_to: casty.Ref[bool]


@dataclass(frozen=True, slots=True)
class Open:
    """Take on a stream: the generator's code and arguments, sent once and not with every item."""

    reply_to: casty.Ref[None]
    code: bytes
    args: bytes


@dataclass(frozen=True, slots=True)
class Next:
    """A stream's next item, as an encoded :data:`~skyward.shared.frames.Step`. See :func:`pull`."""

    reply_to: casty.Ref[bytes]


@dataclass(frozen=True, slots=True)
class Close:
    """Let go of a stream whose reader went away. See :func:`close`."""

    reply_to: casty.Ref[None]


type ExecutionMessage = Run | Result | Stop | Open | Next | Close


@dataclass(frozen=True, slots=True)
class Ping:
    """Which node this is: the answer comes from the loop, so it says the loop is not held."""

    reply_to: casty.Ref[str]


@dataclass(frozen=True, slots=True)
class Topology:
    """Where the other nodes are, when that changes under a running worker. See :func:`control`."""

    reply_to: casty.Ref[None]
    peers: tuple[str, ...]


type ControlMessage = Ping | Topology

CONTROL = "control"
"""The key of :func:`control` on every node."""


@casty.actor(pinned=True, ask_timeout=TRANSFER)
async def execution(ctx: casty.Context[None, ExecutionMessage]) -> None:
    """One execution on this node, keyed by its id.

    A pinned key reads ``@host:port/<id>``, so the id is what follows the slash. Everything the key knows of the
    execution lives in this module and not in the key's state: an attempt belongs to this process, and a worker that
    restarts has none of them, which is what :class:`~skyward.shared.frames.Unknown` tells the daemon.
    """
    id = ctx.key.partition("/")[2]
    async for message in ctx.inbox:
        match message:
            case Run(reply_to, code, args, decision, attempt, settled):
                admit(id, code, args, decision, attempt, settled)
                reply_to.tell(None)
            case Result(reply_to):
                _detach(answer(reply_to, id))
            case Stop(reply_to):
                _detach(_told(reply_to, stop(id)))
            case Open(reply_to, code, args):
                streams[id] = (code, args)
                reply_to.tell(None)
            case Next(reply_to):
                _detach(pull(reply_to, id))
            case Close(reply_to):
                _detach(_told(reply_to, close(id)))
            case _:
                assert_never(message)


@casty.actor(pinned=True)
async def control(ctx: casty.Context[None, ControlMessage]) -> None:
    """The questions about the node, on a key that cannot queue behind the work.

    Where the other nodes are is the one fact a worker is told rather than reads. A worker is started with the world as
    it was, and a compute that grows or drains changes it afterwards, so :class:`Topology` pushes the new one here and
    it lands where :func:`skyward.instance_info` looks for it. Without it a node that was here before the resize goes
    on sharding data into the number of ways there used to be, which is not an error anywhere: it is rows processed
    twice and rows processed never.
    """
    async for message in ctx.inbox:
        match message:
            case Ping(reply_to):
                reply_to.tell(os.environ["SKYWARD_NODE"])
            case Topology(reply_to, peers):
                os.environ["SKYWARD_PEERS"] = ",".join(peers)
                reply_to.tell(None)
            case _:
                assert_never(message)


def admit(id: str, code: bytes, args: bytes, decision: bytes = b"", attempt: int = 1, settled: tuple[str, ...] = ()) -> None:
    """Take on one attempt: from here on its outcome is owed, to whoever asks :func:`answer` for it.

    Taking on an attempt twice is taking it on once. A daemon that lost the answer to :class:`Run` cannot tell whether
    it arrived, and the attempt it sends again is the one already running.

    The worker admits ``concurrency + buffer`` attempts at once; the executor runs ``concurrency`` of them. The gap is
    the buffer: those attempts wait at the executor's door, so a slot that frees finds the next task in hand. An
    attempt waiting for admission is already in :data:`outcomes`, so a daemon asking about it hears that it is not done
    rather than that the worker never had it — and does not run it a second time on the strength of that.

    An attempt that ends without an outcome — cancelled under it, or ``execute`` itself broken — is settled as lost, so
    whoever waits on it hears the loss instead of waiting out its deadline.
    """
    for recorded in settled:
        outcomes.pop(recorded, None)
    if id in outcomes:
        return

    loop = asyncio.get_running_loop()
    promise: asyncio.Future[Outcome] = loop.create_future()
    outcomes[id] = promise
    promise.add_done_callback(lambda _: loop.call_later(KEEP_SECONDS, _forget, id, promise))

    async def attempted() -> None:
        try:
            async with admission:
                outcome = await execute(id, code, args, decision, attempt)
        except BaseException as exc:
            promise.set_result(Lost(error=f"the attempt ended without an outcome: {exc!r}"))
            raise
        promise.set_result(outcome)

    _detach(attempted())


async def answer(reply_to: casty.Ref[bytes], id: str) -> None:
    """An attempt's outcome, once there is one — or, held for :data:`HOLD` seconds without one, that there is none yet.

    Held so the daemon does not ask again and again whether the function is done, and let go of before the ask's own
    deadline so an answer never reaches an ask that stopped waiting. An attempt this worker has never heard of is
    answered at once: the worker restarted under it, and nothing here will ever finish it. The wait is shielded,
    because giving up on it must not cancel the outcome the attempt is about to set.
    """
    match outcomes.get(id):
        case None:
            reply_to.tell(encode(Unknown()))
        case promise:
            try:
                async with asyncio.timeout(HOLD):
                    outcome = await asyncio.shield(promise)
            except TimeoutError:
                reply_to.tell(encode(Pending()))
            else:
                reply_to.tell(await _encoded(outcome))


async def pull(reply_to: casty.Ref[bytes], id: str) -> None:
    """One item, because somebody asked for one — or, held for :data:`HOLD` seconds without one, that it is not here yet.

    The stream is a pull and not a push: the daemon asks once per item it has somewhere to put, and the request reading
    it is what asks. So a consumer that stops consuming stops the generator, and the node is never asked to hold what
    nobody has asked for yet.

    A question that is let go of leaves its pull running, and the next one takes the item from it: a generator is
    pulled once per item, however many times the item was asked for. The first pull opens the stream, so a generator
    that cannot be built fails the way one that raises does.
    """
    step = steps.get(id)
    if step is None:
        step = steps[id] = _detach(_step(id))
    try:
        async with asyncio.timeout(HOLD):
            frame = await asyncio.shield(step)
    except TimeoutError:
        reply_to.tell(encode(Pending()))
        return
    if steps.get(id) is step:
        del steps[id]
    reply_to.tell(await _encoded(frame))


async def close(id: str) -> None:
    """Let go of a stream whose reader went away.

    There is nothing to keep. A stream cannot be resumed — the items already sent are gone from here — so a reader that
    has stopped reading is a reader that will not be back for this one.

    The generator is closed in the thread pool, because closing it runs its ``finally`` — the user's code, which may
    reach for a collection that blocks on this very loop. One being pulled right now is left to the pull to close.
    """
    streams.pop(id, None)
    steps.pop(id, None)
    iterator = generators.pop(id, None)
    if iterator is not None and id not in pulling:
        await _finish(iterator)


async def stop(id: str) -> bool:
    """Stop an attempt that ran past its time, and say whether anything here was running it.

    The attempt's own :class:`Result` is what hears ``Stopped``, once the function has unwound; this only says the
    request landed. An attempt that has not started yet is remembered, and never starts. A stream is let go of the way
    one whose reader left is, and its next pull says it was stopped.
    """
    stopping.asked.add(id)
    asyncio.get_running_loop().call_later(KEEP_SECONDS, stopping.asked.discard, id)
    if streams.pop(id, None) is not None:
        return True
    if (iterator := generators.pop(id, None)) is not None:
        if id not in pulling:
            await _finish(iterator)
        return True
    match MODE:
        case "thread":
            return stopping.interrupt(id)
        case "process" | "loky":
            return subprocesses is not None and subprocesses.stop(id)


async def execute(id: str, code: bytes, args: bytes, decision: bytes = b"", attempt: int = 1) -> Outcome:
    """Run one task, off the event loop, start to finish.

    The code and the arguments arrive as two blobs and are unpickled here, on the
    machine that has the user's libraries. The daemon never opens either: it has
    no torch, no pandas and no copy of the module the function came from, and a
    control plane that had to import the user's world in order to dispatch to it
    would be a control plane that dies of somebody else's dependency.

    Nothing here touches the loop, either. The codec threads both directions and
    the function gets a thread of its own — this loop is the one the actors
    answer on, and a node holding it is a node that reads as dead. The plugins wrap
    the call inside that thread with it: a plugin that grabs a lock or sets a thread
    local is talking about the thread the user's code runs on, and would be talking
    about the event loop's if it were wrapped anywhere else.

    A failure to unpickle is therefore a failed task, with a traceback, rather
    than an exception thrown inside an actor. It is also the most common
    failure there is: it is what a version of pandas that differs between the two
    ends looks like from here.

    An attempt :func:`stop` reached unwinds with :class:`stopping.Stop`, in
    the thread or the subprocess running it, and is ``Stopped`` — not ``Lost``, whose
    retry decision would run it again.
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
            value = await loop.run_in_executor(thread_pool, contextvars.copy_context().run, partial(_stoppable, id, wrapped))
            return Done(value=await codec.payload.encode(value))

        assert subprocesses is not None
        try:
            ok, payload = await subprocesses.run(_run_in_process, id, code, args, os.environ["SKYWARD_PEERS"], decision, attempt)
        except BrokenExecutor as exc:
            return Lost(error=f"the process running the task died: {exc}")
        if ok:
            assert isinstance(payload, bytes)
            return Done(value=payload)
        assert isinstance(payload, tuple)
        error, trace, again = payload
        return Failed(error=error, traceback=trace, retry=again)
    except stopping.Stop:
        return Stopped()
    except Exception as exc:
        trace = traceback.format_exc()
        error = str(exc)
        return Failed(error=error, traceback=trace, retry=await asyncio.to_thread(_again, decision, exc, attempt))
    finally:
        task.reset(token)


def _stoppable[T](id: str, call: Callable[[], T]) -> T:
    """Run an attempt in this thread where :func:`stop` can interrupt it."""
    with stopping.running(id):
        return call()


def _again(decision: bytes, exc: Exception, attempt: int) -> bool:
    """The user's retry decision, asked here because the exception is alive here.

    The daemon never unpickles what a function raised — it has none of the
    libraries that raised it — so the question is put on the worker and the answer
    travels back as one bit of the ``Failed`` frame. A decision that cannot be
    loaded is a no, said in the log.
    """
    if not decision:
        return False
    try:
        fn: retry.Retry = codec.loads(decision)
    except Exception:
        logger.warning("the task's retry decision could not be loaded; not retrying", exc_info=True)
        return False
    return retry.decide(fn, exc, attempt)


async def advance(id: str) -> Step:
    """Pull one item, off the event loop, and say what came of it.

    The generator's body runs here and nowhere else, so every item costs a thread.
    That is the right price: a step that blocks is the normal case — a stream is
    usually a file being read or a model emitting tokens — and a step that blocked
    the loop would stop the node answering for as long as it took.
    """
    loop = asyncio.get_running_loop()
    token = task.set(id)
    try:
        if id not in generators and id in stopping.asked:
            return Failed(error=STOPPED, traceback="")
        iterator = generators[id]

        def pull() -> object:
            try:
                return next(iterator)
            except StopIteration:
                return DONE
            finally:
                sys.stdout.flush()
                sys.stderr.flush()

        wrapped = partial(contextvars.copy_context().run, pull)
        pulling.add(id)
        try:
            item = await loop.run_in_executor(thread_pool, wrapped)
        finally:
            pulling.discard(id)
        if id not in generators:
            await _finish(iterator)
            return Failed(error=STOPPED, traceback="") if id in stopping.asked else End()
        if item is DONE:
            generators.pop(id, None)
            return End()
        return Chunk(value=await codec.payload.encode(item))
    except Exception as exc:
        generators.pop(id, None)
        return Failed(error=str(exc), traceback=traceback.format_exc())
    finally:
        task.reset(token)


async def _finish(iterator: Iterator[object]) -> None:
    """Close a stream nobody will pull again, in the thread pool, and never raise.

    Closing a generator runs its ``finally``, which is the user's code: on the loop,
    a ``sky.lock`` there would wait for a loop that is busy waiting for it. What the
    cleanup raises has no caller left to hear it, so it is said in the log.
    """
    def shut() -> None:
        try:
            match iterator:
                case Generator() as running:
                    running.close()
        except Exception:
            logger.warning("the stream's cleanup raised; its consumer had already left", exc_info=True)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

    await asyncio.get_running_loop().run_in_executor(thread_pool, contextvars.copy_context().run, shut)


async def _step(id: str) -> Step:
    """Pull a stream's next item, opening it first if this is its first pull."""
    async with admission:
        if (opening := streams.pop(id, None)) is not None:
            try:
                await _open(id, *opening)
            except Exception as exc:
                return Failed(error=str(exc), traceback=traceback.format_exc())
        return await advance(id)


async def _open(id: str, code: bytes, args: bytes) -> None:
    """Build the generator. Nothing of the user's code has run yet.

    Calling a generator function runs none of its body, which is what makes the
    opening separable from the pulling — and the pulling is what the reader paces.
    The call is made in the thread pool all the same, inside the plugins' ``run``,
    which is the user's code and may block.
    """
    fn = await generator.decode(code)
    positional, keyword = await arguments.decode(args)
    wrapped = plugins.chain(installed, partial(fn, *positional, **keyword), instance_info())
    generators[id] = await asyncio.get_running_loop().run_in_executor(thread_pool, contextvars.copy_context().run, partial(_flushing, wrapped))


def _detach[T](work: Coroutine[object, object, T]) -> asyncio.Task[T]:
    """Run ``work`` on its own, kept in :data:`running` until it ends; what it raises is said in the log."""
    handed = asyncio.get_running_loop().create_task(work)
    running.add(handed)
    handed.add_done_callback(_ended)
    return handed


def _ended(handed: asyncio.Task[object]) -> None:
    running.discard(handed)
    if not handed.cancelled() and (error := handed.exception()) is not None:
        logger.error("work the worker handed on raised: {}", error, exc_info=error)


async def _told[T](reply_to: casty.Ref[T], work: Coroutine[object, object, T]) -> None:
    reply_to.tell(await work)


def _flushing[T](call: Callable[[], T]) -> T:
    try:
        return call()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def _forget(id: str, promise: asyncio.Future[Outcome]) -> None:
    """Drop an outcome kept past its time, unless a later attempt already took its place."""
    if outcomes.get(id) is promise:
        outcomes.pop(id)


async def _encoded(frame: Lookup | Step) -> bytes:
    """Encode a frame, in a thread when its payload is big enough to hold the loop."""
    match frame:
        case Done(value=value) | Chunk(value=value) if len(value) >= codec.THRESHOLD:
            return await asyncio.to_thread(encode, frame)
        case _:
            return encode(frame)


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


def _run_in_process(
    id: str, code: bytes, args: bytes, peers: str, decision: bytes = b"", attempt: int = 1
) -> tuple[Literal[True], bytes] | tuple[Literal[False], tuple[str, str, bool]]:
    """Run one task in a subprocess, and bring back an answer that survives the trip.

    The code and the arguments are unpickled here, where the user's libraries are,
    and the collections the task reaches for go home over the IPC bridge the pool's
    initializer left behind. The exception does not survive the trip: a user's error
    class may not exist here in a form the worker can unpickle, so a failure comes back
    as its message and traceback, already text, and the worker turns those into Failed.

    The peers travel with the task rather than being read off this process's own
    environment: a reused child was forked with the world as it was when the pool
    was built, and a compute that has grown since would be sharded here into the
    number of ways there used to be.

    Output is redirected here for the same reason it is redirected in ``cli``: a
    fresh interpreter writes to the fd it inherited, which is the worker's log file
    and not the journal the daemon is tailing — the one place bootstrap output and
    thread-mode output already stream from.
    """
    os.environ["SKYWARD_PEERS"] = peers
    if not isinstance(sys.stdout, Journal):
        sys.stdout = Journal("stdout")
        sys.stderr = Journal("stderr")
    token = task.set(id)
    try:
        with ipc.attempt(id):
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
        return False, (str(exc), traceback.format_exc(), _again(decision, exc, attempt))
    finally:
        task.reset(token)


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

    An empty list of seeds is the node that opens the cluster, and it waits for nobody.
    Any other waits for one of them to answer, which may be a machine that is still
    installing its dependencies: the list is every node that has an address, not every
    node that is up.
    """
    global installed, thread_pool, subprocesses, admission

    refs = msgspec.json.decode(os.environ.get("SKYWARD_PLUGINS", "[]"), type=tuple[PluginRef, ...])
    installed = plugins.resolve(refs)
    admission = asyncio.Semaphore(CONCURRENCY + BUFFER)

    cluster = casty.Cluster(
        bind=f"0.0.0.0:{PORT}",
        advertise=f"{os.environ['SKYWARD_PEER']}:{PORT}",
        seeds=tuple(seed for seed in os.environ.get("SKYWARD_SEEDS", "").split(",") if seed),
        name=os.environ["SKYWARD_COMPUTE"],
        tls=material(),
        compression=COMPRESSION,
        limits=LIMITS,
    )
    async with AsyncExitStack() as joined:
        async with asyncio.timeout(SEED_TIMEOUT):
            system = await joined.enter_async_context(casty.ActorSystem(cluster=cluster, runtime=casty.Runtime(threads=THREADS)))
        stack = ExitStack()
        try:
            bind_distributed(system)
            thread_pool = stack.enter_context(ThreadPoolExecutor(max_workers=CONCURRENCY))
            match MODE:
                case "process" | "loky" as kind:
                    subprocesses = stack.enter_context(ipc.pool(kind, REUSE, CONCURRENCY))
                case "thread":
                    pass
            await asyncio.to_thread(setup, stack)

            health_monitor = await start_health(health_checks())
            emit(Phase(event="completed", phase="worker"))
            if health_monitor is None:
                await asyncio.Event().wait()
            else:
                emit(Health(reason=await health_monitor))
        finally:
            distributed.unbind()
            await asyncio.to_thread(stack.close)


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

