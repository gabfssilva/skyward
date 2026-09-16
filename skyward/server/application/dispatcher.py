"""Getting a written-down attempt onto a machine, and its answer back.

Everything here is about one execution. Nothing here decides how many machines
there should be — it takes the ones that are ready and places work on them, and if
there are none it does nothing at all and waits to be woken again. That is not a
failure mode, it is the whole protocol: the reconciler is watching the same queue
and will grow the pool, and the task will be offered again when it does.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from casty.errors import ActorUnavailableError, ConnectionLostError

from skyward.server.application.connector import HELD
from skyward.server.application.runtimes import Runtime, Runtimes
from skyward.server.application.ssh import SshUnavailableError
from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.tasks import PENDING, TaskStore
from skyward.shared import codec, retry
from skyward.shared.errors import ComputeNotAcceptingError
from skyward.shared.events import TaskEvent
from skyward.shared.frames import Chunk, Done, End, Failed, Lookup, Lost, Outcome, Step, Unknown
from skyward.shared.observability import logger
from skyward.shared.schemas import Error, Execution, ExecutionState, Task
from skyward.worker import worker

logger = logger.bind(component="dispatcher")

type Wake = Callable[..., None]

_STEPS: codec.Msgpack[Step] = codec.Msgpack(Step)
_OUTCOMES: codec.Msgpack[Outcome] = codec.Msgpack(Outcome)
_LOOKUPS: codec.Msgpack[Lookup] = codec.Msgpack(Lookup)
"""What a worker answers with wraps the user's payload, so it is decoded through
the codec — off the loop past the threshold — rather than inline."""
_DECISIONS: codec.Pickle[retry.Retry] = codec.Pickle()
"""The user's retry decision, the one piece of theirs the daemon unpickles. It is
asked about a :class:`retry.Lost` — a type of skyward's — and a decision that needs
the user's libraries to load is one that cannot be asked here, and is a no."""

IN_FLIGHT: tuple[ExecutionState, ...] = ("dispatching", "accepted", "started")
"""Sent, and not yet answered for.

An attempt in one of these was handed to a machine by somebody — possibly by a
daemon that no longer exists. Finding one this process did not dispatch is not an
error; it is the normal thing to find after a restart, and it is why the worker keeps
its outcomes by execution id.
"""

LINK_ERRORS = (ConnectionLostError, ActorUnavailableError)
"""The call died on the wire, and says nothing about the function.

The worker runs the function as its own task and keeps the outcome under the
execution's id, so a reply lost to a dropped link is a reply that can be waited
for again — see :meth:`Dispatcher._await`. Anything else that escapes the call
is the honest ``indeterminate``.
"""

RELINK = 2.0
"""Seconds between a call dying on the wire and waiting on the link to try again.

Two reasons not to go straight back. A channel notices its connection died a moment
after the calls riding it do, and a link waited on in that moment is found up and
dialled dead. And a call can die with the link up — a worker that does not answer
behind a healthy channel — where going straight back is asking forever, as fast as
the loop turns.
"""


class Dispatcher:
    def __init__(
        self,
        computes: ComputeStore,
        tasks: TaskStore,
        nodes: NodeStore,
        blobs: BlobStore,
        events: EventStore,
        runtimes: Runtimes,
        wake: Wake,
    ) -> None:
        self._computes = computes
        self._tasks = tasks
        self._nodes = nodes
        self._blobs = blobs
        self._events = events
        self._runtimes = runtimes
        self._wake = wake
        self._locks: dict[str, asyncio.Lock] = {}

    async def resume(self, compute_id: str) -> None:
        """Offer the queue whatever is free right now, and stop when nothing is.

        The slots are counted once and spent locally: a pass that re-derived them
        from the store per task would cost the whole queue a read for every task
        that completes, and with two thousand tasks behind a hundred slots the
        control plane spends its day re-discovering that the other nineteen
        hundred still have nowhere to go. A task the break skips is not stranded —
        the next completion is the next pass, and the tick re-offers whatever a
        lost wakeup left behind.
        """
        runtime = self._runtimes.of(compute_id)
        if runtime is None:
            return

        async with self._lock(compute_id):
            spare = await self._spare(compute_id, runtime)
            waiting = await self._tasks.waiting(compute_id)
            logger.bind(compute_id=compute_id).debug(
                "{} free slots across {} ready nodes, {} tasks waiting",
                sum(spare.values()),
                len(spare),
                len(waiting),
            )
            for task_id in waiting:
                if not any(spare.values()):
                    return
                task = await self._tasks.get(task_id)
                for execution in task.executions:
                    if execution.id in runtime.dispatched or execution.state != "created" or task.dispatch == "stream":
                        continue
                    free = tuple(node for node in spare if spare[node])
                    if not free:
                        break
                    node_id = await self._placement(task, execution, free)
                    if node_id is None:
                        continue
                    spare[node_id] -= 1
                    await self._launch(task, execution, runtime, node_id)

    async def task(self, task_id: str) -> None:
        """One task's wake. Anything placeable is left for :meth:`resume`.

        This used to place work itself, under the compute lock — and the tick
        wakes every unsettled task, so a deep queue became hundreds of these
        serializing on that lock to each discover there was no slot, while the
        one resume that *had* slots to spend waited at the back of the line and
        a hundred free workers sat idle. Turning the placeable case into a
        ``compute.dispatch`` wakeup collapses all of them into the one pass the
        emitter was built to coalesce.

        Reattaching stays here: it is per-execution by nature, and it takes no
        slot — waiting on a worker for an outcome it owes needs no lock.
        """
        task = await self._tasks.get(task_id)
        if task.state not in ("queued", "running"):
            return

        runtime = self._runtimes.of(task.compute_id)
        if runtime is None:
            return

        placeable = False
        for execution in task.executions:
            if execution.id in runtime.dispatched:
                continue

            match execution.state:
                case "created" if task.dispatch != "stream":
                    placeable = True
                case state if state in IN_FLIGHT:
                    await self._reattach(task, execution, runtime)
                case _:
                    pass

        if placeable:
            self._wake("compute.dispatch", compute_id=task.compute_id)

    async def deleted(self, compute_id: str) -> None:
        """The compute is gone, and every attempt it still owed an answer for gets the one there is.

        Asked once its machines are terminated and its binding released, so nothing is
        ever going to run these, and a caller waiting on one would otherwise wait for its
        deadline — or forever, for a task that named none.

        An attempt that never left the daemon is ``cancelled``: it did not run, and that
        is certain. One that did leave is ``indeterminate``, for the reason :meth:`_lost`
        gives, and the retry decision is not asked — there is nowhere left to try again.
        Settling a task's last attempt is what hands whoever waits on its result the
        error that goes with the verdict.
        """
        self._locks.pop(compute_id, None)
        owed = await self._tasks.owed(compute_id)
        for task_id in owed:
            task = await self._tasks.get(task_id)
            for execution in task.executions:
                match execution.state:
                    case "created" | "assigned":
                        unplaced = f"compute {compute_id} was deleted before the task reached a machine"
                        await self._tasks.observe(execution.id, "cancelled", error=Error(code="compute_not_accepting", message=unplaced, retryable=False))
                    case state if state in PENDING:
                        held = f"compute {compute_id} was deleted while a machine held the task"
                        await self._tasks.observe(execution.id, "indeterminate", error=Error(code="task_indeterminate", message=held, retryable=False))
                        await self._events.record(TaskEvent(compute=compute_id, task=task_id, state="indeterminate", attempt=execution.ordinal))
                    case _:
                        pass

        if owed:
            logger.bind(compute_id=compute_id).info("the compute is deleted: {} task(s) it still owed are answered for", len(owed))

    async def stream(self, task_id: str) -> AsyncIterator[bytes]:
        """A streaming task, dispatched by the caller who is reading it.

        Nobody else can start one. A stream has a far end, and the only process that
        can hold it is the one consuming it — dispatching it from a background pass
        would produce items with nowhere to go, and the sweep would do it again
        after every restart.

        So the request is the dispatch: it places the execution, opens the generator
        on the worker, and pulls it once per item the reader takes. Nothing is
        produced ahead of the reader, and a reader that goes away closes the
        generator on the machine rather than leaving it running for nobody.

        The closing is shielded because it is the cancellation that usually triggers
        it: a caller who abandoned the stream cancels this coroutine, and an
        unshielded ``await`` in the unwinding would be cancelled on the spot — which
        is precisely how a generator gets left running on a machine somebody is still
        paying for.
        """
        task = await self._tasks.get(task_id)
        runtime = self._runtimes.of(task.compute_id)
        if runtime is None or not (free := await self._free(task.compute_id, runtime)):
            raise ComputeNotAcceptingError(f"compute {task.compute_id} has no free node to stream from")

        execution = task.executions[0]
        node_id = free[execution.ordinal % len(free)] if task.rank is None else await self._pinned(task.compute_id, task.rank, free)
        if node_id is None:
            raise ComputeNotAcceptingError(f"compute {task.compute_id} has no free node at rank {task.rank} to stream from")

        code = await self._blobs.get(task.function)
        args = await self._blobs.get(task.args_sha256)

        node = await self._worker(runtime, node_id)

        logger.bind(compute_id=task.compute_id, node_id=node_id).info("streaming execution {} of task {}", execution.id, task.id)
        runtime.dispatched.add(execution.id)
        await self._tasks.observe(execution.id, "started", node_id=node_id)
        await self._events.record(TaskEvent(compute=task.compute_id, task=task.id, state="started"))

        failure: Error | None = None
        try:
            await node.open(execution.id, code, args)
            while True:
                frame = await node.step(execution.id)
                match await _STEPS.decode(frame):
                    case End():
                        break
                    case Failed(error=error, traceback=trace):
                        failure = Error(code="task_failed", message=error, retryable=False, details={"traceback": trace})
                        yield frame
                        break
                    case Chunk():
                        yield frame
        except Exception as exc:
            await self._lost(task, execution, exc, retry.Lost("unknown", node_id))
            return
        finally:
            runtime.dispatched.discard(execution.id)
            await asyncio.shield(node.close(execution.id))

        if failure:
            await self._tasks.observe(execution.id, "failed", error=failure)
            await self._events.record(TaskEvent(compute=task.compute_id, task=task.id, state="failed"))
        else:
            await self._tasks.observe(execution.id, "succeeded")
            await self._events.record(TaskEvent(compute=task.compute_id, task=task.id, state="succeeded"))

    async def _free(self, compute_id: str, runtime: Runtime) -> tuple[str, ...]:
        """Nodes with a slot going spare, in rank order.

        Two exclusions. A draining node still has a worker and a tunnel and would
        take a task perfectly well — which is exactly the problem: it is being taken
        away *because* nothing is running on it, and placing something on it now is
        how that stops being true a moment before it is killed.

        And a node with every slot busy is not a place to put work. Its worker would
        accept the call and hold it in a mailbox nobody can see, and the task would
        stop being queued without having started — which is precisely the signal the
        reconciler reads to decide the pool is too small. A pool that hides its queue
        never grows.
        """
        spare = await self._spare(compute_id, runtime)
        return tuple(node for node in spare if spare[node])

    async def _spare(self, compute_id: str, runtime: Runtime) -> dict[str, int]:
        """Slots going spare per ready node, in rank order — one read, spent locally."""
        compute = await self._computes.get(compute_id)
        slots = compute.spec.worker.concurrency or 1
        holding, _ = await self._tasks.busy(compute_id)

        ready = {node.rank: node.id for node in await self._nodes.of(compute_id) if node.state == "ready"}
        return {
            ready[rank]: slots - holding[ready[rank]]
            for rank in sorted(ready)
            if ready[rank] in runtime.reachable and holding[ready[rank]] < slots
        }

    async def _launch(self, task: Task, execution: Execution, runtime: Runtime, node_id: str) -> None:
        """Hand one execution to one worker, and stop holding the caller open.

        The call is left running as its own task because it lasts as long as the
        user's function does — hours, possibly — and a pass that waited for it would
        be a control plane that stops controlling while somebody trains a model.
        """
        logger.bind(compute_id=task.compute_id, node_id=node_id).info("execution {} of task {} goes out", execution.id, task.id)
        runtime.dispatched.add(execution.id)
        await self._tasks.observe(execution.id, "dispatching", node_id=node_id)
        asyncio.get_running_loop().create_task(self._run(task, execution, runtime, node_id))

    async def _placement(self, task: Task, execution: Execution, free: tuple[str, ...]) -> str | None:
        """One node, or the one node this execution is owed.

        A task that named a rank is pinned to it, retry included: the caller asked
        for that machine, and the next best one is not what they asked for. If it is
        not free the execution waits, because waiting is what being asked for a
        particular machine means.

        A retry of a task that named none goes somewhere else when there is
        somewhere else: the node that lost the last attempt, or raised on it, is the
        one node with a known reason to do it again. With nowhere else to go, it
        goes there anyway.

        A broadcast is pinned the same way, by the ranks frozen when it was
        admitted: rank 3's execution belongs on the machine that is rank 3, and
        placing it elsewhere would run the user's code twice on one node and never
        on another.
        """
        match task.dispatch, task.rank:
            case "one", None:
                previous = next((e.node_id for e in task.executions if e.id == execution.retry_of), None)
                elsewhere = tuple(node for node in free if node != previous) or free
                return elsewhere[execution.ordinal % len(elsewhere)]
            case "one", int(named):
                return await self._pinned(task.compute_id, named, free)
            case _:
                return await self._pinned(task.compute_id, execution.rank, free)

    async def _pinned(self, compute_id: str, rank: int, free: tuple[str, ...]) -> str | None:
        """The machine at that rank, if it is one of the free ones."""
        ranks = {node.rank: node.id for node in await self._nodes.of(compute_id)}
        pinned = ranks.get(rank)
        return pinned if pinned in free else None

    async def _run(self, task: Task, execution: Execution, runtime: Runtime, node_id: str) -> None:
        started = False
        try:
            code = await self._blobs.get(task.function)
            args = await self._blobs.get(task.args_sha256)
            decision = await self._blobs.get(task.retry) if task.retry else b""

            node = await self._worker(runtime, node_id)

            await self._tasks.observe(execution.id, "started", node_id=node_id)
            await self._events.record(TaskEvent(compute=task.compute_id, task=task.id, state="started", attempt=execution.ordinal))
            started = True

            recorded = tuple(runtime.recorded.pop(node_id, ()))
            outcome = await _OUTCOMES.decode(await node.run(execution.id, code, args, decision, execution.ordinal, recorded))
            await self._settle(task, execution, outcome, node_id)
            runtime.recorded.setdefault(node_id, set()).add(execution.id)
        except LINK_ERRORS as exc:
            logger.bind(compute_id=task.compute_id, node_id=node_id).warning(
                "the link dropped with execution {} in flight ({}); the worker will be asked for it when the link is back",
                execution.id,
                exc,
            )
            await asyncio.sleep(RELINK)
            await self._await(task, execution, runtime, node_id)
        except Exception as exc:
            await self._lost(task, execution, exc, retry.Lost("unknown" if started else "never_started", node_id))
        finally:
            runtime.dispatched.discard(execution.id)
            self._wake("task.changed", task_id=task.id)
            self._wake("compute.dispatch", compute_id=task.compute_id)

    async def _reattach(self, task: Task, execution: Execution, runtime: Runtime) -> None:
        """An attempt in flight that nothing in this process is waiting on.

        The daemon went away and came back; the machine never noticed. The worker
        has been running the user's function the whole time and owes its outcome,
        so the answer is to wait on it for that rather than to declare a loss and
        run it twice. The wait is held in the background and marked dispatched, like
        any call in flight, so the passes that follow leave the attempt alone.

        Whether the node went away is the store's to say, and a node the connector
        holds has not. After a restart the connector takes hold of every node again,
        writing it ``connecting`` and ``bootstrapping`` on the way back to ``ready``,
        for as long as a local wheel takes to build and each link takes to come up.
        A node in any of those, or one this process does not hold ready yet, is
        left to a later pass — not called lost, and the attempt run a second time
        on a worker that is still running the first.
        """
        if execution.node_id is None:
            return

        node = await self._nodes.get(task.compute_id, execution.node_id)
        if node.state not in HELD:
            await self._lost(
                task,
                execution,
                RuntimeError(f"node {execution.node_id} went away while it held the task"),
                retry.Lost("node_gone", execution.node_id),
            )
            return
        if node.state != "ready" or execution.node_id not in runtime.ready:
            return

        async def rejoin(node_id: str) -> None:
            try:
                answered = await self._await(task, execution, runtime, node_id)
            finally:
                runtime.dispatched.discard(execution.id)
            if answered:
                self._wake("task.changed", task_id=task.id)
                self._wake("compute.dispatch", compute_id=task.compute_id)

        logger.bind(compute_id=task.compute_id, node_id=execution.node_id).debug("waiting on the worker for execution {}", execution.id)
        runtime.dispatched.add(execution.id)
        asyncio.get_running_loop().create_task(rejoin(execution.node_id))

    async def _await(self, task: Task, execution: Execution, runtime: Runtime, node_id: str) -> bool:
        """Wait on the worker for an attempt's outcome, across every drop of its link, and settle it.

        The worker answers when the function is done, so this is a call held open the
        way the one that carried the attempt was, and nobody asks twice. A link that
        drops under it is waited on, and the worker asked again once the link is back:
        on a provider that cuts its links on a clock, that is every few minutes for as
        long as the function runs, and at no other time.

        A worker that has never heard of the execution is a worker that restarted
        under it, and that is genuinely lost — the one case where the code may or
        may not have run, and we say so instead of guessing.

        Returns whether the attempt got its verdict. It does not when the node's
        channel is gone for good, or when the wait broke on something other than the
        link: whether the attempt went with its node is the store's to say once the
        node is reported, and the tick brings the task back to :meth:`task`, which
        hears it or waits again.
        """
        log = logger.bind(compute_id=task.compute_id, node_id=node_id)

        async def answer() -> Lookup:
            while True:
                try:
                    await runtime.linked(node_id)
                    member = await runtime.member(node_id)
                    system = await runtime.system(node_id)
                    return await _LOOKUPS.decode(await system.service(worker.Control, at=member).result(execution.id))
                except LINK_ERRORS as exc:
                    log.debug("the link dropped while waiting for execution {}: {}", execution.id, exc)
                    await asyncio.sleep(RELINK)

        try:
            match await answer():
                case Unknown():
                    await self._lost(task, execution, RuntimeError("the worker no longer has it"), retry.Lost("worker_restarted", node_id))
                case Done() | Failed() | Lost() as outcome:
                    await self._settle(task, execution, outcome, node_id)
                    runtime.recorded.setdefault(node_id, set()).add(execution.id)
        except SshUnavailableError as exc:
            log.debug("stopped waiting for execution {}: {}", execution.id, exc)
            return False
        except Exception:
            log.exception("could not wait for execution {}; the next pass waits again", execution.id)
            return False
        return True

    async def _settle(self, task: Task, execution: Execution, outcome: Outcome, node_id: str | None) -> None:
        log = logger.bind(compute_id=task.compute_id)
        match outcome:
            case Done(value=value):
                log.info("execution {} succeeded", execution.id)
                await self._tasks.observe(execution.id, "succeeded", result_sha256=await self._blobs.store(value))
                await self._events.record(TaskEvent(compute=task.compute_id, task=task.id, state="succeeded", attempt=execution.ordinal))
            case Failed(error=error, traceback=trace, retry=again):
                log.info("execution {} failed: {}", execution.id, error)
                failure = Error(code="task_failed", message=error, retryable=False, details={"traceback": trace})
                if again:
                    await self._again(task, execution, "failed", failure)
                    return
                await self._tasks.observe(execution.id, "failed", error=failure)
                await self._events.record(TaskEvent(compute=task.compute_id, task=task.id, state="failed", attempt=execution.ordinal))
            case Lost(error=error):
                await self._lost(task, execution, RuntimeError(error), retry.Lost("process_died", node_id))

    async def _lost(self, task: Task, execution: Execution, exc: Exception, loss: retry.Lost) -> None:
        """The call died, and we do not know whether the function did.

        This is the only honest verdict available. The worker may have run the
        user's code to completion and lost the reply on the way back, so calling it
        ``failed`` would be the system deciding on the user's behalf that a duplicate
        side effect is acceptable. Whether to try again is the task's retry decision
        to make, and it is asked here with what the daemon knows: the loss, and
        which attempt it was.

        The worker says the same thing itself, as ``Lost``, when the subprocess
        running the function died under it: nothing was raised, and the function
        may have done half of what it was going to.
        """
        logger.warning("execution {} lost", execution.id, exc_info=exc)
        failure = Error(code="task_indeterminate", message=f"{type(exc).__name__}: {exc}", retryable=False)
        if await self._retry(task, loss, execution.ordinal):
            await self._again(task, execution, "indeterminate", failure)
            return
        await self._tasks.observe(execution.id, "indeterminate", error=failure)
        await self._events.record(TaskEvent(compute=task.compute_id, task=task.id, state="indeterminate", attempt=execution.ordinal))

    async def _retry(self, task: Task, loss: retry.Lost, attempt: int) -> bool:
        if task.retry is None:
            return False
        try:
            decision = await _DECISIONS.decode(await self._blobs.get(task.retry))
        except Exception:
            logger.bind(compute_id=task.compute_id).warning("the retry decision of task {} cannot be loaded here; not retrying", task.id, exc_info=True)
            return False
        return retry.decide(decision, loss, attempt)

    async def _again(self, task: Task, execution: Execution, state: ExecutionState, error: Error) -> None:
        """The attempt is over and the next one is written down, in one breath.

        The task never shows a terminal state in between, so a caller waiting on it
        is not handed the ending a moment before the retry that was meant to spare
        them. The new execution is ``created``, and the next pass places it.
        """
        logger.bind(compute_id=task.compute_id).info(
            "execution {} {}; attempt {} of task {} is written down", execution.id, state, execution.ordinal + 1, task.id
        )
        await self._tasks.observe(execution.id, state, error=error, again=True)
        await self._events.record(TaskEvent(compute=task.compute_id, task=task.id, state="retrying", attempt=execution.ordinal + 1))
        self._wake("compute.dispatch", compute_id=task.compute_id)

    async def _worker(self, runtime: Runtime, node_id: str) -> worker.Worker:
        system = await runtime.system(node_id)
        return system.service(worker.Worker, at=await runtime.member(node_id))

    def _lock(self, compute_id: str) -> asyncio.Lock:
        """One placement decision per compute at a time.

        Two tasks arriving together would otherwise both read the same free slot and
        both take it, and the node they agreed on would hold one of them in a mailbox
        — invisible to the queue, and to the pressure the queue is there to express.
        """
        return self._locks.setdefault(compute_id, asyncio.Lock())

