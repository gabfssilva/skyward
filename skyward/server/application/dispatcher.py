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

from skyward.server.application.runtimes import Runtime, Runtimes
from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.tasks import TaskStore
from skyward.shared import codec
from skyward.shared.errors import ComputeNotAcceptingError
from skyward.shared.events import Event, TaskEvent
from skyward.shared.frames import Chunk, Done, End, Failed, Lookup, Outcome, Pending, Step, Unknown
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
execution's id, so a reply lost to a dropped link is a reply that can be asked
for again — see :meth:`Dispatcher._reattach`. Anything else that escapes the call
is the honest ``indeterminate``.
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
        slot — asking a worker for an outcome it already holds needs no lock.
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
        node_id = free[execution.ordinal % len(free)]

        code = await self._blobs.get(task.function)
        args = await self._blobs.get(task.args_sha256)

        node = await self._worker(runtime, node_id)

        logger.bind(compute_id=task.compute_id, node_id=node_id).info("streaming execution {} of task {}", execution.id, task.id)
        runtime.dispatched.add(execution.id)
        await self._tasks.observe(execution.id, "started", node_id=node_id)
        await self._record("task.started", TaskEvent(compute=task.compute_id, task=task.id, state="started"))

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
            await self._lost(task, execution, exc)
            return
        finally:
            runtime.dispatched.discard(execution.id)
            await asyncio.shield(node.close(execution.id))

        if failure:
            await self._tasks.observe(execution.id, "failed", error=failure)
            await self._record("task.failed", TaskEvent(compute=task.compute_id, task=task.id, state="failed"))
        else:
            await self._tasks.observe(execution.id, "succeeded")
            await self._record("task.succeeded", TaskEvent(compute=task.compute_id, task=task.id, state="succeeded"))

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

        A broadcast is pinned: its ranks were frozen when it was admitted, and rank
        3's execution belongs on the machine that is rank 3. If that machine is not
        there, the execution waits — placing it elsewhere would run the user's code
        twice on one node and never on another.
        """
        if task.dispatch == "one":
            return free[execution.ordinal % len(free)]

        ranks = {node.rank: node.id for node in await self._nodes.of(task.compute_id)}
        pinned = ranks.get(execution.rank)
        return pinned if pinned in free else None

    async def _run(self, task: Task, execution: Execution, runtime: Runtime, node_id: str) -> None:
        try:
            code = await self._blobs.get(task.function)
            args = await self._blobs.get(task.args_sha256)

            node = await self._worker(runtime, node_id)

            await self._tasks.observe(execution.id, "started", node_id=node_id)
            await self._record("task.started", TaskEvent(compute=task.compute_id, task=task.id, state="started"))

            await self._settle(task, execution, await _OUTCOMES.decode(await node.run(execution.id, code, args)))
        except LINK_ERRORS as exc:
            logger.bind(compute_id=task.compute_id, node_id=node_id).warning(
                "the link dropped with execution {} in flight ({}); the worker will be asked for it when the link is back",
                execution.id,
                exc,
            )
        except Exception as exc:
            await self._lost(task, execution, exc)
        finally:
            runtime.dispatched.discard(execution.id)
            self._wake("task.changed", task_id=task.id)
            self._wake("compute.dispatch", compute_id=task.compute_id)

    async def _reattach(self, task: Task, execution: Execution, runtime: Runtime) -> None:
        """An attempt this process did not dispatch, and did not see finish.

        The daemon went away and came back; the machine never noticed. The worker
        has been running the user's function the whole time and is holding its
        outcome, so the answer is to ask for it rather than to declare a loss and
        run it twice.

        A worker that has never heard of the execution is a worker that restarted
        under it, and that is genuinely lost — the one case where the code may or
        may not have run, and we say so instead of guessing.
        """
        if execution.node_id is None:
            return

        node = await self._nodes.get(task.compute_id, execution.node_id)
        if node.state != "ready" or execution.node_id not in runtime.ready:
            await self._lost(task, execution, RuntimeError(f"node {execution.node_id} went away while it held the task"))
            return
        if execution.node_id not in runtime.reachable:
            return

        logger.bind(compute_id=task.compute_id, node_id=execution.node_id).debug("asking the worker for execution {}", execution.id)
        try:
            member = await runtime.member(execution.node_id)
            system = await runtime.system(execution.node_id)
            control = system.service(worker.Control, at=member)
            lookup = await _LOOKUPS.decode(await control.result(execution.id))
        except LINK_ERRORS as exc:
            logger.bind(compute_id=task.compute_id, node_id=execution.node_id).debug("the link dropped while asking: {}", exc)
            return

        match lookup:
            case Pending():
                logger.bind(node_id=execution.node_id).debug("execution {} is still running", execution.id)
            case Unknown():
                await self._lost(task, execution, RuntimeError("the worker no longer has it"))
            case Done() | Failed() as outcome:
                await self._settle(task, execution, outcome)
                self._wake("task.changed", task_id=task.id)

    async def _settle(self, task: Task, execution: Execution, outcome: Outcome) -> None:
        log = logger.bind(compute_id=task.compute_id)
        match outcome:
            case Done(value=value):
                log.info("execution {} succeeded", execution.id)
                await self._tasks.observe(execution.id, "succeeded", result_sha256=await self._blobs.store(value))
                await self._record("task.succeeded", TaskEvent(compute=task.compute_id, task=task.id, state="succeeded"))
            case Failed(error=error, traceback=trace):
                log.info("execution {} failed: {}", execution.id, error)
                await self._tasks.observe(
                    execution.id,
                    "failed",
                    error=Error(code="task_failed", message=error, retryable=False, details={"traceback": trace}),
                )
                await self._record("task.failed", TaskEvent(compute=task.compute_id, task=task.id, state="failed"))

    async def _lost(self, task: Task, execution: Execution, exc: Exception) -> None:
        """The call died, and we do not know whether the function did.

        This is the only honest verdict available. The worker may have run the
        user's code to completion and lost the reply on the way back, so calling it
        ``failed`` — which is retryable without asking — would be the system deciding
        on the user's behalf that a duplicate side effect is acceptable.
        """
        logger.warning("execution {} lost", execution.id, exc_info=exc)
        await self._tasks.observe(
            execution.id,
            "indeterminate",
            error=Error(code="task_indeterminate", message=f"{type(exc).__name__}: {exc}", retryable=False),
        )
        await self._record("task.indeterminate", TaskEvent(compute=task.compute_id, task=task.id, state="indeterminate"))

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

    async def _record(self, name: str, payload: TaskEvent) -> None:
        await self._events.record(
            name,
            await codec.json(Event).encode(payload),
            compute=payload.compute,
            task=payload.task,
        )
