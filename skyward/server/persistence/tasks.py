from __future__ import annotations

import asyncio
import base64
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from datetime import datetime, timedelta
from itertools import batched
from statistics import fmean
from typing import Any, NamedTuple

import msgspec
from msgspec import UNSET
from piccolo.columns import Column
from piccolo.querystring import QueryString

from skyward.server.application.ports import Held, Pace
from skyward.server.persistence.computes import LIVE, ComputeStore
from skyward.server.persistence.db import POSITIONS
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.store import ident, now, once, packed, unpacked
from skyward.server.persistence.tables import ComputeRow, ExecutionRow, TaskRow
from skyward.shared.errors import (
    ComputeNotAcceptingError,
    DuplicationNotAcknowledgedError,
    NotFoundError,
    TaskFailedError,
    TaskIndeterminateError,
)
from skyward.shared.schemas import (
    ComputeState,
    Dispatch,
    Error,
    Execution,
    ExecutionCreate,
    ExecutionState,
    Page,
    Task,
    TaskCreate,
    TaskOrder,
    TaskState,
)

ACCEPTING: tuple[ComputeState, ...] = ("requested", "provisioning", "ready", "degraded")

PENDING: tuple[ExecutionState, ...] = ("created", "assigned", "dispatching", "accepted", "started", "cancel_requested")

UNPLACED: tuple[ExecutionState, ...] = ("created", "assigned")
"""Pending, and still in the daemon: no machine has been handed these."""

BATCH = 500
"""How many tasks' attempts one query reads."""


class Pressure(NamedTuple):
    """What a compute's queue asks of its nodes, read once."""

    load: int
    holding: Counter[str]
    owed: frozenset[int]


class Expired(NamedTuple):
    """An attempt answered for because its time ran out."""

    execution: str
    task: str
    compute: str
    ordinal: int
    node: str | None
    """The machine still holding it, which has to be asked to stop; ``None`` for one that never left the queue."""


class TaskStore:
    """Tasks, and the outcome that is derived from their attempts.

    Nothing here polls. A caller waiting on a result waits on an event that the
    write of the last execution sets — the store is not a queue and must never be
    scanned like one.
    """

    def __init__(self, computes: ComputeStore, nodes: NodeStore, blobs: BlobStore) -> None:
        self._computes = computes
        self._nodes = nodes
        self._blobs = blobs
        self._settled: dict[str, asyncio.Event] = defaultdict(asyncio.Event)

    async def submit(self, body: TaskCreate, idempotency_key: str) -> tuple[Task, bool]:
        """Persist the task and its attempts, before anything is dispatched.

        A task that reached a worker but was never written down is a task nobody
        can ask about after a crash — and the worker will happily run it anyway.
        The order is not an implementation detail.

        A task that named no limits takes the compute's, and takes them here: they
        are facts about the task, and a task is only admitted once. The deadlines are
        the attempts', because each attempt waits and runs on a clock of its own.
        """

        async def insert() -> str:
            compute = await self._computes.get(body.compute)
            if compute.status.state not in ACCEPTING:
                raise ComputeNotAcceptingError(f"compute {compute.id} is {compute.status.state}")

            args = body.args_sha256 or await self._blobs.store(body.args_inline or b"")
            if not await self._blobs.exists(args):
                raise NotFoundError(f"no such args blob: {args}")

            task = ident("tsk")
            retry = compute.spec.retry if body.retry is UNSET else body.retry
            await TaskRow(
                id=task,
                compute_id=compute.id,
                generation=compute.generation,
                function=body.function,
                args_sha256=args,
                dispatch=body.dispatch,
                state="queued",
                rank=body.rank,
                decision=retry,
                correlation_id=body.correlation_id,
                submitted_at=now(),
                queue_timeout=_limit(body.queue_timeout_seconds, compute.spec.options.task_queue_timeout),
                run_timeout=_limit(body.run_timeout_seconds, compute.spec.options.task_run_timeout),
            ).save().run()

            for rank in await self._ranks(compute.id, body):
                await self.attempt(task, rank=rank, ordinal=1, retry_of=None)

            return task

        task_id, created = await once("task.submit", idempotency_key, body, insert)
        return await self.get(task_id), created

    async def get(self, task_id: str) -> Task:
        (task,) = await _tasks([await self._row(task_id)])
        return task

    async def list(
        self,
        cursor: str | None,
        limit: int,
        compute: str | None = None,
        states: Sequence[TaskState] = (),
        correlation_id: str | None = None,
        function: str | None = None,
        order: TaskOrder = "submitted",
    ) -> Page[Task]:
        """A page of the tasks the filters match, in ``order``, and how many they match.

        ``submitted`` is newest first. A compute that keeps working keeps adding tasks,
        so what it has the most of is what it has already done: paged from the oldest
        end, the first page is the first tasks the compute ever ran. ``state`` puts
        what is running first, newest submitted first, then the queue in the order it
        is served, then the finished, latest to finish first. ``finished`` is latest to
        finish first, and then what has not finished, newest submitted first.

        ``function`` is a name, and every upload of code under it: a function edited
        and sent again is another digest and the same function.

        The cursor is the position the page ended on, not a task: a task that changes
        state moves in the ``state`` and ``finished`` orders, and a walk resumed from
        where that task went would skip or repeat whatever it passed over.
        """
        position = POSITIONS[order]
        narrowed = [
            *([QueryString("compute_id = {}", compute)] if compute else []),
            *([QueryString(f"state IN ({', '.join('{}' for _ in states)})", *states)] if states else []),
            *([QueryString("correlation_id = {}", correlation_id)] if correlation_id else []),
            *([QueryString("function IN (SELECT sha256 FROM functions WHERE name = {})", function)] if function else []),
        ]
        matched = QueryString(" AND ".join("{}" for _ in narrowed) or "1", *narrowed)

        seek = QueryString("1")
        if cursor:
            at, past = _position(cursor, order)
            seek = QueryString(f"({position}) >= {{}} AND (({position}), id) > ({{}}, {{}})", at, at, past)

        page = f"SELECT id, {position} AS position FROM tasks WHERE {{}} AND {{}} ORDER BY position, id LIMIT {{}}"
        rows = await TaskRow.raw(page, matched, seek, limit).run()
        found = {row.id: row for row in await TaskRow.objects().where(TaskRow.id.is_in([row["id"] for row in rows]))} if rows else {}
        (counted,) = await TaskRow.raw("SELECT count(*) AS total FROM tasks WHERE {}", matched).run()

        return Page(
            items=await _tasks([found[row["id"]] for row in rows]),
            next_cursor=_cursor(order, rows[-1]["position"], rows[-1]["id"]) if len(rows) == limit else None,
            total=counted["total"],
        )

    async def cancel(self, task_id: str, idempotency_key: str) -> Task:
        """Stop it if it has not started; ask it to stop if it has.

        An execution that has started is Python already running on a machine. It is
        moved to ``cancel_requested`` and stays there until the worker confirms it
        stopped — code that may still be running is never declared cancelled.
        """

        async def request() -> str:
            for execution in await self.attempts(task_id):
                if execution.state in PENDING:
                    await self.observe(
                        execution.id,
                        "cancel_requested" if execution.state == "started" else "cancelled",
                    )
            return task_id

        await once("task.cancel", idempotency_key, None, request)
        return await self.get(task_id)

    def close(self) -> None:
        """Answer every result being waited on, now, with what there is.

        The daemon is going away, and its server cuts whatever is still open once it
        has waited a moment. A long poll cut before it answered is answered for it with
        a 500, which the caller can only read as the daemon's verdict on the task, and
        gives up. Woken here instead, each wait reads its task as it stands and says
        there is no outcome yet — the answer that sends the caller back to ask again,
        of whichever daemon is there by then.
        """
        for settled in self._settled.values():
            settled.set()

    async def result(self, task_id: str, wait_seconds: int) -> bytes | None:
        task = await self.get(task_id)

        if task.state in ("queued", "running") and wait_seconds:
            try:
                async with asyncio.timeout(wait_seconds):
                    await self._settled[task_id].wait()
            except TimeoutError:
                return None
            task = await self.get(task_id)

        match task.state:
            case "succeeded" if task.result_sha256:
                return await self._blobs.get(task.result_sha256)
            case "failed" | "cancelled" | "timed_out":
                raise TaskFailedError(_reason(task), task=task_id, state=task.state, **_cause(task))
            case "indeterminate":
                raise TaskIndeterminateError(_reason(task), task=task_id, **_cause(task))
            case _:
                return None

    async def observe(
        self,
        execution_id: str,
        state: ExecutionState,
        node_id: str | None = None,
        result_sha256: str | None = None,
        error: Error | None = None,
        again: bool = False,
        stopping: bool = False,
    ) -> bool:
        """What a worker did with an attempt, and the outcome that follows from it.

        The first verdict stands. An attempt already answered for is not written
        again, and ``False`` says so: a worker whose answer arrives after the attempt
        timed out is too late to turn it into a success, and a deadline passing a
        moment after the answer landed is too late to turn it into a timeout. The
        write is conditional on the row still pending, so the tick and the dispatcher
        racing over one attempt cannot both win.

        ``again`` writes the next attempt down in the same breath as this one's end,
        so the task is never terminal in between: a caller waiting on the result
        would otherwise be woken by the ending and handed it, a moment before the
        retry that was meant to spare them exactly that. ``stopping`` is an ending
        written while a machine still runs the attempt, whose slot stays taken until
        :meth:`release`.

        Starting is when the run's clock starts: the deadline stops being the wait's
        and becomes the run's, counted from here rather than from the submission.
        """
        row = await ExecutionRow.objects().where(ExecutionRow.id == execution_id).first()
        if row is None:
            raise NotFoundError(f"no such execution: {execution_id}")
        if row.state not in PENDING:
            return False

        changes: dict[Column | str, Any] = {ExecutionRow.state: state}
        if node_id:
            changes[ExecutionRow.node_id] = node_id
        if result_sha256:
            changes[ExecutionRow.result_sha256] = result_sha256
        if error:
            changes[ExecutionRow.error] = await packed(error)
        if state == "started" and row.started_at is None:
            started = now()
            limit = await TaskRow.select(TaskRow.run_timeout).where(TaskRow.id == row.task_id).first()
            run = limit["run_timeout"] if limit else None
            changes[ExecutionRow.started_at] = started
            changes[ExecutionRow.deadline_at] = started + timedelta(seconds=run) if run else None
        if state not in PENDING:
            changes[ExecutionRow.finished_at] = now()
            changes[ExecutionRow.stopping] = stopping

        landed = await ExecutionRow.update(changes).where(
            (ExecutionRow.id == execution_id) & ExecutionRow.state.is_in(list(PENDING)),
        ).returning(ExecutionRow.id).run()
        if not landed:
            return False

        if again:
            await self.attempt(row.task_id, row.rank, row.ordinal + 1, retry_of=row.id)

        await self.settle(row.task_id)
        return True

    async def release(self, execution_id: str) -> None:
        """The worker let go of an attempt that was answered for while it still ran: its slot is free again."""
        await ExecutionRow.update({ExecutionRow.stopping: False}).where(
            (ExecutionRow.id == execution_id) & ExecutionRow.stopping.eq(True),
        ).run()

    async def settle(self, task_id: str) -> None:
        """Recompute the task from its attempts. The only writer of ``TaskRow.state``.

        Only the latest attempt of each rank counts: a retry supersedes what it
        retried, and a task whose second attempt succeeded is a task that
        succeeded — the failed first attempt stays in the history and out of the
        verdict.
        """
        row = await self._row(task_id)
        latest = _latest(await self.attempts(task_id))
        state = _verdict(latest)

        row.state = state
        if state == "succeeded":
            row.result_sha256 = next((e.result_sha256 for e in latest if e.result_sha256), None)
        if state not in ("queued", "running") and row.finished_at is None:
            row.finished_at = now()
        await row.save().run()

        if state not in ("queued", "running"):
            self._settled[task_id].set()

    async def _ranks(self, compute: str, body: TaskCreate) -> tuple[int, ...]:
        """Which nodes this task is for, decided once, at admission.

        A broadcast freezes the set of ready nodes here. A node that joins a second
        later does not get an execution: the user asked for the compute they had,
        and silently growing the fan-out would make ``@`` mean something different
        on every call.
        """
        if body.dispatch in ("one", "stream"):
            return (body.rank or 0,)

        nodes = await self._nodes.list(compute, include_terminal=False, generation=None)
        return tuple(node.rank for node in nodes.items if node.state == "ready")

    async def attempt(self, task_id: str, rank: int, ordinal: int, retry_of: str | None) -> str:
        """Write one attempt down, with the wait it is allowed starting now — a retry's included."""
        limit = await TaskRow.select(TaskRow.queue_timeout).where(TaskRow.id == task_id).first()
        wait = limit["queue_timeout"] if limit else None
        execution = ident("exe")
        await ExecutionRow(
            id=execution,
            task_id=task_id,
            rank=rank,
            ordinal=ordinal,
            state="created",
            retry_of=retry_of,
            deadline_at=now() + timedelta(seconds=wait) if wait else None,
        ).save().run()
        return execution

    async def attempts(self, task_id: str) -> list[ExecutionRow]:
        return await ExecutionRow.objects().where(ExecutionRow.task_id == task_id).order_by(ExecutionRow.ordinal)

    async def owners(self, executions: Collection[str]) -> dict[str, str]:
        """The task each of these executions is an attempt at, leaving out any the store never wrote."""
        if not executions:
            return {}
        rows = await ExecutionRow.select(ExecutionRow.id, ExecutionRow.task_id).where(ExecutionRow.id.is_in(list(executions)))
        return {row["id"]: row["task_id"] for row in rows}

    async def unsettled(self) -> tuple[str, ...]:
        """Tasks that have not reached a verdict and still have a compute to reach one on — what the sweep re-offers.

        A task that has its verdict but an attempt still ``stopping`` is among them: its
        machine has not let go yet, and a daemon that restarted meanwhile is the one that
        has to ask it again.

        A deleted compute's tasks are not among them. Nothing is left to run them, and
        offering them anyway is a read per task per tick that grows with every compute
        the daemon has ever deleted; :meth:`stranded` is how the sweep finds them instead.
        """
        live = ComputeRow.select(ComputeRow.id).where(ComputeRow.status_state.is_in(list(LIVE)))
        held = ExecutionRow.select(ExecutionRow.task_id).where(ExecutionRow.stopping.eq(True))
        rows = await TaskRow.select(TaskRow.id).where(
            (TaskRow.state.is_in(["queued", "running"]) | TaskRow.id.is_in(held)) & TaskRow.compute_id.is_in(live),
        )
        return tuple(row["id"] for row in rows)

    async def stranded(self) -> tuple[str, ...]:
        """Deleted computes still holding an attempt without a verdict.

        Their attempts are answered for the moment the compute is deleted, so this is
        empty unless that moment was missed: a daemon that died in between, or a store
        written before anything answered for them.

        An attempt and not merely a task, because an attempt is what gets answered for.
        A broadcast admitted while no node was ready has none, and a task a crash left
        between an attempt's end and its verdict has only finished ones: a compute
        offered for either would be offered on every tick, with nothing to answer.
        """
        deleted = ComputeRow.select(ComputeRow.id).where(ComputeRow.status_state == "deleted")
        owing = ExecutionRow.select(ExecutionRow.task_id).where(ExecutionRow.state.is_in(list(PENDING)))
        rows = await TaskRow.select(TaskRow.compute_id).where(
            TaskRow.state.is_in(["queued", "running"]) & TaskRow.compute_id.is_in(deleted) & TaskRow.id.is_in(owing),
        ).distinct()
        return tuple(row["compute_id"] for row in rows)

    async def owed(self, compute: str) -> tuple[str, ...]:
        """Every task of this compute with an attempt still without a verdict, oldest first, whatever state the compute is in."""
        owing = ExecutionRow.select(ExecutionRow.task_id).where(ExecutionRow.state.is_in(list(PENDING)))
        rows = await TaskRow.select(TaskRow.id).where(
            (TaskRow.compute_id == compute) & TaskRow.state.is_in(["queued", "running"]) & TaskRow.id.is_in(owing),
        ).order_by(TaskRow.submitted_at)
        return tuple(row["id"] for row in rows)

    async def expire(self) -> tuple[Expired, ...]:
        """Time out the attempts whose deadline has passed.

        It runs on the tick because a deadline passing is not something anybody does:
        there is no write to react to, and the only way to notice is to look.

        Each attempt is on its own clock — the wait until it starts, then the run — so
        a task submitted with a week of work ahead of it is not timed out by the week.
        One still waiting for a machine is simply over, and certain never to have run.
        One a machine holds is answered for too, since its caller asked for an answer
        within a window, but it is written ``stopping``: the function is still running
        there, its slot stays taken until the worker lets go, and asking it to stop is
        the dispatcher's.
        """
        rows = await ExecutionRow.select(
            ExecutionRow.id, ExecutionRow.task_id, ExecutionRow.ordinal, ExecutionRow.node_id, ExecutionRow.state, ExecutionRow.started_at
        ).where(ExecutionRow.state.is_in(list(PENDING)) & (ExecutionRow.deadline_at < now()))

        expired: list[Expired] = []
        for row in rows:
            task = await TaskRow.select(TaskRow.compute_id, TaskRow.queue_timeout, TaskRow.run_timeout).where(TaskRow.id == row["task_id"]).first()
            if task is None:
                continue
            held = None if row["state"] in UNPLACED else row["node_id"]
            error = (
                Error(code="task_failed", message=f"ran for more than {task['run_timeout']:g}s", retryable=False, details={"timeout": "run"})
                if row["started_at"]
                else Error(code="task_failed", message=f"waited more than {task['queue_timeout']:g}s to start", retryable=False, details={"timeout": "queue"})
            )
            if await self.observe(row["id"], "timed_out", error=error, stopping=held is not None):
                expired.append(Expired(row["id"], row["task_id"], task["compute_id"], row["ordinal"], held))

        return tuple(expired)

    async def pressure(self, compute: str) -> Pressure:
        """What this compute's queue asks of its nodes, from one read.

        The load is how many attempts the compute still owes an answer for: queued
        and running together, because they are the same demand seen a moment apart.
        Sizing the pool to what is running would size it to what the pool can
        already do, and a queue would never be a reason to grow.

        An attempt timed out while its machine still runs it holds that machine and
        is not demand: nothing more will be asked of the queue on its behalf, but the
        slot is not free until the worker lets go of it.
        """
        pending = await self._pending(compute)
        return Pressure(
            load=len(pending),
            holding=Counter(held.node for held in await self.held(compute)),
            owed=frozenset(row["rank"] for row in pending),
        )

    async def held(self, compute: str) -> tuple[Held, ...]:
        """The attempts this compute's machines are holding, the earliest started first.

        Placed and still owed an answer, or answered for while the machine still ran
        them: a slot is the worker's until it lets go, whatever the attempt's verdict.
        """
        live = TaskRow.select(TaskRow.id).where((TaskRow.compute_id == compute) & TaskRow.state.is_in(["queued", "running"]))
        stopping = TaskRow.select(TaskRow.id).where(TaskRow.compute_id == compute)
        rows = await ExecutionRow.select(ExecutionRow.node_id, ExecutionRow.task_id, ExecutionRow.ordinal, ExecutionRow.started_at).where(
            ExecutionRow.node_id.is_not_null()
            & (
                (ExecutionRow.task_id.is_in(live) & ExecutionRow.state.is_in(list(PENDING)))
                | (ExecutionRow.stopping.eq(True) & ExecutionRow.task_id.is_in(stopping))
            ),
        ).order_by(ExecutionRow.started_at)
        if not rows:
            return ()

        functions = await TaskRow.select(TaskRow.id, TaskRow.function).where(TaskRow.id.is_in(list({row["task_id"] for row in rows})))
        named = {row["id"]: row["function"] for row in functions}
        return tuple(Held(row["node_id"], row["task_id"], row["ordinal"], named[row["task_id"]], row["started_at"]) for row in rows)

    async def pace(self, compute: str, since: datetime) -> Pace:
        """How many of this compute's tasks finished since ``since``, and how long they took.

        A task takes from its first attempt starting to its verdict, retries and all,
        which is the wait a caller holding its future sat through once it was running.
        One that never started — cancelled in the queue, timed out waiting — finished
        without taking any time, and is counted without being averaged.
        """
        finished = await TaskRow.select(TaskRow.id, TaskRow.finished_at).where((TaskRow.compute_id == compute) & (TaskRow.finished_at >= since))
        if not finished:
            return Pace(0, None)

        started: dict[str, datetime] = {}
        for ids in batched([row["id"] for row in finished], BATCH):
            for row in await ExecutionRow.select(ExecutionRow.task_id, ExecutionRow.started_at).where(
                ExecutionRow.task_id.is_in(list(ids)) & ExecutionRow.started_at.is_not_null(),
            ):
                started[row["task_id"]] = min(started.get(row["task_id"], row["started_at"]), row["started_at"])

        took = [(row["finished_at"] - started[row["id"]]).total_seconds() for row in finished if row["id"] in started]
        return Pace(len(finished), fmean(took) if took else None)

    async def busy(self, compute: str) -> tuple[Counter[str], frozenset[int]]:
        """How much each node is holding, and which ranks are spoken for.

        Two different claims. The count is what the dispatcher reads before placing
        anything — a node with every slot taken gets nothing more, which is the only
        reason a queue exists at all, and a queue is the only thing the reconciler
        can read as pressure. Placing eagerly onto a busy node would drain the queue
        into a mailbox nobody can see, and the pool would never grow.

        The ranks are the broadcast's: it froze them when it was admitted, and rank 3
        is owed an execution even if nothing has been placed on it yet. Kill the node
        that is rank 3 and the broadcast waits for a machine that is never coming back.
        """
        pressure = await self.pressure(compute)
        return pressure.holding, pressure.owed

    async def waiting(self, compute: str) -> tuple[str, ...]:
        """Tasks of this compute with an attempt still to be placed, oldest first.

        Two filters, both load-bearing. Only tasks holding a ``created`` execution
        appear — a running task whose attempts are all on machines has nothing to
        offer a free slot, and walking a hundred of them to reach the first task
        that does is what the dispatcher used to spend its passes on. And the
        order is the fairness: the dispatcher stops offering once the slots are
        spent, so whoever is first in this tuple is whoever runs next.
        """
        placeable = ExecutionRow.select(ExecutionRow.task_id).where(ExecutionRow.state == "created")
        rows = await TaskRow.select(TaskRow.id).where(
            (TaskRow.compute_id == compute)
            & TaskRow.state.is_in(["queued", "running"])
            & TaskRow.id.is_in(placeable),
        ).order_by(TaskRow.submitted_at)
        return tuple(row["id"] for row in rows)

    async def _pending(self, compute: str) -> list[dict[str, Any]]:
        live = TaskRow.select(TaskRow.id).where(
            (TaskRow.compute_id == compute) & TaskRow.state.is_in(["queued", "running"]),
        )
        return await ExecutionRow.select(ExecutionRow.node_id, ExecutionRow.rank).where(
            ExecutionRow.task_id.is_in(live) & ExecutionRow.state.is_in(list(PENDING)),
        )

    async def _row(self, task_id: str) -> TaskRow:
        row = await TaskRow.objects().where(TaskRow.id == task_id).first()
        if row is None:
            raise NotFoundError(f"no such task: {task_id}")
        return row


class ExecutionStore:
    def __init__(self, tasks: TaskStore) -> None:
        self._tasks = tasks

    async def list(self, task_id: str) -> Page[Execution]:
        return Page(items=tuple([await _to_execution(row) for row in await self._tasks.attempts(task_id)]))

    async def get(self, task_id: str, ordinal: int) -> Execution:
        row = await ExecutionRow.objects().where(
            (ExecutionRow.task_id == task_id) & (ExecutionRow.ordinal == ordinal),
        ).first()
        if row is None:
            raise NotFoundError(f"no such execution: {task_id}/{ordinal}")
        return await _to_execution(row)

    async def create(self, task_id: str, body: ExecutionCreate, idempotency_key: str) -> Task:
        """Retry: another physical attempt at the same task, never another task.

        Retrying an ``indeterminate`` outcome is refused unless the caller says out
        loud that a duplicate is acceptable. We do not know whether the previous
        attempt had side effects, and the one thing we must not do is pretend we
        do.
        """

        async def retry() -> str:
            task = await self._tasks.get(task_id)
            if task.state == "indeterminate" and not body.acknowledge_duplication:
                raise DuplicationNotAcknowledgedError(
                    f"task {task_id} may have run; retrying requires acknowledge_duplication",
                    task=task_id,
                )

            attempts = _latest(await self._tasks.attempts(task_id))
            wanted = body.ranks or tuple(e.rank for e in attempts)

            for previous in (e for e in attempts if e.rank in wanted):
                await self._tasks.attempt(task_id, previous.rank, previous.ordinal + 1, retry_of=previous.id)

            await self._tasks.settle(task_id)
            return task_id

        await once("execution.create", idempotency_key, body, retry)
        return await self._tasks.get(task_id)


def _latest(attempts: list[ExecutionRow]) -> list[ExecutionRow]:
    by_rank: dict[int, ExecutionRow] = {}
    for attempt in attempts:
        if attempt.rank not in by_rank or attempt.ordinal > by_rank[attempt.rank].ordinal:
            by_rank[attempt.rank] = attempt
    return list(by_rank.values())


def _verdict(attempts: list[ExecutionRow]) -> TaskState:
    """One outcome out of many attempts.

    Nothing is terminal while an attempt is still in flight — a broadcast where one
    node failed and another is still running is a task that is still running, and
    saying otherwise would hand the caller a result while the machine is still
    computing it.

    Once they are all in, the worst one wins, and ``indeterminate`` is the worst:
    a broadcast in which one node cleanly failed and another is unaccounted for is
    a task that may have run somewhere, and says so.
    """
    states = {attempt.state for attempt in attempts}

    if not states:
        return "queued"

    if pending := states & set(PENDING):
        return "queued" if pending == {"created"} else "running"

    for outcome in ("indeterminate", "timed_out", "failed", "cancelled"):
        if outcome in states:
            return msgspec.convert(outcome, TaskState)

    return "succeeded"


def _reason(task: Task) -> str:
    errors = [execution.error.message for execution in task.executions if execution.error]
    return errors[0] if errors else f"task {task.id} is {task.state}"


def _cause(task: Task) -> dict[str, Any]:
    """What the execution said, carried up with the exception.

    The remote traceback is the only part of a failure worth having, and it is
    written on the execution. An error that arrives without it tells the caller
    that something broke and nothing about where.
    """
    errors = [execution.error for execution in task.executions if execution.error]
    return dict(errors[0].details) if errors and errors[0].details else {}


async def _tasks(rows: Sequence[TaskRow]) -> tuple[Task, ...]:
    """Each task with its attempts, the attempts of all of them read together.

    One query per task made a page of two hundred cost two hundred and one
    statements. The ids go in batches only because each one is a bound
    parameter, and SQLite caps how many a statement takes.
    """
    attempts: defaultdict[str, list[ExecutionRow]] = defaultdict(list)
    for ids in batched([row.id for row in rows], BATCH):
        for attempt in await ExecutionRow.objects().where(ExecutionRow.task_id.is_in(list(ids))).order_by(ExecutionRow.ordinal):
            attempts[attempt.task_id].append(attempt)
    return tuple([await _to_task(row, attempts[row.id]) for row in rows])


async def _to_task(row: TaskRow, attempts: Sequence[ExecutionRow]) -> Task:
    return Task(
        id=row.id,
        compute_id=row.compute_id,
        generation=row.generation,
        function=row.function,
        args_sha256=row.args_sha256,
        dispatch=msgspec.convert(row.dispatch, Dispatch),
        state=msgspec.convert(row.state, TaskState),
        retry=row.decision,
        executions=tuple([await _to_execution(attempt) for attempt in attempts]),
        submitted_at=row.submitted_at,
        rank=row.rank,
        correlation_id=row.correlation_id,
        queue_timeout_seconds=row.queue_timeout,
        run_timeout_seconds=row.run_timeout,
        result_sha256=row.result_sha256,
        finished_at=row.finished_at,
    )


def _limit(asked: float | None, default: float) -> float | None:
    """The limit a task runs under: its own when it named one, the compute's otherwise, and none at all for ``0``."""
    return (default if asked is None else asked) or None


def _cursor(order: TaskOrder, position: int, task: str) -> str:
    return base64.urlsafe_b64encode(msgspec.json.encode((order, position, task))).decode()


def _position(cursor: str, order: TaskOrder) -> tuple[int, str]:
    """Where a page picks up: the position and the task the page before it ended on."""
    try:
        paged, position, task = msgspec.json.decode(base64.urlsafe_b64decode(cursor), type=tuple[TaskOrder, int, str])
    except (ValueError, msgspec.DecodeError) as exc:
        raise NotFoundError(f"no such cursor: {cursor}") from exc
    if paged != order:
        raise NotFoundError(f"cursor {cursor} pages the {paged} order, not {order}")
    return position, task


async def _to_execution(row: ExecutionRow) -> Execution:
    return Execution(
        id=row.id,
        rank=row.rank,
        ordinal=row.ordinal,
        state=msgspec.convert(row.state, ExecutionState),
        node_id=row.node_id,
        retry_of=row.retry_of,
        result_sha256=row.result_sha256,
        error=await unpacked(row.error, Error) if row.error else None,
        started_at=row.started_at,
        finished_at=row.finished_at,
        deadline_at=row.deadline_at,
        stopping=row.stopping,
    )
