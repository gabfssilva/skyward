from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Any

import msgspec
from msgspec import UNSET

from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.store import after, ident, now, once, packed, unpacked
from skyward.server.persistence.tables import ExecutionRow, TaskRow
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
    RetryPolicy,
    Task,
    TaskCreate,
    TaskState,
)

ACCEPTING: tuple[ComputeState, ...] = ("requested", "provisioning", "ready", "degraded")

PENDING: tuple[ExecutionState, ...] = ("created", "assigned", "dispatching", "accepted", "started", "cancel_requested")


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

        A task that named no timeout takes the compute's, and takes it here: the
        deadline is a fact about the task, and a task is only admitted once.
        """

        async def insert() -> str:
            compute = await self._computes.get(body.compute)
            if compute.status.state not in ACCEPTING:
                raise ComputeNotAcceptingError(f"compute {compute.id} is {compute.status.state}")

            args = body.args_sha256 or await self._blobs.store(body.args_inline or b"")
            if not await self._blobs.exists(args):
                raise NotFoundError(f"no such args blob: {args}")

            task = ident("tsk")
            retry = body.retry if body.retry is not UNSET else compute.spec.retry
            timeout = body.timeout_seconds or compute.spec.options.default_compute_timeout
            await TaskRow(
                id=task,
                compute_id=compute.id,
                generation=compute.generation,
                function=body.function,
                args_sha256=args,
                dispatch=body.dispatch,
                state="queued",
                retry=await packed(retry),
                correlation_id=body.correlation_id,
                submitted_at=now(),
                deadline_at=now() + timedelta(seconds=timeout) if timeout else None,
            ).save().run()

            for rank in await self._ranks(compute.id, body):
                await self.attempt(task, rank=rank, ordinal=1, retry_of=None)

            return task

        task_id, created = await once("task.submit", idempotency_key, body, insert)
        return await self.get(task_id), created

    async def get(self, task_id: str) -> Task:
        return await _to_task(await self._row(task_id))

    async def list(self, cursor: str | None, limit: int, compute: str | None, state: TaskState | None, correlation_id: str | None) -> Page[Task]:
        query = TaskRow.objects()

        if pivot := await after(TaskRow, cursor):
            query = query.where(TaskRow.submitted_at > pivot)
        if compute:
            query = query.where(TaskRow.compute_id == compute)
        if state:
            query = query.where(TaskRow.state == state)
        if correlation_id:
            query = query.where(TaskRow.correlation_id == correlation_id)

        rows = await query.order_by(TaskRow.submitted_at).limit(limit)
        items = tuple([await _to_task(row) for row in rows])
        return Page(items=items, next_cursor=items[-1].id if len(items) == limit else None)

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
    ) -> None:
        """What a worker did with an attempt, and the outcome that follows from it."""
        row = await ExecutionRow.objects().where(ExecutionRow.id == execution_id).first()
        if row is None:
            raise NotFoundError(f"no such execution: {execution_id}")

        row.state = state
        row.node_id = node_id or row.node_id
        row.result_sha256 = result_sha256 or row.result_sha256
        row.error = await packed(error) if error else row.error
        if state == "started" and row.started_at is None:
            row.started_at = now()
        if state not in PENDING:
            row.finished_at = now()
        await row.save().run()

        await self.settle(row.task_id)

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
        execution = ident("exe")
        await ExecutionRow(
            id=execution,
            task_id=task_id,
            rank=rank,
            ordinal=ordinal,
            state="created",
            retry_of=retry_of,
        ).save().run()
        return execution

    async def attempts(self, task_id: str) -> list[ExecutionRow]:
        return await ExecutionRow.objects().where(ExecutionRow.task_id == task_id).order_by(ExecutionRow.ordinal)

    async def unsettled(self) -> tuple[str, ...]:
        """Tasks that have not reached a verdict — what the sweep re-offers."""
        rows = await TaskRow.select(TaskRow.id).where(TaskRow.state.is_in(["queued", "running"]))
        return tuple(row["id"] for row in rows)

    async def expire(self) -> tuple[str, ...]:
        """Time out the tasks whose deadline has passed.

        The deadline is written at admission and nothing else reads it, so this is what
        makes it mean anything. It runs on the tick because a deadline passing is not
        something anybody does: there is no write to react to, and the only way to
        notice is to look.

        Every attempt still in flight goes, the ones that never left included — a
        caller who asked for an answer within a window is no better served by an
        attempt still waiting for a machine than by one that is running.
        """
        rows = await TaskRow.select(TaskRow.id).where(
            TaskRow.state.is_in(["queued", "running"]) & (TaskRow.deadline_at < now()),
        )
        expired = tuple(row["id"] for row in rows)

        for task_id in expired:
            for execution in await self.attempts(task_id):
                if execution.state in PENDING:
                    await self.observe(execution.id, "timed_out")

        return expired

    async def load(self, compute: str) -> int:
        """How many attempts this compute still owes an answer for.

        Queued and running together, because they are the same demand seen a moment
        apart: sizing the pool to what is running would size it to what the pool can
        already do, and a queue would never be a reason to grow.
        """
        return len(await self._pending(compute))

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
        pending = await self._pending(compute)
        return (
            Counter(row["node_id"] for row in pending if row["node_id"]),
            frozenset(row["rank"] for row in pending),
        )

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

    Once they are all in, the worst one wins, and ``indeterminate`` is the worst.
    It is the only outcome that forbids an automatic retry, so a broadcast in which
    one node cleanly failed and another is unaccounted for must not be quietly
    repeated on the strength of the clean failure.
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


async def _to_task(row: TaskRow) -> Task:
    attempts = await ExecutionRow.objects().where(ExecutionRow.task_id == row.id).order_by(ExecutionRow.ordinal)
    return Task(
        id=row.id,
        compute_id=row.compute_id,
        generation=row.generation,
        function=row.function,
        args_sha256=row.args_sha256,
        dispatch=msgspec.convert(row.dispatch, Dispatch),
        state=msgspec.convert(row.state, TaskState),
        retry=await unpacked(row.retry, RetryPolicy),
        executions=tuple([await _to_execution(attempt) for attempt in attempts]),
        submitted_at=row.submitted_at,
        correlation_id=row.correlation_id,
        deadline_at=row.deadline_at,
        result_sha256=row.result_sha256,
        finished_at=row.finished_at,
    )


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
    )
