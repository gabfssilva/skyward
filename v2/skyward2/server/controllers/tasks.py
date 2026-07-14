from __future__ import annotations

from collections.abc import AsyncIterator

from litestar import Controller, MediaType, Response, delete, get, post
from litestar.params import Parameter
from litestar.response import Stream

from skyward2.application import ports
from skyward2.application.reconciler import Wakeup
from skyward2.protocol.schemas import (
    Execution,
    ExecutionCreate,
    Page,
    Task,
    TaskCreate,
    TaskState,
)

BLOB = "application/vnd.skyward.blob"
FRAMES = "application/vnd.skyward.frames"


async def framed(frames: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Each frame, with its length in front of it.

    A stream of msgpack values needs a boundary, and the transport does not supply
    one: HTTP chunks are how the network felt like splitting the bytes, not how the
    generator yielded them.
    """
    async for frame in frames:
        yield len(frame).to_bytes(4, "big") + frame


class TaskController(Controller):
    path = "/tasks"
    tags = ["tasks"]

    @get(summary="List tasks")
    async def list(
        self,
        tasks: ports.Tasks,
        cursor: str | None = None,
        limit: int = 50,
        compute: str | None = None,
        task_state: TaskState | None = Parameter(query="state", default=None),
        correlation_id: str | None = Parameter(default=None, description="Groups the tasks of an `&`/`gather`/`map`. A field, not a resource."),
    ) -> Page[Task]:
        return await tasks.list(cursor, limit, compute, task_state, correlation_id)

    @post(
        status_code=201,
        summary="Submit a task",
        description=(
            "A task is **one call**: `function` + `args` → `result`. It is what the SDK's `Future[T]` points at, and its "
            "`id` survives retries and SDK restarts.\n\n"
            "`dispatch: one` (the `>>` operator) creates a single execution. `dispatch: all` (the `@` operator) freezes "
            "the set of `ready` nodes at admission and creates one execution per rank — a later scale-up does not add "
            "executions.\n\n"
            "The task and its first execution are persisted **before** any dispatch to a worker. The worker dedupes on "
            "`(task_id, execution_id, args_hash)`."
        ),
    )
    async def submit(
        self,
        data: TaskCreate,
        tasks: ports.Tasks,
        wake: Wakeup,
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Response[Task]:
        task, created = await tasks.submit(data, idempotency_key)
        wake("task.changed", task_id=task.id)
        return Response(task, status_code=201 if created else 200)

    @get(
        "/{task_id:str}",
        summary="Read a task",
        description="`state` is **derived** from the executions — never written alongside them.",
    )
    async def read(self, task_id: str, tasks: ports.Tasks) -> Task:
        return await tasks.get(task_id)

    @delete(
        "/{task_id:str}",
        status_code=202,
        summary="Cancel a task",
        description=(
            "Before `started`, cancellation is guaranteed (CAS + worker ack). After `started` it is best-effort: the "
            "execution enters `cancel_requested` and only becomes `cancelled` once the worker confirms it stopped. "
            "Without that confirmation the outcome is `indeterminate` — Python code that may still be running is never "
            "declared cancelled."
        ),
    )
    async def cancel(
        self,
        task_id: str,
        tasks: ports.Tasks,
        wake: Wakeup,
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Response[Task]:
        task = await tasks.cancel(task_id, idempotency_key)
        wake("task.changed", task_id=task.id)
        return Response(task, status_code=202)

    @get(
        "/{task_id:str}/result",
        media_type=BLOB,
        summary="Download the result",
        description=(
            "The result survives controller and SDK restarts, and reading does not consume it — only an explicit purge "
            "removes it.\n\n"
            "`204` means there is no terminal outcome yet. A non-successful terminal outcome answers `409` with the "
            "error code (`task_failed`, `task_indeterminate`, ...)."
        ),
    )
    async def result(self, task_id: str, tasks: ports.Tasks, wait: int = 0) -> Response[bytes | None]:
        """A body, or nothing at all.

        The empty answer carries no media type, because there is no media: a 204 with
        a content type is a promise of a body that is not there, and Litestar is right
        to refuse to render it. The caller asks again — a task outlives any one poll,
        and running out of patience is not an outcome.
        """
        if (blob := await tasks.result(task_id, wait)) is None:
            return Response(b"", status_code=204, media_type=MediaType.TEXT)

        return Response(blob, status_code=200, media_type=BLOB)

    @get(
        "/{task_id:str}/stream",
        media_type=FRAMES,
        summary="Read a streaming task",
        description=(
            "For a task submitted with `dispatch: stream` — a generator on the far side. **This request is the "
            "dispatch**: nothing starts a streaming task on its own, because the only process that can hold the "
            "far end of a stream is the one consuming it.\n\n"
            "Length-prefixed msgpack frames: 4 bytes big-endian, then the frame. A frame is an item or a failure, and "
            "a failure is the last one — by the time a generator raises, the caller already has the items before it, "
            "and there is no status code left to fail with.\n\n"
            "Not resumable. A dropped stream is a dead stream; submit another task."
        ),
    )
    async def stream(self, task_id: str, dispatcher: ports.Dispatcher) -> Stream:
        return Stream(framed(dispatcher.stream(task_id)), media_type=FRAMES)

    @get("/{task_id:str}/executions", summary="List the physical attempts")
    async def list_executions(self, task_id: str, executions: ports.Executions) -> Page[Execution]:
        return await executions.list(task_id)

    @get("/{task_id:str}/executions/{ordinal:int}", summary="Read one attempt")
    async def get_execution(self, task_id: str, ordinal: int, executions: ports.Executions) -> Execution:
        return await executions.get(task_id, ordinal)

    @post(
        "/{task_id:str}/executions",
        status_code=202,
        summary="Create an execution (retry)",
        description=(
            "Retrying **is** creating another physical attempt of the same task — the `task_id` stays the user's handle. "
            "The new execution points `retry_of` at the previous one.\n\n"
            "Retrying an `indeterminate` outcome requires `acknowledge_duplication: true`: the system does not know "
            "whether the previous execution had side effects, and will not pretend it does."
        ),
    )
    async def create_execution(
        self,
        task_id: str,
        data: ExecutionCreate,
        executions: ports.Executions,
        wake: Wakeup,
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Response[Task]:
        task = await executions.create(task_id, data, idempotency_key)
        wake("task.changed", task_id=task.id)
        return Response(task, status_code=202)
