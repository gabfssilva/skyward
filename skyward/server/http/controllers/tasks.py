from __future__ import annotations

from collections.abc import AsyncIterator

import msgspec
from litestar import Controller, MediaType, Response, delete, get, post
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.response import Stream

from skyward.api import v1
from skyward.server.application import ports
from skyward.server.application.reading import Reader
from skyward.server.application.reconciler import Wakeup
from skyward.server.http import representation
from skyward.server.http.exceptions import failures
from skyward.server.http.references import narrowed
from skyward.server.http.representation import recast
from skyward.shared import codec
from skyward.shared.schemas import ExecutionCreate, TaskCreate

BLOB = "application/vnd.skyward.blob"
FRAMES = "application/vnd.skyward.frames"


async def pickled(data: TaskCreate) -> TaskCreate:
    """The arguments as everything below takes them, however the caller said them.

    A ``call`` is JSON because the caller had no interpreter to pickle with. What
    is written down is the same inline pickle every other caller sends, so the
    store, the dispatcher and the worker never learn there were two ways to say
    it — the second way ends here.
    """
    if data.call is None:
        return data

    arguments = (tuple(data.call.args or ()), dict(data.call.kwargs or {}))
    return msgspec.structs.replace(data, call=None, args_inline=await codec.payload.encode(arguments))


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
    dependencies = {"compute_id": Provide(narrowed)}

    @get(
        summary="List tasks",
        description=(
            "Every call this daemon has been asked to make, a page at a time. `correlation_id` is how the tasks of one "
            "`&`, `gather` or `map` are found together — it is a field on each of them, not a resource of its own.\n\n"
            "`function` is a name and `lineage` is a function: two functions called `train` in two different files share "
            "the name and not the lineage, and a function edited and sent again keeps the lineage under a new digest.\n\n"
            "`order` is `submitted` (newest first), `state` (running, newest submitted first; then queued, oldest "
            "submitted first; then finished, latest to finish first) or `finished` (latest to finish first, then the "
            "unfinished, newest submitted first). `total` counts what every filter matches, and `next_cursor` pages "
            "the order it came from, from where the page ended: a task submitted or moved mid-walk does not shift it."
        ),
    )
    async def list(
        self,
        reader: Reader,
        compute_id: str | None,
        cursor: str | None = None,
        limit: int = Parameter(default=50, ge=1),
        task_states: list[v1.TaskState] | None = Parameter(query="state", default=None, description="Any of these; repeat it for more than one."),
        correlation_id: str | None = Parameter(default=None, description="Groups the tasks of an `&`/`gather`/`map`. A field, not a resource."),
        function: str | None = Parameter(default=None, description="A function's name, which takes in every upload of its code."),
        lineage: str | None = Parameter(default=None, description="One function, by the `lineage` its uploads share."),
        order: v1.TaskOrder = "submitted",
    ) -> v1.Page[v1.TaskResource]:
        page = await reader.tasks(cursor, limit, compute_id, tuple(task_states or ()), correlation_id, function, lineage, order)
        return v1.Page(items=tuple(representation.task(each) for each in page.items), next_cursor=page.next_cursor, total=page.total)

    @post(
        status_code=201,
        summary="Submit a task",
        description=(
            "A task is **one call**: `function` + `args` → `result`. It is what the SDK's `Future[T]` points at, and its "
            "`id` survives retries and SDK restarts.\n\n"
            "`dispatch: one` (the `>>` operator) creates a single execution. `dispatch: all` (the `@` operator) freezes "
            "the set of `ready` nodes at admission and creates one execution per rank — a later scale-up does not add "
            "executions.\n\n"
            "The arguments are a pickle: `args_inline` under a size threshold, `args_sha256` over it. A caller with no "
            "Python to pickle with sends `call` instead — the values in JSON — and the daemon encodes them here.\n\n"
            "The task and its first execution are persisted **before** any dispatch to a worker. The worker dedupes on "
            "`(task_id, execution_id, args_hash)`."
        ),
        responses={
            **failures(404, 409, 422),
            200: ResponseSpec(v1.TaskResource, description="The task this `Idempotency-Key` already created"),
        },
    )
    async def submit(
        self,
        data: v1.CreateTaskResource,
        tasks: ports.Tasks,
        reader: Reader,
        wake: Wakeup,
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Response[v1.TaskResource]:
        task, created = await tasks.submit(await pickled(recast(data, TaskCreate)), idempotency_key)
        wake("task.changed", task_id=task.id)
        wake("compute.changed", compute_id=task.compute_id)
        return Response(representation.task(await reader.task(task.id)), status_code=201 if created else 200)

    @get(
        "/{task_id:str}",
        summary="Read a task",
        description="`state` is **derived** from the executions — never written alongside them.",
        responses=failures(404),
    )
    async def read(self, task_id: str, reader: Reader) -> v1.TaskResource:
        return representation.task(await reader.task(task_id))

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
        responses=failures(404, 409),
    )
    async def cancel(
        self,
        task_id: str,
        tasks: ports.Tasks,
        reader: Reader,
        wake: Wakeup,
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Response[v1.TaskResource]:
        task = await tasks.cancel(task_id, idempotency_key)
        wake("task.changed", task_id=task.id)
        return Response(representation.task(await reader.task(task.id)), status_code=202)

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
        responses={
            200: ResponseSpec(
                bytes,
                media_type=BLOB,
                description="The return value, in the codec the function was written with",
                generate_examples=False,
            ),
            **failures(404, 409),
            204: ResponseSpec(None, description="No terminal outcome yet — ask again"),
        },
    )
    async def result(self, task_id: str, tasks: ports.Tasks, wait: int = 0) -> Response[bytes]:
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
            "The body is that framing, so it has no JSON schema: OpenAPI can describe a stream's bytes and not the "
            "boundaries a reader has to cut them on.\n\n"
            "Not resumable. A dropped stream is a dead stream; submit another task."
        ),
        responses={
            200: ResponseSpec(
                bytes,
                media_type=FRAMES,
                description="Length-prefixed msgpack frames, one per item, until the generator ends or fails",
                generate_examples=False,
            ),
            **failures(404, 422),
        },
    )
    async def stream(self, task_id: str, dispatcher: ports.Dispatcher) -> Stream:
        return Stream(framed(dispatcher.stream(task_id)), media_type=FRAMES)

    @get(
        "/{task_id:str}/executions",
        summary="List the physical attempts",
        description="One row per attempt, `ordinal`-counted. A retry appears here; it never appears as another task.",
        responses=failures(404),
    )
    async def list_executions(self, task_id: str, executions: ports.Executions) -> v1.Page[v1.ExecutionResource]:
        return recast(await executions.list(task_id), v1.Page[v1.ExecutionResource])

    @get(
        "/{task_id:str}/executions/{ordinal:int}",
        summary="Read one attempt",
        description="One attempt by its ordinal, including which node took it and what it ended as.",
        responses=failures(404),
    )
    async def get_execution(self, task_id: str, ordinal: int, executions: ports.Executions) -> v1.ExecutionResource:
        return recast(await executions.get(task_id, ordinal), v1.ExecutionResource)

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
        responses=failures(404, 409),
    )
    async def create_execution(
        self,
        task_id: str,
        data: v1.CreateExecutionResource,
        executions: ports.Executions,
        reader: Reader,
        wake: Wakeup,
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Response[v1.TaskResource]:
        task = await executions.create(task_id, recast(data, ExecutionCreate), idempotency_key)
        wake("task.changed", task_id=task.id)
        return Response(representation.task(await reader.task(task.id)), status_code=202)
