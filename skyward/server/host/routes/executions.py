"""HTTP read and write routes for :class:`TaskExecution` resources.

Exposes ``GET /v1/executions`` (list, with ``compute``/``status``/``task``/
``group`` filters), ``GET /v1/executions/{id}`` (detail, 404 on miss), and
``POST /v1/compute/{name}/tasks`` (enqueue a task execution with an
octet-stream payload).

The ``task`` query parameter uses ``module:qualname`` syntax. Missing or
malformed values yield a 400 ``{"error": "bad_task_key"}`` response.

The POST handler is a Phase D stub that writes ``task_executions`` directly;
Phase G3 moves the insert behind the task-manager actor, leaving the payload
upload and ``Task`` upsert on the HTTP boundary.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from skyward.server.host.domain import (
    Broadcast,
    Queued,
    Run,
    Task,
    TaskExecution,
    TaskExecutionKind,
    TaskKey,
)
from skyward.server.host.pool_host import PoolHost
from skyward.server.wire import to_dict


def _parse_task(raw: str | None) -> TaskKey | None | Response:
    match raw:
        case None:
            return None
        case value if ":" in value:
            module, qualname = value.split(":", 1)
            return (module, qualname)
        case _:
            return JSONResponse({"error": "bad_task_key"}, status_code=400)


async def _list(request: Request) -> Response:
    store = request.app.state.store
    compute = request.query_params.get("compute")
    status = request.query_params.get("status")
    group = request.query_params.get("group")
    task_parsed = _parse_task(request.query_params.get("task"))
    if isinstance(task_parsed, Response):
        return task_parsed
    executions = await store.list_executions(
        compute=compute,
        status=status,
        task=task_parsed,
        group=group,
    )
    return JSONResponse([to_dict(e) for e in executions])


async def _detail(request: Request) -> Response:
    store = request.app.state.store
    exec_id = request.path_params["id"]
    e = await store.get_execution(exec_id)
    if e is None:
        return JSONResponse(
            {"error": "not_found", "resource": "execution", "name": exec_id},
            status_code=404,
        )
    return JSONResponse(to_dict(e))


def _pack_broadcast(shards: list[bytes]) -> bytes:
    """Encode broadcast replies as ``[u32][shard0][u32][shard1]...``.

    Matches the wire shape the client driver unpacks in Phase H2; the
    length prefix is big-endian to stay consistent with the rest of
    the host protocol framing.
    """
    import struct

    return b"".join(struct.pack(">I", len(s)) + s for s in shards)


def _parse_kind(raw: str) -> TaskExecutionKind | None:
    match raw.lower():
        case "run":
            return Run()
        case "broadcast":
            return Broadcast()
        case _:
            return None


async def _create(request: Request) -> Response:
    """Enqueue a task execution for ``/v1/compute/{name}/tasks``.

    When a :class:`PoolHost` is attached, the route hands the bytes to
    :meth:`PoolHost.submit_task` and returns the pickled result as
    ``application/octet-stream`` (G3 contract). Otherwise the request
    falls through to the D-phase stub that merely persists a
    ``Queued`` row — kept alive so the D3 contract tests stay green.
    """
    store = request.app.state.store
    blobs = request.app.state.blobs
    pool_host: PoolHost | None = getattr(request.app.state, "pool_host", None)
    compute_name = request.path_params["name"]

    if (await store.get_compute(compute_name)) is None:
        return JSONResponse(
            {"error": "not_found", "resource": "compute", "name": compute_name},
            status_code=404,
        )

    module = request.headers.get("x-task-module")
    qualname = request.headers.get("x-task-qualname")
    if not module or not qualname:
        return JSONResponse(
            {"error": "missing_task_headers"}, status_code=422
        )

    try:
        timeout_s = float(request.headers.get("x-timeout", "60"))
    except ValueError:
        return JSONResponse({"error": "invalid_timeout"}, status_code=422)

    kind = _parse_kind(request.headers.get("x-kind") or "run")
    if kind is None:
        return JSONResponse({"error": "invalid_kind"}, status_code=422)

    client_id = request.headers.get("x-client-id")
    payload = await request.body()

    if pool_host is not None:
        try:
            match kind:
                case Broadcast():
                    execution_id, shards = await pool_host.broadcast(
                        compute_name, (module, qualname), payload,
                        timeout_s, client_id,
                    )
                    return Response(
                        content=_pack_broadcast(shards),
                        media_type="application/octet-stream",
                        headers={"X-Execution-Id": execution_id},
                    )
                case Run():
                    execution_id, result_bytes = await pool_host.submit_task(
                        compute_name, (module, qualname), payload,
                        timeout_s, client_id,
                    )
                    return Response(
                        content=result_bytes,
                        media_type="application/octet-stream",
                        headers={"X-Execution-Id": execution_id},
                    )
        except TimeoutError:
            return JSONResponse({"error": "timeout"}, status_code=504)
        except LookupError as exc:
            return JSONResponse(
                {"error": "not_found", "reason": str(exc)}, status_code=404,
            )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                {"error": "execution_failed", "reason": str(exc)},
                status_code=500,
            )

    blob_id = await blobs.put(payload, kind="payload")
    await store.put_task(Task(module=module, qualname=qualname))
    execution_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    await store.put_execution(
        TaskExecution(
            id=execution_id,
            task=(module, qualname),
            compute=compute_name,
            kind=kind,
            payload=blob_id,
            timeout=timedelta(seconds=timeout_s),
            client=client_id,
            submitted_at=now,
            status=Queued(),
        )
    )
    return JSONResponse({"id": execution_id}, status_code=202)


routes: list[Route] = [
    Route("/v1/executions", _list, methods=["GET"]),
    Route("/v1/executions/{id}", _detail, methods=["GET"]),
    Route("/v1/compute/{name}/tasks", _create, methods=["POST"]),
]


__all__ = ["routes"]
