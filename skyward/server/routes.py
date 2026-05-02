"""HTTP route handlers for the Skyward server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import TYPE_CHECKING, Any

from starlette.responses import JSONResponse, Response, StreamingResponse

from skyward.observability.logger import logger
from skyward.server.wire import decode, encode, event_to_json, pool_view_to_json

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from concurrent.futures import Future

    from starlette.requests import Request

    from skyward.server.state import PoolEntry, ServerState


def _state(request: Request) -> ServerState:
    return request.app.state.server_state


def _pool_info(entry: PoolEntry) -> dict:
    pool = entry.pool
    if pool is not None:
        current_nodes = pool.current_nodes()
        concurrency = pool.concurrency
        is_active = pool.is_active
    else:
        current_nodes = 0
        concurrency = 0
        is_active = False
    return {
        "name": entry.name,
        "status": entry.status,
        "current_nodes": current_nodes,
        "concurrency": concurrency,
        "is_active": is_active,
        "error": entry.error,
    }


async def create_pool(request: Request) -> Response:
    state = _state(request)
    name = request.query_params.get("name") or f"pool-{uuid.uuid4().hex[:8]}"
    body = await request.body()

    try:
        specs, options = decode(body)
    except Exception as e:
        return JSONResponse({"error": f"invalid payload: {e!r}"}, status_code=400)

    if name in state.pools:
        return JSONResponse({"error": f"pool {name!r} already exists"}, status_code=409)

    async def _provision() -> None:
        try:
            pool = await asyncio.to_thread(
                state.session.compute, *specs, name=name, options=options,
            )
        except asyncio.CancelledError:
            logger.info(f"pool {name!r} provisioning cancelled")
            raise
        except Exception as exc:
            logger.warning(f"pool {name!r} provisioning failed: {exc!r}")
            state.set_failed(name, repr(exc))
            return
        state.set_ready(name, pool)
        logger.info(f"pool {name!r} ready")

    task = asyncio.create_task(_provision(), name=f"create-pool:{name}")
    entry = state.register_creating(name, task)
    return JSONResponse(_pool_info(entry), status_code=202)


async def list_pools(request: Request) -> Response:
    state = _state(request)
    return JSONResponse([_pool_info(e) for e in state.pools.values()])


async def get_pool(request: Request) -> Response:
    state = _state(request)
    name = request.path_params["name"]
    entry = state.get_pool(name)
    if entry is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(_pool_info(entry))


async def delete_pool(request: Request) -> Response:
    state = _state(request)
    name = request.path_params["name"]
    entry = state.get_pool(name)
    if entry is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    state.set_stopping(name)

    # Signal the underlying pool actor to terminate, whether it's still
    # provisioning (ref tracked in session._pending_pool_refs) or already
    # ready (in session._pools). This actually tears down cloud instances —
    # in contrast to merely cancelling the asyncio task, which leaves the
    # provisioning thread running and leaks resources.
    try:
        await asyncio.to_thread(state.session.stop_pool, name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"pool {name!r} stop_pool raised: {exc!r}")

    # If a creation task is still in flight, wait briefly for it to wind
    # down (it should fail quickly now that the actor was signalled).
    if entry.task is not None and not entry.task.done():
        with contextlib.suppress(TimeoutError, asyncio.CancelledError, Exception):
            await asyncio.wait_for(asyncio.shield(entry.task), timeout=10.0)

    state.drop_pool(name)
    return Response(status_code=204)


async def submit_execution(request: Request) -> Response:
    state = _state(request)
    name = request.path_params["name"]
    mode = request.query_params.get("mode", "run")
    if mode not in ("run", "broadcast"):
        return JSONResponse({"error": f"invalid mode: {mode!r}"}, status_code=400)

    entry = state.get_pool(name)
    if entry is None:
        return JSONResponse({"error": "pool not found"}, status_code=404)
    if entry.status != "ready" or entry.pool is None:
        return JSONResponse(
            {"error": "pool not ready", "status": entry.status},
            status_code=409,
        )

    pool = entry.pool
    body = await request.body()
    try:
        pending = decode(body)
    except Exception as e:
        return JSONResponse({"error": f"invalid payload: {e!r}"}, status_code=400)

    future: Future
    match mode:
        case "run":
            future = pool.run_async(pending)
        case _:
            future = state.broadcast_executor.submit(pool.broadcast, pending)

    eid = state.register_execution(future)
    location = f"/compute/{name}/executions/{eid}"
    return JSONResponse(
        {"id": eid},
        status_code=202,
        headers={"Location": location},
    )


async def get_execution(request: Request) -> Response:
    state = _state(request)
    eid = request.path_params["id"]
    future = state.get_execution(eid)
    if future is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    if not future.done():
        return JSONResponse({"status": "pending"}, status_code=202)

    state.drop_execution(eid)
    exc = future.exception()
    if exc is not None:
        return Response(
            encode(exc),
            status_code=500,
            media_type="application/octet-stream",
            headers={"X-Skyward-Error": "1"},
        )
    return Response(
        encode(future.result()),
        status_code=200,
        media_type="application/octet-stream",
    )


async def delete_execution(request: Request) -> Response:
    state = _state(request)
    eid = request.path_params["id"]
    future = state.drop_execution(eid)
    if future is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    future.cancel()
    return Response(status_code=204)


async def health(request: Request) -> Response:
    from skyward import __version__

    state = _state(request)
    return JSONResponse({
        "status": "ok",
        "version": __version__,
        "pools": len(state.pools),
        "executions": len(state.executions),
    })


async def shutdown(request: Request) -> Response:
    import os
    import signal

    del request
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=202)


# ── SSE event stream ─────────────────────────────────────────────


def _sse_frame(event_type: str, payload: Any) -> bytes:
    """Format a Server-Sent Event frame."""
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


async def stream_events(request: Request) -> Response:
    """Stream pool/node lifecycle events as text/event-stream.

    Initial frame is ``event: snapshot`` with the current ``PoolView``
    (or ``null`` if the pool exists but no events have flowed yet). Each
    subsequent domain event with matching ``pool_name`` is delivered as
    a JSON SSE frame. The stream closes with ``event: done`` once the
    pool reaches ``failed``/``stopping`` or the client disconnects.
    """
    state = _state(request)
    name = request.path_params["name"]
    if state.get_pool(name) is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1024)

    def _enqueue(event: Any) -> None:
        if getattr(event, "pool_name", None) != name:
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, event)

    projection = state.session.projection
    unsubscribe = projection.subscribe(on_event=_enqueue, on_log=_enqueue)

    async def gen() -> AsyncIterator[bytes]:
        try:
            view = projection.view.pools.get(name)
            yield _sse_frame("snapshot", pool_view_to_json(view) if view else None)

            while True:
                if await request.is_disconnected():
                    return

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield b": keepalive\n\n"
                    entry = state.get_pool(name)
                    if entry is None:
                        yield _sse_frame("done", {"status": "deleted"})
                        return
                    continue

                payload = event_to_json(event)
                yield _sse_frame(payload["type"], payload)

                entry = state.get_pool(name)
                if entry is None:
                    yield _sse_frame("done", {"status": "deleted"})
                    return
                if entry.status in ("failed", "stopping"):
                    yield _sse_frame("done", {"status": entry.status, "error": entry.error})
                    return
        finally:
            try:
                unsubscribe()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"sse unsubscribe failed: {exc!r}")

    return StreamingResponse(gen(), media_type="text/event-stream")
