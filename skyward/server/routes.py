"""HTTP route handlers for the Skyward server."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, cast

from starlette.responses import JSONResponse, Response

from skyward.server.wire import decode, encode

if TYPE_CHECKING:
    from concurrent.futures import Future

    from starlette.requests import Request

    from skyward.api.pool import Pool
    from skyward.server.state import ServerState


def _state(request: Request) -> ServerState:
    return request.app.state.server_state


def _pool_info(name: str, pool: Pool) -> dict:
    return {
        "name": name,
        "current_nodes": pool.current_nodes(),
        "concurrency": pool.concurrency,
        "is_active": pool.is_active,
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

    try:
        pool = await asyncio.to_thread(
            state.session.compute, *specs, name=name, options=options,
        )
    except Exception as e:
        return JSONResponse({"error": repr(e)}, status_code=500)

    state.register_pool(name, pool)
    return JSONResponse(_pool_info(name, pool), status_code=201)


async def list_pools(request: Request) -> Response:
    state = _state(request)
    return JSONResponse([_pool_info(n, p) for n, p in state.pools.items()])


async def get_pool(request: Request) -> Response:
    state = _state(request)
    name = request.path_params["name"]
    pool = state.get_pool(name)
    if pool is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(_pool_info(name, pool))


async def delete_pool(request: Request) -> Response:
    state = _state(request)
    name = request.path_params["name"]
    pool = state.drop_pool(name)
    if pool is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    await asyncio.to_thread(cast(Any, pool).__exit__, None, None, None)
    return Response(status_code=204)


async def submit_execution(request: Request) -> Response:
    state = _state(request)
    name = request.path_params["name"]
    mode = request.query_params.get("mode", "run")
    if mode not in ("run", "broadcast"):
        return JSONResponse({"error": f"invalid mode: {mode!r}"}, status_code=400)

    pool = state.get_pool(name)
    if pool is None:
        return JSONResponse({"error": "pool not found"}, status_code=404)

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
