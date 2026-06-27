"""HTTP route handlers for the Skyward server."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.responses import JSONResponse, Response, StreamingResponse

from skyward.observability.logger import logger
from skyward.server import reattach
from skyward.server.wire import decode, encode, event_to_json, pool_view_to_json

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from concurrent.futures import Future

    from starlette.requests import Request

    from skyward.actors.messages import NodeFileResult, NodeSelection, NodeTarget
    from skyward.actors.snapshot import NodeSnapshot, PoolSnapshot
    from skyward.core.model import Cluster, Instance
    from skyward.core.pool import ComputePool
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
        provider_config = specs[0].provider if specs else None
        with contextlib.suppress(Exception):
            await asyncio.to_thread(reattach.persist_handle, name, provider_config, pool)
            reattach.subscribe_repersist(state, name, provider_config, pool)
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


def _node_info(
    ns: NodeSnapshot, inst: Instance | None, cluster: Cluster | None,
) -> dict:
    """Build per-node SSH addressing JSON from a live snapshot.

    No password value is returned; only ``has_password`` signals
    password-only auth so the client can refuse interactive mode.
    """
    ssh_key_path = cluster.ssh_key_path if cluster is not None else None
    return {
        "rank": ns.node_id,
        "status": ns.status.name.lower(),
        "is_head": ns.node_id == 0,
        "instance_id": ns.instance_id or None,
        "ip": inst.ip if inst is not None else None,
        "private_ip": inst.private_ip if inst is not None else None,
        "ssh_port": inst.ssh_port if inst is not None else None,
        "ssh_user": cluster.ssh_user if cluster is not None else None,
        "ssh_key_path": ssh_key_path,
        "has_password": inst.ssh_password is not None if inst is not None else False,
        "key_exists_on_server": bool(ssh_key_path) and Path(ssh_key_path).exists(),
    }


async def list_nodes(request: Request) -> Response:
    """Return per-node SSH addressing for a ready session as rank-ordered JSON.

    404 if unknown; 409 (with ``status``) if not ready. Each node:
    ``{rank, status, is_head, instance_id, ip, private_ip, ssh_port,
    ssh_user, ssh_key_path, has_password, key_exists_on_server}``.
    """
    state = _state(request)
    name = request.path_params["name"]
    entry = state.get_pool(name)
    if entry is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    if entry.status != "ready" or entry.pool is None:
        return JSONResponse(
            {"error": "pool not ready", "status": entry.status},
            status_code=409,
        )

    pool = entry.pool
    snap: PoolSnapshot = await asyncio.to_thread(pool.snapshot)
    instances = {i.id: i for i in snap.instances}
    nodes = [
        _node_info(ns, instances.get(ns.instance_id), snap.cluster)
        for ns in sorted(snap.nodes, key=lambda n: n.node_id)
    ]
    return JSONResponse({"name": name, "nodes": nodes})


async def get_pool_log(request: Request) -> Response:
    """Return recorded event history for a pool.

    Body: ``{"name", "count", "events": [...], "sources": {eid: source}}``.
    404 if no history exists. Reads ``history`` independent of live pool
    existence, so deleted pools still export until server restart.
    """
    state = _state(request)
    name = request.path_params["name"]
    if not state.history.has(name):
        return JSONResponse({"error": "no history"}, status_code=404)
    limit_q = request.query_params.get("limit")
    limit = int(limit_q) if limit_q and limit_q.isdigit() else None
    events = state.history.get(name, limit)
    return JSONResponse({
        "name": name,
        "count": len(events),
        "events": events,
        "sources": state.history.sources(name),
    })


def _ready_pool(state: ServerState, name: str) -> tuple[ComputePool | None, Response | None]:
    """Resolve a ready pool (with file-op facades) or the matching error response."""
    from skyward.core.pool import ComputePool

    entry = state.get_pool(name)
    if entry is None:
        return None, JSONResponse({"error": "not found"}, status_code=404)
    if entry.status != "ready" or entry.pool is None:
        return None, JSONResponse(
            {"error": "pool not ready", "status": entry.status}, status_code=409,
        )
    if not isinstance(entry.pool, ComputePool):
        return None, JSONResponse(
            {"error": "pool does not support file operations"}, status_code=500,
        )
    return entry.pool, None


def _parse_node(value: str | None, default: NodeSelection, *, allow_all: bool = True) -> NodeSelection:
    """Parse the ``?node=`` query into a ``NodeSelection`` (raises on invalid)."""
    if value is None:
        return default
    if value == "head":
        return "head"
    if value == "all":
        if not allow_all:
            raise ValueError("node=all not supported here")
        return "all"
    if value.lstrip("-").isdigit():
        return int(value)
    raise ValueError(f"invalid node: {value!r}")


def _file_result_json(r: NodeFileResult) -> dict:
    return {"node_id": r.node_id, "success": r.success, "listing": r.listing, "error": r.error}


async def list_files(request: Request) -> Response:
    """``ls -la`` ``?path=`` on the selected node(s) (default head)."""
    state = _state(request)
    pool, err = _ready_pool(state, request.path_params["name"])
    if err is not None:
        return err
    assert pool is not None
    path = request.query_params.get("path")
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    try:
        sel = _parse_node(request.query_params.get("node"), "head")
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    results = await asyncio.to_thread(pool.ls, path, sel)
    return JSONResponse({"results": [_file_result_json(r) for r in results]})


async def remove_file(request: Request) -> Response:
    """``rm -rf`` ``?path=`` on the selected node(s) (default all)."""
    state = _state(request)
    pool, err = _ready_pool(state, request.path_params["name"])
    if err is not None:
        return err
    assert pool is not None
    path = request.query_params.get("path")
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    try:
        sel = _parse_node(request.query_params.get("node"), "all")
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    results = await asyncio.to_thread(pool.rm, path, sel)
    return JSONResponse({"results": [_file_result_json(r) for r in results]})


async def upload_file(request: Request) -> Response:
    """Write the octet-stream body to ``?path=`` on the selected node(s) (default all)."""
    state = _state(request)
    pool, err = _ready_pool(state, request.path_params["name"])
    if err is not None:
        return err
    assert pool is not None
    path = request.query_params.get("path")
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    try:
        sel = _parse_node(request.query_params.get("node"), "all")
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    content = await request.body()
    results = await asyncio.to_thread(pool.upload_file, content, path, sel)
    return JSONResponse({"results": [_file_result_json(r) for r in results]})


async def download_file(request: Request) -> Response:
    """Stream ``?path=`` from a single node as octet-stream (``node=all`` → 422)."""
    state = _state(request)
    pool, err = _ready_pool(state, request.path_params["name"])
    if err is not None:
        return err
    assert pool is not None
    path = request.query_params.get("path")
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    if request.query_params.get("node") == "all":
        return JSONResponse({"error": "download does not support node=all"}, status_code=422)
    try:
        sel = _parse_node(request.query_params.get("node"), "head", allow_all=False)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    results = await asyncio.to_thread(pool.download_file, path, sel)
    if not results:
        return JSONResponse({"error": "no such node"}, status_code=404)
    first = results[0]
    if not first.success:
        return JSONResponse({"error": first.error or "download failed"}, status_code=404)
    return Response(first.content, media_type="application/octet-stream")


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
    reattach.drop_persistence(state, name)
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

    eid = uuid.uuid4().hex
    source_b64 = request.headers.get("X-Skyward-Source")
    if source_b64:
        with contextlib.suppress(Exception):
            state.history.set_source(
                name, eid, base64.b64decode(source_b64).decode("utf-8"),
            )
    future: Future
    match mode:
        case "run":
            node = request.query_params.get("node")
            target: NodeTarget | None = None
            if node == "head":
                target = "head"
            elif node is not None:
                if not node.lstrip("-").isdigit():
                    return JSONResponse({"error": f"invalid node: {node!r}"}, status_code=400)
                target = int(node)
            future = pool.run_async(pending, task_id=eid, target=target)
        case _:
            future = state.broadcast_executor.submit(
                pool.broadcast, pending, task_id=eid,
            )

    state.register_execution(eid, future)
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
    # ``on_event`` already fires for every dispatched event including
    # ``Log.Emitted`` (see ``SessionProjection.handle``). Subscribing
    # ``on_log`` too would enqueue every log twice.
    unsubscribe = projection.subscribe(on_event=_enqueue)

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
