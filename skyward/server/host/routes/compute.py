"""HTTP routes for :class:`Compute` resources.

Read handlers are thin adapters over :class:`Store` list/detail methods.
Write handlers (POST/DELETE) are a Phase D stub: they directly mutate the
store, intentionally violating the single-writer invariant. Phase G2 will
delete the direct ``put_compute`` calls here and replace them with
``PoolHost.ensure_compute`` / ``PoolHost.shutdown_compute``; the pool actor
becomes the sole writer for the ``compute`` row at that point.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from skyward.server.host.domain import (
    Compute,
    ComputeSpec,
    Failed,
    Provisioning,
    Ready,
    Stopped,
    Stopping,
)
from skyward.server.wire import from_dict, to_dict


def encode(obj: Any) -> JSONResponse:
    """Encode a domain object through the wire codec as a JSON response."""
    return JSONResponse(to_dict(obj))


async def _list(request: Request) -> Response:
    store = request.app.state.store
    status = request.query_params.get("status")
    computes = await store.list_compute(status=status)
    return JSONResponse([to_dict(c) for c in computes])


async def _detail(request: Request) -> Response:
    store = request.app.state.store
    name = request.path_params["name"]
    c = await store.get_compute(name)
    if c is None:
        return JSONResponse(
            {"error": "not_found", "resource": "compute", "name": name},
            status_code=404,
        )
    return encode(c)


async def _create(request: Request) -> Response:
    """Create a new :class:`Compute` in the ``provisioning`` state.

    When a :class:`PoolHost` is attached to ``app.state``, the route
    hands off to :meth:`PoolHost.ensure_compute` so the pool actor is
    the single writer for the ``compute`` row (the G-era invariant).
    Without a host — the early-phase D stub — the row is written
    directly; this path keeps the D2 contract tests green while the
    full HTTP ↔ actor bridge is being assembled.

    Returns
    -------
    Response
        ``202`` with the persisted compute on success, ``409`` if the
        name already exists, or ``422`` on invalid body.
    """
    store = request.app.state.store
    pool_host = getattr(request.app.state, "pool_host", None)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=422)
    name = body.get("name") if isinstance(body, dict) else None
    spec_raw = body.get("spec") if isinstance(body, dict) else None
    if not isinstance(name, str) or not isinstance(spec_raw, dict):
        return JSONResponse({"error": "missing_fields"}, status_code=422)
    existing = await store.get_compute(name)
    if existing is not None:
        match existing.status:
            case Ready() if pool_host is not None and name in pool_host.session._pools:
                return JSONResponse(to_dict(existing), status_code=200)
            case Stopped() | Failed():
                pass
            case _:
                return JSONResponse(
                    {"error": "exists", "name": name}, status_code=409,
                )
    try:
        spec = from_dict(spec_raw, ComputeSpec)
    except (ValueError, TypeError, KeyError):
        return JSONResponse({"error": "invalid_spec"}, status_code=422)
    if pool_host is not None:
        try:
            await pool_host.ensure_compute(name, spec)
        except Exception as exc:  # noqa: BLE001 - surface a 500 to the client
            return JSONResponse(
                {"error": "provision_failed", "reason": str(exc)},
                status_code=500,
            )
        persisted = await store.get_compute(name)
        if persisted is None:
            return JSONResponse({"error": "race"}, status_code=500)
        return JSONResponse(to_dict(persisted), status_code=202)
    at = datetime.now(UTC)
    compute = Compute(
        name=name,
        spec=spec,
        created_at=at,
        status=Provisioning(started_at=at),
    )
    await store.put_compute(compute)
    return JSONResponse(to_dict(compute), status_code=202)


async def _delete(request: Request) -> Response:
    """Transition a :class:`Compute` to the ``stopping`` state.

    When a :class:`PoolHost` is attached the pool actor drives the
    transition via :meth:`PoolHost.shutdown_compute`. Absent a host,
    the store is updated directly — kept to preserve the D-phase
    contract tests until the full bridge is wired everywhere.
    Terminal states (``stopped``/``failed``) return ``409 already_terminal``
    rather than being treated as a no-op so callers can distinguish
    "I shut it down" from "it was already gone".
    """
    store = request.app.state.store
    pool_host = getattr(request.app.state, "pool_host", None)
    name = request.path_params["name"]
    existing = await store.get_compute(name)
    if existing is None:
        return JSONResponse(
            {"error": "not_found", "resource": "compute", "name": name},
            status_code=404,
        )
    match existing.status:
        case Stopped() | Failed():
            return JSONResponse(
                {"error": "already_terminal"}, status_code=409
            )
        case status:
            if pool_host is not None:
                await pool_host.shutdown_compute(name)
                persisted = await store.get_compute(name)
                if persisted is not None:
                    return JSONResponse(to_dict(persisted), status_code=202)
            now = datetime.now(UTC)
            started = getattr(status, "started_at", now)
            stopping = Stopping(started_at=started, stopping_since=now)
            new = replace(existing, status=stopping)
            await store.put_compute(new)
            return JSONResponse(to_dict(new), status_code=202)


async def _events(request: Request) -> Response:
    """Replay (and in future iterations live-stream) events as SSE.

    The current iteration only replays rows that already live in the
    events table up to ``?since=N``; live fan-out across the
    :class:`PoolHost` subscriber registry will land alongside actor
    event wiring.
    """
    from json import dumps

    from starlette.responses import StreamingResponse

    pool_host = getattr(request.app.state, "pool_host", None)
    name = request.path_params["name"]
    try:
        since = int(request.query_params.get("since", "0"))
    except ValueError:
        return JSONResponse({"error": "invalid_since"}, status_code=422)

    if pool_host is None:
        return JSONResponse({"error": "no_pool_host"}, status_code=503)
    try:
        aiter_ = pool_host.subscribe_events(name, since=since)
    except ValueError as exc:
        return JSONResponse(
            {"error": "invalid_name", "reason": str(exc)}, status_code=422,
        )

    async def _stream() -> AsyncIterator[bytes]:
        async for ev in aiter_:
            data = dumps({
                "id": ev.id,
                "aggregate": ev.aggregate,
                "type": ev.type,
                "payload": ev.payload,
                "ts": ev.ts,
            })
            yield f"id: {ev.id}\nevent: {ev.type}\ndata: {data}\n\n".encode()

    return StreamingResponse(_stream(), media_type="text/event-stream")


async def _session_events(request: Request) -> Response:
    """Stream pickled :class:`SessionEvent` bytes for ``compute:{name}``.

    Frames each event as ``data: <base64>\\n\\n`` so the client can
    unpickle and feed a local ``SessionProjection``. Live-only — no
    replay — because domain events are ephemeral UI state.
    """
    import asyncio
    import base64

    from starlette.responses import StreamingResponse

    pool_host = getattr(request.app.state, "pool_host", None)
    name = request.path_params["name"]
    if pool_host is None:
        return JSONResponse({"error": "no_pool_host"}, status_code=503)

    queue = pool_host.subscribe_session_events(name)

    async def _stream() -> AsyncIterator[bytes]:
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                encoded = base64.b64encode(data).decode("ascii")
                yield f"data: {encoded}\n\n".encode()
        finally:
            pool_host.unsubscribe_session_events(name, queue)

    return StreamingResponse(_stream(), media_type="text/event-stream")


routes: list[Route] = [
    Route("/v1/compute", _list, methods=["GET"]),
    Route("/v1/compute", _create, methods=["POST"]),
    Route("/v1/compute/{name}", _detail, methods=["GET"]),
    Route("/v1/compute/{name}", _delete, methods=["DELETE"]),
    Route("/v1/compute/{name}/events", _events, methods=["GET"]),
    Route(
        "/v1/compute/{name}/session-events", _session_events, methods=["GET"],
    ),
]


__all__ = ["encode", "routes"]
