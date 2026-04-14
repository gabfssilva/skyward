"""HTTP routes for :class:`Compute` resources.

Read handlers are thin adapters over :class:`Store` list/detail methods.
Write handlers (POST/DELETE) are a Phase D stub: they directly mutate the
store, intentionally violating the single-writer invariant. Phase G2 will
delete the direct ``put_compute`` calls here and replace them with
``PoolHost.ensure_compute`` / ``PoolHost.shutdown_compute``; the pool actor
becomes the sole writer for the ``compute`` row at that point.
"""

from __future__ import annotations

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

    Phase D stub: writes directly to the store. Replaced in G2 by
    ``PoolHost.ensure_compute``.

    Returns
    -------
    Response
        ``202`` with the persisted compute on success, ``409`` if the
        name already exists, or ``422`` on invalid body.
    """
    store = request.app.state.store
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=422)
    name = body.get("name") if isinstance(body, dict) else None
    spec_raw = body.get("spec") if isinstance(body, dict) else None
    if not isinstance(name, str) or not isinstance(spec_raw, dict):
        return JSONResponse({"error": "missing_fields"}, status_code=422)
    if (await store.get_compute(name)) is not None:
        return JSONResponse({"error": "exists", "name": name}, status_code=409)
    try:
        spec = from_dict(spec_raw, ComputeSpec)
    except (ValueError, TypeError, KeyError):
        return JSONResponse({"error": "invalid_spec"}, status_code=422)
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

    Phase D stub: writes directly to the store. Replaced in G2 by
    ``PoolHost.shutdown_compute``. Terminal states (``stopped``/``failed``)
    return ``409 already_terminal`` rather than being treated as a no-op so
    callers can distinguish "I shut it down" from "it was already gone".
    """
    store = request.app.state.store
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
            now = datetime.now(UTC)
            started = getattr(status, "started_at", now)
            stopping = Stopping(started_at=started, stopping_since=now)
            new = replace(existing, status=stopping)
            await store.put_compute(new)
            return JSONResponse(to_dict(new), status_code=202)


routes: list[Route] = [
    Route("/v1/compute", _list, methods=["GET"]),
    Route("/v1/compute", _create, methods=["POST"]),
    Route("/v1/compute/{name}", _detail, methods=["GET"]),
    Route("/v1/compute/{name}", _delete, methods=["DELETE"]),
]


__all__ = ["encode", "routes"]
