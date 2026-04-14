"""HTTP routes for :class:`Provider` resources.

Exposes ``GET /v1/providers`` (list, ordered by name),
``GET /v1/providers/{name}`` (detail, 404 on miss), and
``PUT /v1/providers/{name}`` (upsert). The write handler is a Phase D
stub: it writes directly to the store, mirroring the compute-write
stubs. ``Provider.config`` is decoded via the ``ProviderConfig``
discriminated union so unknown tags are rejected with ``422``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from skyward.api.provider import ProviderConfig
from skyward.server.host.domain import Provider
from skyward.server.wire import from_dict, to_dict


async def _list(request: Request) -> Response:
    store = request.app.state.store
    providers = await store.list_providers()
    return JSONResponse([to_dict(p) for p in providers])


async def _detail(request: Request) -> Response:
    store = request.app.state.store
    name = request.path_params["name"]
    p = await store.get_provider(name)
    if p is None:
        return JSONResponse(
            {"error": "not_found", "resource": "provider", "name": name},
            status_code=404,
        )
    return JSONResponse(to_dict(p))


async def _put(request: Request) -> Response:
    """Insert or replace a :class:`Provider` row.

    Phase D stub: writes directly to the store. Expects a JSON body of the
    form ``{"type": "<tag>", "config": {"type": "<tag>", ...}}`` where the
    inner tag matches a registered :class:`ProviderConfig` subclass.

    Returns
    -------
    Response
        ``200`` with the persisted provider on success, ``422`` when the
        body is malformed or the config tag is unknown.
    """
    store = request.app.state.store
    name = request.path_params["name"]
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=422)
    if not isinstance(body, dict):
        return JSONResponse({"error": "invalid_json"}, status_code=422)
    ptype = body.get("type")
    config_raw = body.get("config")
    if not isinstance(ptype, str) or not isinstance(config_raw, dict):
        return JSONResponse({"error": "missing_fields"}, status_code=422)
    try:
        config = from_dict(config_raw, ProviderConfig)
    except (ValueError, TypeError, KeyError):
        return JSONResponse({"error": "invalid_config"}, status_code=422)
    now = datetime.now(UTC)
    existing = await store.get_provider(name)
    created_at = existing.created_at if existing is not None else now
    provider = Provider(
        name=name,
        type=ptype,  # type: ignore[arg-type]
        config=config,
        created_at=created_at,
        updated_at=now,
        last_used_at=existing.last_used_at if existing is not None else None,
    )
    await store.put_provider(provider)
    return JSONResponse(to_dict(provider), status_code=200)


routes: list[Route] = [
    Route("/v1/providers", _list, methods=["GET"]),
    Route("/v1/providers/{name}", _detail, methods=["GET"]),
    Route("/v1/providers/{name}", _put, methods=["PUT"]),
]


__all__ = ["routes"]
