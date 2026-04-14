"""HTTP routes for blob metadata and binary content.

Exposes ``GET /v1/blobs/{id}/meta`` returning the JSON row for a blob and
``GET /v1/blobs/{id}`` streaming the raw bytes as ``application/octet-stream``.
Evicted rows return ``410 Gone`` with ``{"error": "evicted", "at": <iso>}``.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from skyward.server.host.domain import Blob


def _encode_blob(b: Blob) -> dict[str, Any]:
    return {
        "id": b.id,
        "path": str(b.path),
        "size": b.size,
        "sha256": b.sha256,
        "kind": b.kind,
        "created_at": b.created_at.isoformat(),
        "evicted_at": None if b.evicted_at is None else b.evicted_at.isoformat(),
    }


async def _meta(request: Request) -> Response:
    store = request.app.state.store
    raw = request.path_params["id"]
    try:
        blob_id = int(raw)
    except ValueError:
        return JSONResponse(
            {"error": "not_found", "resource": "blob", "name": raw},
            status_code=404,
        )
    blob = await store.get_blob(blob_id)
    if blob is None:
        return JSONResponse(
            {"error": "not_found", "resource": "blob", "name": raw},
            status_code=404,
        )
    return JSONResponse(_encode_blob(blob))


async def _stream(request: Request) -> Response:
    store = request.app.state.store
    blobs = request.app.state.blobs
    raw = request.path_params["id"]
    try:
        blob_id = int(raw)
    except ValueError:
        return JSONResponse(
            {"error": "not_found", "resource": "blob", "id": raw},
            status_code=404,
        )
    row = await store.get_blob(blob_id)
    match row:
        case None:
            return JSONResponse(
                {"error": "not_found", "resource": "blob", "id": blob_id},
                status_code=404,
            )
        case blob if blob.evicted_at is not None:
            return JSONResponse(
                {"error": "evicted", "at": blob.evicted_at.isoformat()},
                status_code=410,
            )
        case _:
            data = await blobs.read(blob_id)
            return Response(content=data, media_type="application/octet-stream")


routes: list[Route] = [
    Route("/v1/blobs/{id}", _stream, methods=["GET"]),
    Route("/v1/blobs/{id}/meta", _meta, methods=["GET"]),
]


__all__ = ["routes"]
