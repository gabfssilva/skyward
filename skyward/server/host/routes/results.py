"""HTTP read route for :class:`TaskResult` rows scoped by execution.

Exposes ``GET /v1/executions/{id}/results``. An unknown execution id
returns an empty list — the Store filters purely by foreign key.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from skyward.server.wire import to_dict


async def _list(request: Request) -> Response:
    store = request.app.state.store
    exec_id = request.path_params["id"]
    results = await store.list_results(execution=exec_id)
    return JSONResponse([to_dict(r) for r in results])


routes: list[Route] = [
    Route("/v1/executions/{id}/results", _list, methods=["GET"]),
]


__all__ = ["routes"]
