"""HTTP read route for nodes attached to a compute.

Exposes ``GET /v1/compute/{compute_name}/nodes`` listing every node row
owned by the named compute. Listing a non-existent compute is not an
error — the Store returns an empty list, and so does this route.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from skyward.server.wire import to_dict


async def _list(request: Request) -> Response:
    store = request.app.state.store
    compute = request.path_params["compute_name"]
    nodes = await store.list_nodes(compute=compute)
    return JSONResponse([to_dict(n) for n in nodes])


routes: list[Route] = [
    Route("/v1/compute/{compute_name}/nodes", _list, methods=["GET"]),
]


__all__ = ["routes"]
