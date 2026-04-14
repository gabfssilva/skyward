"""HTTP read route for task-scoped executions.

Exposes ``GET /v1/tasks/{module}/{qualname}/executions`` which lists every
task execution whose ``(module, qualname)`` key matches. Unknown tasks
return an empty list — consistent with the Store's filter-based query.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from skyward.server.wire import to_dict


async def _list_executions(request: Request) -> Response:
    store = request.app.state.store
    module = request.path_params["module"]
    qualname = request.path_params["qualname"]
    executions = await store.list_executions(task=(module, qualname))
    return JSONResponse([to_dict(e) for e in executions])


routes: list[Route] = [
    Route(
        "/v1/tasks/{module}/{qualname}/executions",
        _list_executions,
        methods=["GET"],
    ),
]


__all__ = ["routes"]
