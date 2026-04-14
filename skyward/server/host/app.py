"""Starlette HTTP application for the Skyward host server.

Exposes ``/v1/health`` and ``/v1/info`` and enforces the
``X-Skyward-Api: 1`` protocol version header. Additional routes are mounted
by later phases (compute, node, task, event streaming).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp

import skyward
from skyward.server.host.blobs import Blobs
from skyward.server.host.routes import (
    blobs as blobs_routes,
)
from skyward.server.host.routes import (
    compute as compute_routes,
)
from skyward.server.host.routes import (
    executions as executions_routes,
)
from skyward.server.host.routes import (
    nodes as nodes_routes,
)
from skyward.server.host.routes import (
    providers as providers_routes,
)
from skyward.server.host.routes import (
    results as results_routes,
)
from skyward.server.host.routes import (
    tasks as tasks_routes,
)
from skyward.server.host.store import Store

API_VERSION: int = 1


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def _info(_request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "python_version": sys.version,
            "api_version": API_VERSION,
            "skyward_version": skyward.__version__,
            "pid": os.getpid(),
        }
    )


class _ApiVersionMiddleware(BaseHTTPMiddleware):
    """Reject requests whose ``X-Skyward-Api`` header does not match the server."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        raw = request.headers.get("x-skyward-api")
        got = _parse_version(raw)
        if got == API_VERSION:
            return await call_next(request)
        return JSONResponse(
            {"error": "api_version_mismatch", "required": API_VERSION, "got": got},
            status_code=409,
        )


def _parse_version(raw: str | None) -> int | None:
    match raw:
        case None:
            return None
        case value:
            try:
                return int(value)
            except ValueError:
                return None


def create_app(
    store: Store,
    pool_host: Any | None = None,
    blobs: Blobs | None = None,
) -> Starlette:
    """Build the Starlette app with health and info routes and version middleware.

    Parameters
    ----------
    store
        Opened :class:`Store` providing persistence for later route phases.
    pool_host
        Reserved for Phase G, which wires in the pool-host actor bridge.
    blobs
        :class:`Blobs` service used by the binary blob streaming route. When
        omitted, a default instance rooted at a temporary directory is created.

    Returns
    -------
    Starlette
        Configured application with ``store``, ``pool_host``, and ``blobs``
        available on ``app.state``.
    """
    routes = [
        Route("/v1/health", _health, methods=["GET"]),
        Route("/v1/info", _info, methods=["GET"]),
        *compute_routes.routes,
        *nodes_routes.routes,
        *providers_routes.routes,
        *tasks_routes.routes,
        *executions_routes.routes,
        *results_routes.routes,
        *blobs_routes.routes,
    ]
    app = Starlette(routes=routes, middleware=[Middleware(_ApiVersionMiddleware)])
    app.state.store = store
    app.state.pool_host = pool_host
    app.state.blobs = blobs if blobs is not None else Blobs(
        store=store,
        root=Path(tempfile.gettempdir()) / "skyward-server-blobs",
    )
    return app


__all__ = ["API_VERSION", "create_app"]
