"""Starlette app factory for the Skyward HTTP server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Route

import skyward as sky
from skyward.observability.logger import logger
from skyward.server import routes as routes
from skyward.server.state import ServerState


@asynccontextmanager
async def _lifespan(app: Starlette) -> AsyncIterator[None]:
    session = sky.Session(console=False)
    session.__enter__()
    state = ServerState(session=session)
    app.state.server_state = state
    logger.info("skyward server: session ready")
    try:
        yield
    finally:
        logger.info("skyward server: shutting down")
        for name, entry in list(state.pools.items()):
            if entry.task is not None and not entry.task.done():
                entry.task.cancel()
            state.pools.pop(name, None)
        for eid, fut in list(state.executions.items()):
            fut.cancel()
            state.executions.pop(eid, None)
        state.broadcast_executor.shutdown(wait=False, cancel_futures=True)
        try:
            session.__exit__(None, None, None)
        except Exception as e:
            logger.warning(f"error closing session: {e!r}")


def create_app() -> Starlette:
    return Starlette(
        lifespan=_lifespan,
        routes=[
            Route("/health", routes.health, methods=["GET"]),
            Route("/shutdown", routes.shutdown, methods=["POST"]),
            Route("/compute", routes.create_pool, methods=["POST"]),
            Route("/compute", routes.list_pools, methods=["GET"]),
            Route("/compute/{name}", routes.get_pool, methods=["GET"]),
            Route("/compute/{name}", routes.delete_pool, methods=["DELETE"]),
            Route(
                "/compute/{name}/executions",
                routes.submit_execution,
                methods=["POST"],
            ),
            Route(
                "/compute/{name}/executions/{id}",
                routes.get_execution,
                methods=["GET"],
            ),
            Route(
                "/compute/{name}/executions/{id}",
                routes.delete_execution,
                methods=["DELETE"],
            ),
            Route(
                "/compute/{name}/events",
                routes.stream_events,
                methods=["GET"],
            ),
        ],
    )
