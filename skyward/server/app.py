"""Starlette app factory for the Skyward HTTP server."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Route

import skyward as sky
from skyward.observability.logger import logger
from skyward.server import reattach
from skyward.server import routes as routes
from skyward.server.history import attach_history
from skyward.server.state import ServerState


@asynccontextmanager
async def _lifespan(app: Starlette) -> AsyncIterator[None]:
    import os

    log_file = os.environ.get("SKYWARD_SERVER_LOG")
    logging = sky.LogConfig(level="DEBUG", file=log_file) if log_file else False
    session = sky.Session(console=False, logging=logging)
    session.__enter__()
    state = ServerState(session=session)
    app.state.server_state = state
    unsubscribe_history = attach_history(session.projection, state.history)
    with contextlib.suppress(Exception):
        await asyncio.to_thread(reattach.reattach_pools, state)
    logger.info("skyward server: session ready")
    try:
        yield
    finally:
        logger.info("skyward server: shutting down")
        with contextlib.suppress(Exception):
            unsubscribe_history()
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
            Route("/compute/{name}/nodes", routes.list_nodes, methods=["GET"]),
            Route("/compute/{name}/log", routes.get_pool_log, methods=["GET"]),
            Route("/compute/{name}/files", routes.list_files, methods=["GET"]),
            Route("/compute/{name}/files", routes.remove_file, methods=["DELETE"]),
            Route("/compute/{name}/files", routes.upload_file, methods=["PUT"]),
            Route("/compute/{name}/files/content", routes.download_file, methods=["GET"]),
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
