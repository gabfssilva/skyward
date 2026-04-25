"""Skyward HTTP server — exposes a long-lived Session over HTTP.

The ``Client`` class is always available (uses only core deps).  The
server-side ``app`` requires the ``[server]`` extra (Starlette + uvicorn)
and is loaded lazily so importing ``skyward`` never pulls Starlette.

Run with: ``uvicorn skyward.server:app --port 7590``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skyward.server.client import Client

if TYPE_CHECKING:
    from starlette.applications import Starlette


def create_app() -> Starlette:
    from skyward.server.app import create_app as _create_app
    return _create_app()


def __getattr__(name: str) -> Any:
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Client", "app", "create_app"]
