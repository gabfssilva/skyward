"""HTTP route modules for the Skyward host server.

Each module in this package exports a ``routes: list[Route]`` sequence
that :func:`skyward.server.host.app.create_app` mounts onto the Starlette
application.
"""

from __future__ import annotations

__all__: list[str] = []
