"""The browser console, served by the daemon it drives.

The console is a single-page app: every route it draws — ``/computes/<id>``, ``/market`` — is the same
``index.html``, and the browser reads the path. So anything outside ``/assets`` answers with that page. The
API is not shadowed by it: a path under ``/v1`` that no controller has is still a 404, because the router
never falls back from a prefix it already matched.

The page is sent with ``no-cache``, since the files it names are renamed on every build and a stale copy
would ask for assets that no longer exist.
"""

from pathlib import Path

from litestar import Router, get
from litestar.datastructures import CacheControlHeader
from litestar.enums import MediaType
from litestar.response import File
from litestar.static_files import create_static_files_router


def console(directory: Path) -> Router:
    """The routes that serve the console built into ``directory``."""

    @get(["/", "/{route:path}"], include_in_schema=False, sync_to_thread=False, cache_control=CacheControlHeader(no_cache=True))
    def page() -> File:
        return File(directory / "index.html", filename="index.html", media_type=MediaType.HTML, content_disposition_type="inline")

    return Router(path="/", route_handlers=[create_static_files_router(path="/assets", directories=[directory / "assets"]), page])
