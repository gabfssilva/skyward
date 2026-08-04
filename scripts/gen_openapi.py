"""Generate ``docs/openapi.json`` from the Litestar application.

The spec is what ``docs/http-api.md`` renders, so it is documentation rather than
a build artifact and lives under ``docs/`` where mkdocs serves it. It is derived
entirely from the controllers and their annotations — the schema Litestar already
builds for ``/v1/schema/openapi.json``, taken in process so no daemon has to be
running.

Run with ``uv run python scripts/gen_openapi.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from litestar.serialization import encode_json

from skyward.server.http.app import create_app, services

SPEC_PATH = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"


def render() -> str:
    """The whole daemon's surface, not the part a mock happens to carry.

    Built from :func:`services` rather than the default mocks: forwarding, shell
    and file transfer are registered only when the service behind them exists, so
    an app built without them publishes a spec that is missing three controllers.
    Nothing here opens the store — ``create_app`` connects on startup, and this
    never starts.
    """
    schema = json.loads(encode_json(create_app(services()).openapi_schema.to_schema()))
    return json.dumps(schema, indent=2, sort_keys=True)


def main() -> None:
    spec = render()
    SPEC_PATH.write_text(spec)
    paths = json.loads(spec)["paths"]
    operations = sum(len(methods) for methods in paths.values())
    print(f"wrote {len(paths)} paths, {operations} operations to {SPEC_PATH}")


if __name__ == "__main__":
    main()
