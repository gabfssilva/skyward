"""Entry point for ``python -m skyward.server.host`` — binds Starlette to UDS or TCP."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import uvicorn

from skyward.server.host.app import create_app
from skyward.server.host.store import Store

DEFAULT_DB_PATH: str = str(Path.home() / ".skyward" / "state.db")


def main() -> None:
    """Parse CLI flags and run the host server.

    Bind to a Unix domain socket via ``--socket`` or to a TCP address via
    ``--host``/``--port``.  Exactly one of ``--socket`` or ``--port`` must
    be provided.
    """
    parser = argparse.ArgumentParser(prog="skyward.server.host")
    parser.add_argument("--socket", help="UDS path to bind")
    parser.add_argument("--host", default="127.0.0.1", help="TCP bind host (with --port)")
    parser.add_argument("--port", type=int, help="TCP bind port")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite state path")
    args = parser.parse_args()

    if not args.socket and args.port is None:
        parser.error("either --socket or --port is required")
    if args.socket and args.port is not None:
        parser.error("--socket and --port are mutually exclusive")

    db_path = args.db
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    store = Store(path=db_path)
    asyncio.run(store.open())
    app = create_app(store)

    if args.socket:
        uvicorn.run(app, uds=args.socket)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
