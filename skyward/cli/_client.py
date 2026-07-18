"""How a command reaches the control plane.

The SDK's client is async and the CLI is not, so this is the whole bridge:
one coroutine, run to completion, against a client the command never opened.
Which client it gets is resolution, not choice — an explicit ``--url``, else
``SKYWARD_URL``, else the daemon running in this process against the default
database. A command that reads computes reads them the same way in all three.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from skyward.sdk.client import Client

type Work[T] = Callable[[Client], Awaitable[T]]


def resolve(url: str | None) -> str | None:
    """Return the daemon's URL, or None when there is no daemon to dial.

    Parameters
    ----------
    url
        The value of ``--url``, if the command was given one.
    """
    if url:
        return url.rstrip("/")
    if env := os.environ.get("SKYWARD_URL"):
        return env.rstrip("/")
    return None


def call[T](work: Work[T], *, url: str | None = None, database: Path | None = None) -> T:
    """Run ``work`` against a client and return its result.

    Parameters
    ----------
    work
        Receives the open client and does the calls.
    url
        Overrides ``SKYWARD_URL``.
    database
        Where the embedded daemon keeps its state. Ignored when a URL resolves.
    """
    return asyncio.run(_run(work, url, database))


async def _run[T](work: Work[T], url: str | None, database: Path | None) -> T:
    if target := resolve(url):
        client = await Client.remote(target)
    else:
        from skyward.persistence.db import DEFAULT_PATH

        client = await Client.embedded(database or DEFAULT_PATH)

    try:
        return await work(client)
    finally:
        await client.close()


__all__ = ["Work", "call", "resolve"]
