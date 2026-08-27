"""How a command reaches the control plane.

The SDK's client is async and the CLI is not, so this is the whole bridge:
one coroutine, run to completion, against a client the command never opened.

Which daemon it reaches is resolution, not choice — an explicit ``--url``, else
``SKYWARD_URL``, else the address ``sky server start`` binds. There is always a
daemon at the end of it: the CLI owns nothing and outlives no command, so it has
no business being a control plane, and a second one over a database some daemon
already owns is two reconcilers buying the same machine.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable

import httpx

from skyward.core.client import Client

type Work[T] = Callable[[Client], Awaitable[T]]

HOST = "127.0.0.1"
PORT = 17590
DEFAULT_URL = f"http://{HOST}:{PORT}"
"""Where ``sky server start`` binds, and so where a command looks by default."""


def resolve(url: str | None) -> str:
    """Return the daemon's URL.

    Parameters
    ----------
    url
        The value of ``--url``, if the command was given one.
    """
    return (url or os.environ.get("SKYWARD_URL") or DEFAULT_URL).rstrip("/")


def call[T](work: Work[T], *, url: str | None = None) -> T:
    """Run ``work`` against the resolved daemon and return its result.

    Parameters
    ----------
    work
        Receives the open client and does the calls.
    url
        Overrides ``SKYWARD_URL``.
    """
    return asyncio.run(_run(work, resolve(url)))


async def _run[T](work: Work[T], url: str) -> T:
    client = await Client.remote(url)
    try:
        return await work(client)
    except httpx.ConnectError:
        raise SystemExit(f"no daemon at {url} — run: sky server start") from None
    finally:
        await client.close()


__all__ = ["DEFAULT_URL", "HOST", "PORT", "Work", "call", "resolve"]
