"""``sky monitor`` — the pool's own dashboard, on a compute you did not start.

The SDK already renders a compute while it holds one: :func:`skyward.core.console.watcher`
picks the richest view the terminal can hold and follows the event stream with it.
This is that, detached — the same watcher over the same stream, for a compute this
process never created and will not delete.

So there is no second dashboard, and no second decision about whether to draw one:
the watcher probes the terminal and falls back to the line console on its own, and
a command that repeated the probe would only be able to get it wrong.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from skyward.cli import app
from skyward.cli._client import call
from skyward.core.client import Client
from skyward.core.console import watcher
from skyward.core.errors import SkywardError
from skyward.shared.schemas import Compute


@app.command(name="monitor")
def monitor(
    ref: Annotated[str, Parameter(help="A compute id or name")],
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
) -> None:
    """Watch a live compute until interrupted.

    Parameters
    ----------
    ref
        The compute to attach to, by id or by name. It has to exist already —
        monitoring creates nothing.
    url
        Overrides ``SKYWARD_URL``.
    database
        Where the embedded daemon keeps its state. Ignored when a URL resolves.
    """

    async def work(client: Client) -> None:
        compute = await client.call("GET", f"/v1/computes/{ref}", Compute)
        follower = await asyncio.to_thread(watcher, client, compute.id)
        await follower.follow()

    try:
        call(work, url=url, database=database)
    except SkywardError as error:
        raise SystemExit(f"{error.code}: {error.message}") from None
    except KeyboardInterrupt:
        return


__all__ = ["monitor"]
