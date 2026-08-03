"""Watch an existing compute through the Rich footer or the line log."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from skyward.cli import app
from skyward.cli._client import call
from skyward.core.client import Client
from skyward.core.console import ConsoleMode, watcher
from skyward.core.errors import SkywardError
from skyward.shared.schemas import Compute


@app.command(name="monitor")
def monitor(
    ref: Annotated[str, Parameter(help="A compute id or name")],
    *,
    mode: Annotated[ConsoleMode, Parameter(help="rich or log")] = "rich",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
) -> None:
    """Watch a live compute until interrupted.

    Parameters
    ----------
    ref
        The compute to attach to, by id or by name. It has to exist already —
        monitoring creates nothing.
    mode
        ``rich`` for the live footer or ``log`` for plain lines.
    url
        Overrides ``SKYWARD_URL``.
    database
        Where the embedded daemon keeps its state. Ignored when a URL resolves.
    """

    async def work(client: Client) -> None:
        compute = await client.call("GET", f"/v1/computes/{ref}", Compute)
        follower = await asyncio.to_thread(watcher, client, compute.id, mode=mode)
        await follower.follow()

    try:
        call(work, url=url, database=database)
    except SkywardError as error:
        raise SystemExit(f"{error.code}: {error.message}") from None
    except KeyboardInterrupt:
        return


__all__ = ["monitor"]
