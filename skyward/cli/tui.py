"""``sky app``: the fleet on one screen."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from skyward.cli import app
from skyward.cli._client import call, resolve
from skyward.core.client import Client


@app.command(name="app")
def dashboard(*, url: Annotated[str | None, Parameter(help="Daemon URL")] = None) -> None:
    """Watch every live compute on one screen, until ``q``.

    Parameters
    ----------
    url
        Overrides ``SKYWARD_URL``.
    """
    try:
        from skyward.core.tui import Dashboard
    except ModuleNotFoundError as missing:
        raise SystemExit("sky app needs: pip install 'skyward[tui]'") from missing

    async def work(client: Client) -> None:
        await Dashboard(client, resolve(url)).run_async()

    call(work, url=url)


__all__ = ["dashboard"]
