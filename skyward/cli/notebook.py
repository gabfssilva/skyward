"""sky notebook — install or remove the Skyward remote-kernel for a session."""

from __future__ import annotations

from typing import Annotated

import httpx
from cyclopts import Parameter

from . import app
from ._output import console
from ._session_store import resolve_session


@app.command
def notebook(
    action: Annotated[str, Parameter(help="install | remove")] = "install",
    *,
    session: Annotated[str | None, Parameter(name="--session", help="Session/pool name (defaults to the current session)")] = None,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Install (or remove) the Skyward remote-kernel for a session.

    ``sky notebook install`` resolves the target session, then writes and
    registers a Jupyter kernelspec bound to it. Open your own Jupyter and
    pick the ``Skyward (<session>)`` kernel. ``sky notebook remove`` deletes
    the kernelspec.

    Parameters
    ----------
    action
        ``install`` (default) or ``remove``.
    session
        Session/pool name; defaults to the current session.
    url
        Server URL override.
    """
    from skyward.notebook.kernelspec import install_kernelspec, remove_kernelspec

    try:
        name = resolve_session(session, url)
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {url}[/red]")
        raise SystemExit(1) from None

    match action:
        case "install":
            kernel = install_kernelspec(name)
            console.print(
                f"[green]Installed kernel[/green] [bold]{kernel}[/bold]. "
                f'Open Jupyter and pick "Skyward ({name})".'
            )
        case "remove":
            remove_kernelspec(name)
            console.print(f"[green]Removed kernel[/green] [bold]skyward-{name}[/bold].")
        case _:
            console.print(f"[red]Unknown action '{action}'. Use install or remove.[/red]")
            raise SystemExit(1)
