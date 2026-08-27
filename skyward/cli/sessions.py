"""The session verbs — ``status``, ``sessions``, ``stop`` — at the top level.

v1 kept a session store next to the pool: a session was a named thing the
client remembered, and the verbs read it to know which pool you meant. v2 has
no such store, because a compute already is one — it has a name, a lifetime and
a revision, and the daemon holds all three.

So these are not a second concept. They are the ``sky compute`` commands under
the names you reach for when the question is "what is running", and every one
of them calls straight through to :mod:`skyward.cli.compute` rather than
repeating the request it would make.
"""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from skyward.cli import app
from skyward.cli._output import Output
from skyward.cli.compute import delete_compute, get_compute, list_computes


@app.command(name="status")
def status(
    ref: Annotated[str | None, Parameter(help="A compute id or name; omit for all of them")] = None,
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Show one compute, or every compute the daemon knows about."""
    if ref is None:
        list_computes(url=url, output=output)
        return
    get_compute(ref, url=url, output=output)


@app.command(name="sessions")
def sessions(
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """List every compute — ``status`` with no argument, under the older name."""
    list_computes(url=url, output=output)


@app.command(name="stop")
def stop(
    ref: Annotated[str, Parameter(help="A compute id or name")],
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Tear a compute down.

    Accepted, not done: what comes back is still ``deleting``, and stays that
    way until the provider confirms the machines are gone.
    """
    delete_compute(ref, url=url, output=output)


__all__ = ["sessions", "status", "stop"]
