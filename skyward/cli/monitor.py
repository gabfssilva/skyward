"""Watch computes from now on: one through the Rich footer, or any number of them, every one by default, as lines."""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated, get_args

from cyclopts import Parameter

from skyward.api.v1 import ComputeResource, LogEntryResource, Page
from skyward.cli import app
from skyward.cli._client import call
from skyward.core import console
from skyward.core.client import Client
from skyward.core.console import ConsoleMode, Observer, watcher
from skyward.core.errors import SkywardError
from skyward.core.view import decoded
from skyward.shared.events import ComputeDeleted, Event
from skyward.shared.schemas import NodeState

TYPES = (
    "compute.provisioning",
    "compute.ready",
    "compute.degraded",
    "compute.deleting",
    "compute.deletion_failed",
    "compute.deleted",
    *(f"node.{state}" for state in get_args(NodeState.__value__)),
    *(f"task.{state}" for state in ("succeeded", "failed", "cancelled", "timed_out", "indeterminate")),
)
"""The frames a line is drawn from: every change of state, and how each task ended. The gauges are left on the daemon."""

SAID = ("node.console", "node.progress")
"""What the nodes print, and how far a machine is from an address: the frames ``--quiet`` leaves out."""


@app.command(name="monitor")
def monitor(
    *refs: Annotated[str, Parameter(help="Compute ids or names; none watches every compute")],
    mode: Annotated[ConsoleMode, Parameter(help="rich or log")] = "rich",
    quiet: Annotated[bool, Parameter(name=("-q", "--quiet"), help="Lines for changes of state and task endings alone, not what the nodes print")] = False,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
) -> None:
    """Watch computes from now on, as they change.

    One compute, on a terminal, is drawn as the live footer a pool draws, until
    interrupted; ``--mode log`` asks for lines instead. Any other number of them —
    several, or none, which is every compute the daemon has — is lines on stdout,
    one per event worth one: a compute or a machine changing state, a line a node
    printed, an attempt at a task ending. Watching more than one, each line starts
    with the compute's name.

    Given computes, the lines end once every one of them is deleted; given none,
    they go on until interrupted. Nothing from before the command started is
    replayed: that is ``sky log``. ``--quiet`` keeps the lines to changes of state
    and task endings, which is what a reader of every compute can take in.

    Parameters
    ----------
    refs
        The computes to watch, by id or by name; they have to exist already.
    mode
        ``rich`` for the live footer, which one compute on a terminal gets, or ``log`` for lines.
    quiet
        Leave out of the lines what the nodes print and their progress.
    url
        Overrides ``SKYWARD_URL``.
    """

    async def work(client: Client) -> None:
        page = await client.call("GET", "/v1/events/log", Page[LogEntryResource], limit=1)
        head = page.items[0].sequence if page.items else None
        if len(refs) == 1 and mode == "rich" and not quiet and sys.stdout.isatty() and sys.stderr.isatty():
            compute = await client.call("GET", f"/v1/computes/{refs[0]}", ComputeResource)
            follower = await asyncio.to_thread(watcher, mode=mode)
            await Observer(client, compute.id, watchers=(follower,), after=head).follow()
        else:
            await _lines(client, refs, head, TYPES if quiet else (*TYPES, *SAID))

    try:
        call(work, url=url)
    except SkywardError as error:
        raise SystemExit(f"{error.code}: {error.message}") from None
    except KeyboardInterrupt:
        return


async def _lines(client: Client, refs: tuple[str, ...], head: int | None, types: tuple[str, ...] = (*TYPES, *SAID)) -> None:
    """Every event worth a line, on stdout, until the computes named are all deleted, or for good when none were.

    A compute is followed by a stream of its own, so the daemon sends only what is
    asked for; with none named, the one stream is the daemon's whole log, narrowed
    to the same frames. A compute already deleted has nothing more to say, and one
    that appears later is named by a read the first time it speaks.
    """
    found = [await client.call("GET", f"/v1/computes/{ref}", ComputeResource) for ref in refs]
    watched = {compute.id for compute in found if compute.status.state != "deleted"}
    if refs and not watched:
        raise SystemExit(f"nothing to watch: {', '.join(refs)} {'is' if len(refs) == 1 else 'are'} deleted")

    names = {compute.id: compute.name or compute.id for compute in found}
    color = sys.stdout.isatty()
    arrived: asyncio.Queue[Event] = asyncio.Queue()

    async def name(compute: str) -> str:
        if compute not in names:
            read = await client.call("GET", f"/v1/computes/{compute}", ComputeResource)
            names[compute] = read.name or compute
        return names[compute]

    async def pump(compute: str | None) -> None:
        async for _, payload in client.events(compute, types=types, after=head):
            if (event := decoded(payload)) is not None:
                await arrived.put(event)

    async with asyncio.TaskGroup() as group:
        pumps = [group.create_task(pump(compute)) for compute in (sorted(watched) or [None])]
        while not refs or watched:
            event = await arrived.get()
            if line := console.render(event, color) or console.ended(event, color):
                print(line if len(refs) == 1 else console.named(await name(event.compute), line, color), flush=True)
            match event:
                case ComputeDeleted(compute=compute):
                    watched.discard(compute)
        for running in pumps:
            running.cancel()


__all__ = ["monitor"]
