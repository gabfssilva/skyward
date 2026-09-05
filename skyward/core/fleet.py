"""Every live compute the daemon holds, each folded into its own view.

One :class:`~skyward.core.console.Observer` follows one compute, which is what
``sky monitor`` and the pool want. ``sky app`` wants all of them, and the shape
is the same thing multiplied: one observer per live compute, plus a listener on
the daemon's ``compute.created`` frames to learn of the next one. A compute that
reaches ``deleted`` is dropped and its observer cancelled, so an app left open
for a week holds exactly the streams it is showing.

The views are published as a mapping that is replaced by value, never mutated
in place: the observers write from the thread the console dispatch runs in, and
a reader on the event loop takes whichever snapshot is current.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import MappingProxyType

from skyward.core.client import Client
from skyward.core.console import Observer
from skyward.core.view import ComputeView, decoded
from skyward.shared.events import ComputeCreated, ComputeDeleted, Event
from skyward.shared.schemas import Compute, Page

type Fleet = Mapping[str, ComputeView]

LIVE = frozenset({"requested", "provisioning", "ready", "degraded", "deleting"})
"""The compute states that still owe something: what the fleet is made of."""


class FleetObserver:
    """The live computes, followed together.

    Parameters
    ----------
    client
        An open client to the daemon whose computes are watched.
    """

    def __init__(self, client: Client) -> None:
        self._client = client
        self._views: Fleet = MappingProxyType({})
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def views(self) -> Fleet:
        """Each live compute by id, as far as its stream and the API have said."""
        return self._views

    async def follow(self) -> None:
        """Watch until cancelled: what is live now, and whatever is created later."""
        loop = asyncio.get_running_loop()
        listed = await self._client.call("GET", "/v1/computes", Page[Compute], live=True)
        async with asyncio.TaskGroup() as group:
            for compute in listed.items:
                self._watch(group, loop, compute.id)
            async for _, payload in self._client.events(types=("compute.created",)):
                match decoded(payload):
                    case ComputeCreated(compute=compute_id) if compute_id not in self._tasks:
                        compute = await self._client.call("GET", f"/v1/computes/{compute_id}", Compute)
                        if compute.status.state in LIVE:
                            self._watch(group, loop, compute_id)
                    case _:
                        continue

    def _watch(self, group: asyncio.TaskGroup, loop: asyncio.AbstractEventLoop, compute_id: str) -> None:
        self._publish(compute_id, ComputeView(id=compute_id))
        observer = Observer(self._client, compute_id, watchers=(_Member(self, compute_id, loop),))
        self._tasks[compute_id] = group.create_task(observer.follow())

    def _publish(self, compute_id: str, view: ComputeView) -> None:
        self._views = MappingProxyType({**self._views, compute_id: view})

    def _forget(self, compute_id: str) -> None:
        self._views = MappingProxyType({key: view for key, view in self._views.items() if key != compute_id})
        if task := self._tasks.pop(compute_id, None):
            task.cancel()


class _Member:
    """The watcher one compute's observer dispatches to: it hands the view up."""

    def __init__(self, fleet: FleetObserver, compute_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self._fleet = fleet
        self._compute = compute_id
        self._loop = loop

    def opened(self, view: ComputeView) -> None:
        self._fleet._publish(self._compute, view)

    def event(self, event: Event, view: ComputeView) -> None:
        self._fleet._publish(self._compute, view)
        match event:
            case ComputeDeleted():
                self._loop.call_soon_threadsafe(self._fleet._forget, self._compute)
            case _:
                return None

    def closed(self, view: ComputeView) -> None:
        return None


__all__ = ["LIVE", "Fleet", "FleetObserver"]
