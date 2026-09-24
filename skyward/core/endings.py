"""Which of a compute's tasks have ended, heard on the one stream the daemon says so on.

A caller waiting on a result used to hold a request open until the task settled —
a long poll per pending future, each one a connection for as long as its function
ran. The pool of connections is finite, so the calls a driver could have in flight
were as many as the connections, however many slots the compute had: a hundred, on
a compute of three hundred workers, and the rest of them idle behind the pool.

So the waiting is done once per compute. The daemon records an ending for every
attempt that ends, and one stream carries all of them; a caller asks for its result
only once it has heard that its task's attempt ended, and the request answers at
once. An ending is a hint, not the verdict: a broadcast ends once per node, and the
result is what says whether the last of them is in — a caller told there is none
yet waits for the next ending.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import msgspec

from skyward.api.v1 import TaskStateEvent
from skyward.core.client import Client

ENDINGS = ("task.succeeded", "task.failed", "task.cancelled", "task.timed_out", "task.indeterminate")
"""The frames an attempt's ending goes out under; ``task.started`` and ``task.retrying`` end nothing."""

KEPT = 300.0
"""How long an ending nobody was waiting for is kept.

A task can end before the answer to its submission is back — the ending travels on
the stream, the answer on a connection of its own — and the caller who submitted it
starts waiting a moment after it was heard. What nobody claims in this long is some
other process's task on the same compute.
"""


class Endings:
    """One compute's endings, and the callers waiting on them."""

    def __init__(self, client: Client, compute: str) -> None:
        self._client = client
        self._compute = compute
        self._waiting: dict[str, asyncio.Event] = {}
        self._unclaimed: dict[str, float] = {}
        self._lost: Exception | None = None

    async def follow(self, after: int | None) -> None:
        """Hear every ending past ``after`` until cancelled; a stream that is lost for good fails whoever is waiting."""
        try:
            async for _, payload in self._client.events(self._compute, types=ENDINGS, after=after):
                self._heard(msgspec.json.decode(payload, type=TaskStateEvent).task)
        except Exception as lost:
            self._lost = lost
            for waiting in self._waiting.values():
                waiting.set()
            raise

    @asynccontextmanager
    async def watching(self, task: str) -> AsyncIterator[Callable[[], Awaitable[None]]]:
        """Wait on one task's endings: what comes back returns at the next one, or at once for one already heard."""
        ended = self._waiting[task] = asyncio.Event()
        if self._unclaimed.pop(task, None) is not None:
            ended.set()

        async def next_ending() -> None:
            if self._lost is None:
                await ended.wait()
                ended.clear()
            if self._lost is not None:
                raise ConnectionError(f"the daemon's stream of endings for compute {self._compute} was lost") from self._lost

        try:
            yield next_ending
        finally:
            del self._waiting[task]

    def _heard(self, task: str) -> None:
        if (waiting := self._waiting.get(task)) is not None:
            waiting.set()
            return

        heard = time.monotonic()
        self._unclaimed.pop(task, None)
        self._unclaimed[task] = heard
        while (oldest := next(iter(self._unclaimed.items()), None)) is not None and heard - oldest[1] > KEPT:
            del self._unclaimed[oldest[0]]
