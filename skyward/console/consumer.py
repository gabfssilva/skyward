"""Console consumer — serializes updates onto one renderer via a queue.

Replaces the console actor's mailbox: producers call ``tell`` from any
thread, a single asyncio task drains the queue and drives the renderer,
so renderer code never runs concurrently.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol

from .messages import ConsoleInput


class Renderer(Protocol):
    tick_interval: float | None

    def start(self, post: Callable[[ConsoleInput], None]) -> None: ...
    def handle(self, msg: ConsoleInput) -> None: ...
    def tick(self) -> None: ...
    def stop(self) -> None: ...


class ConsoleConsumer:
    def __init__(self, renderer: Renderer, loop: asyncio.AbstractEventLoop) -> None:
        self._renderer = renderer
        self._loop = loop
        self._queue: asyncio.Queue[ConsoleInput | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._renderer.start(self.tell)
        self._task = self._loop.create_task(self._run())

    def tell(self, msg: ConsoleInput) -> None:
        """Thread-safe enqueue."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._queue.put_nowait(None)
        await self._task
        self._task = None

    async def _run(self) -> None:
        interval = self._renderer.tick_interval
        try:
            while True:
                if interval is None:
                    msg = await self._queue.get()
                else:
                    try:
                        async with asyncio.timeout(interval):
                            msg = await self._queue.get()
                    except TimeoutError:
                        self._renderer.tick()
                        continue
                if msg is None:
                    return
                self._renderer.handle(msg)
        finally:
            self._renderer.stop()
