"""Application infrastructure: MonitorManager.

MonitorManager runs background monitoring loops.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


class MonitorManager:
    """Manages background monitor tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._shutdown = False

    def start(
        self,
        name: str,
        fn: Callable[..., Coroutine[Any, Any, Any]],
        interval: float,
    ) -> None:
        if name in self._tasks:
            raise ValueError(f"Monitor {name} already running")

        async def loop() -> None:
            while not self._shutdown:
                try:
                    await fn()
                except Exception as e:
                    print(f"[ERROR] Monitor {name} failed: {e}")
                await asyncio.sleep(interval)

        task = asyncio.create_task(loop())
        self._tasks[name] = task

    async def stop(self, name: str) -> None:
        task = self._tasks.pop(name, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def stop_all(self) -> None:
        self._shutdown = True
        for task in self._tasks.values():
            task.cancel()

        for _, task in list(self._tasks.items()):
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

    @property
    def running(self) -> list[str]:
        return list(self._tasks.keys())


__all__ = [
    "MonitorManager",
]
