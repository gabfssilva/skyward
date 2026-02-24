from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator


def async_to_sync[T](async_iterable: AsyncIterable[T]) -> Iterator[T]:
    q: queue.Queue[tuple[str, T | Exception | None]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        async def _consume() -> None:
            try:
                async for item in async_iterable:
                    await asyncio.to_thread(q.put, ("item", item))
            except Exception as e:
                await asyncio.to_thread(q.put, ("error", e))
            finally:
                await asyncio.to_thread(q.put, ("done", None))

        asyncio.run(_consume())

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        match q.get():
            case ("done", _):
                break
            case ("error", Exception() as e):
                raise e
            case ("item", payload):
                yield payload  # type: ignore[misc]


async def sync_to_async[T](iterable: Iterable[T]) -> AsyncIterator[T]:
    q: queue.Queue[tuple[str, T | Exception | None]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            for item in iterable:
                q.put(("item", item))
        except Exception as e:
            q.put(("error", e))
        finally:
            q.put(("done", None))

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        match await asyncio.to_thread(q.get):
            case ("done", _):
                break
            case ("error", Exception() as e):
                raise e
            case ("item", payload):
                yield payload  # type: ignore[misc]
