from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from contextlib import suppress
from typing import Any

from skyward.observability.logger import logger

_FDS_PER_NODE: int = 10
_FD_BASE_OVERHEAD: int = 50


def check_fd_budget(nodes: int) -> None:
    import resource

    estimated = nodes * _FDS_PER_NODE + _FD_BASE_OVERHEAD
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= estimated:
        return

    target = min(int(estimated * 1.5), hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        logger.info(
            "Raised file descriptor limit from {old} to {new}",
            old=soft, new=target,
        )
    except (ValueError, OSError):
        logger.warning(
            "File descriptor limit ({soft}) may be insufficient for {nodes} nodes "
            "(estimated need: {estimated}). Consider running: ulimit -n {target}",
            soft=soft, nodes=nodes, estimated=estimated, target=target,
        )


def run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def run_sync[T](
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, T],
    timeout: float = 3600.0,
) -> T:
    future: Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def cleanup_loop(
    loop: asyncio.AbstractEventLoop | None,
    thread: threading.Thread | None,
) -> None:
    if loop is None:
        return

    loop.call_soon_threadsafe(loop.stop)
    logger.debug("Stopping event loop")

    if thread is not None:
        thread.join(timeout=10)
        if thread.is_alive():
            logger.warning("Event loop thread did not stop within 10s")

    if not loop.is_running():
        with suppress(Exception):
            _drain_pending_tasks(loop)
        with suppress(Exception):
            loop.close()


def _drain_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel lingering asyncio tasks and await their cancellation.

    During shutdown, a race between ``_stop_async`` resolving ask-futures
    and actors finishing their ``_do_stop`` child-cleanup can leave actor
    ``_run_loop`` tasks still pending when the event loop stops.  Cancelling
    and draining them here prevents 'Task was destroyed but it is pending'
    warnings and 'Event loop is closed' errors.
    """

    async def _cancel_all() -> None:
        tasks = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task()
        ]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    loop.run_until_complete(_cancel_all())
