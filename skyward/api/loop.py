from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from contextlib import suppress
from typing import Any

from skyward.observability.logger import logger


def cancel_pending_tasks() -> None:
    current = asyncio.current_task()
    for task in asyncio.all_tasks():
        if task is not current and not task.done():
            task.cancel()


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
            loop.close()
