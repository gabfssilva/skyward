"""Generic wait/polling utilities for providers.

Provides reusable polling functions to reduce duplication across providers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


async def wait_for_ready[T](
    poll_fn: Callable[[], Awaitable[T | None]],
    ready_check: Callable[[T], bool],
    *,
    terminal_check: Callable[[T], bool] | None = None,
    timeout: float = 300.0,
    interval: float = 5.0,
    description: str = "resource",
) -> T:
    """Wait until poll_fn returns something that passes ready_check.

    Args:
        poll_fn: Async function that polls for the resource state.
        ready_check: Function that returns True when resource is ready.
        terminal_check: Optional function that returns True if resource reached
            a terminal failure state (e.g., terminated, failed).
        timeout: Maximum time to wait in seconds.
        interval: Time between polls in seconds.
        description: Description for error messages.

    Returns:
        The ready resource.

    Raises:
        TimeoutError: If timeout is exceeded.
        RuntimeError: If resource reaches terminal state.
    """
    loop = asyncio.get_event_loop()
    start = loop.time()

    while True:
        result = await poll_fn()

        if result is not None:
            if ready_check(result):
                return result

            if terminal_check is not None and terminal_check(result):
                raise RuntimeError(f"{description} reached terminal state: {result}")

        elapsed = loop.time() - start
        if elapsed > timeout:
            raise TimeoutError(
                f"Timeout waiting for {description} after {timeout:.1f}s"
            )

        await asyncio.sleep(interval)
