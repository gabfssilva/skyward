"""Throttle decorator for rate limiting async functions.

Combines concurrency limiting (semaphore) with interval enforcement
to control the rate of function execution.

Example:
    from skyward import throttle, Limiter

    # Basic: 3 concurrent + 500ms between calls
    @throttle(max_concurrent=3, interval=0.5)
    async def api_call(): ...

    # Concurrency only
    @throttle(max_concurrent=5)
    async def bulk_operation(): ...

    # Interval only
    @throttle(interval=1.0)
    async def polling(): ...

    # Fail instead of wait
    @throttle(max_concurrent=2, on_limit="fail")
    async def time_sensitive(): ...

    # Shared limiter across functions
    shared = Limiter(max_concurrent=3, interval=0.5)

    @throttle(limiter=shared)
    async def method_a(): ...

    @throttle(limiter=shared)
    async def method_b(): ...
"""

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from loguru import logger


class ThrottleError(Exception):
    """Raised when throttle limit is reached and on_limit='fail'."""


class Limiter:
    """Shared throttle state for controlling concurrency and intervals.

    Combines semaphore-based concurrency limiting with interval enforcement
    between calls. Can be shared across multiple decorated functions.

    Args:
        max_concurrent: Maximum simultaneous executions. None = unlimited.
        interval: Minimum seconds between call starts. None = no limit.

    Example:
        # Create a shared limiter
        api_limiter = Limiter(max_concurrent=2, interval=0.5)

        # Use with multiple functions
        @throttle(limiter=api_limiter)
        async def endpoint_a(): ...

        @throttle(limiter=api_limiter)
        async def endpoint_b(): ...
    """

    def __init__(
        self,
        max_concurrent: int | None = None,
        interval: float | None = None,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._interval = interval
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, block: bool = True) -> bool:
        """Acquire permission to execute.

        Args:
            block: If True, wait until allowed. If False, return False immediately
                   if not allowed.

        Returns:
            True if acquired, False if non-blocking and couldn't acquire.
        """
        # Handle interval first (controls spacing between calls)
        if self._interval is not None:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_call
                wait_time = self._interval - elapsed

                if wait_time > 0:
                    if not block:
                        return False
                    logger.debug(f"Throttle: waiting {wait_time:.2f}s for interval")
                    await asyncio.sleep(wait_time)

                self._last_call = time.monotonic()

        # Handle concurrency (controls max parallel executions)
        if self._semaphore is not None:
            if not block:
                # Check if semaphore is available without blocking
                if self._semaphore.locked():
                    return False
            await self._semaphore.acquire()

        return True

    def release(self) -> None:
        """Release semaphore after execution completes."""
        if self._semaphore is not None:
            self._semaphore.release()

    def __repr__(self) -> str:
        return f"Limiter(max_concurrent={self._max_concurrent}, interval={self._interval})"


def throttle[**P, T](
    max_concurrent: int | None = None,
    interval: float | None = None,
    limiter: Limiter | None = None,
    on_limit: Literal["wait", "fail"] = "wait",
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator that throttles async function calls.

    Combines concurrency limiting (semaphore) with interval enforcement
    to control the rate of function execution.

    Args:
        max_concurrent: Maximum simultaneous executions. None = unlimited.
        interval: Minimum seconds between call starts. None = no limit.
        limiter: Shared Limiter instance. If provided, max_concurrent and
                 interval parameters are ignored.
        on_limit: Behavior when limit is reached:
                  - "wait": Block until allowed (default)
                  - "fail": Raise ThrottleError immediately

    Returns:
        Decorated async function with throttling behavior.

    Example:
        # Basic usage
        @throttle(max_concurrent=3, interval=0.5)
        async def api_call():
            ...

        # Shared limiter across functions
        api_limiter = Limiter(max_concurrent=2, interval=1.0)

        @throttle(limiter=api_limiter)
        async def endpoint_a(): ...

        @throttle(limiter=api_limiter)
        async def endpoint_b(): ...

        # Fail immediately if throttled
        @throttle(max_concurrent=1, on_limit="fail")
        async def exclusive_operation(): ...
    """
    # Create limiter if not provided
    _limiter = limiter or Limiter(max_concurrent=max_concurrent, interval=interval)

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            block = on_limit == "wait"

            acquired = await _limiter.acquire(block=block)
            if not acquired:
                raise ThrottleError(
                    f"Throttle limit reached for {func.__name__}. "
                    f"max_concurrent={_limiter._max_concurrent}, "
                    f"interval={_limiter._interval}s"
                )

            try:
                return await func(*args, **kwargs)
            finally:
                _limiter.release()

        return wrapper  # type: ignore[return-value]

    return decorator
