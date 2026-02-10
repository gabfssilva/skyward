"""Generic retry decorator with exponential backoff.

Provides a flexible retry mechanism that works with any async function
and any type of exception.

Example:
    from skyward.retry import retry

    # Retry on any exception
    @retry()
    async def flaky_operation():
        ...

    # Retry only on specific exceptions
    @retry(on=ConnectionError)
    async def fetch_data():
        ...

    # Retry with custom predicate
    @retry(on=lambda e: "timeout" in str(e).lower())
    async def slow_api_call():
        ...

    # Retry on HTTP 429
    @retry(on=on_status_code(429))
    async def rate_limited_api():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
T = TypeVar("T")

# Type for the retry predicate
RetryPredicate = Callable[[Exception], bool]


def retry(
    on: type[Exception] | tuple[type[Exception], ...] | RetryPredicate = Exception,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator that retries async functions with exponential backoff.

    Args:
        on: When to retry. Can be:
            - An exception class (retry on that exception and subclasses)
            - A tuple of exception classes (retry on any of them)
            - A callable predicate (retry when predicate returns True)
            Default: retry on any Exception.
        max_attempts: Maximum number of attempts (including the first one).
        base_delay: Initial delay in seconds before first retry.
        exponential_base: Multiplier for exponential backoff.
            Delay formula: min(base_delay * (exponential_base ** attempt), max_delay)
        max_delay: Maximum delay cap in seconds.
        jitter: Whether to add random jitter (up to 10%) to prevent thundering herd.

    Returns:
        Decorated async function with retry behavior.

    Example:
        @retry(on=ConnectionError, max_attempts=3, base_delay=1.0)
        async def connect():
            return await client.connect()

        @retry(on=lambda e: isinstance(e, HTTPError) and e.status == 429)
        async def api_call():
            return await client.get("/data")
    """
    # Normalize the predicate
    if isinstance(on, type) and issubclass(on, Exception):
        should_retry: RetryPredicate = lambda e: isinstance(e, on)
    elif isinstance(on, tuple):
        should_retry = lambda e: isinstance(e, on)
    else:
        should_retry = on

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    has_retries_left = attempt < max_attempts - 1

                    if should_retry(e) and has_retries_left:
                        delay = min(base_delay * (exponential_base**attempt), max_delay)
                        if jitter:
                            delay += random.uniform(0, delay * 0.1)

                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} after {type(e).__name__}: "
                            f"{e}. Waiting {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        continue

                    # Don't retry: either predicate returned False or no retries left
                    raise

            # Should never reach here, but satisfy type checker
            assert last_exception is not None
            raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Common Predicates
# =============================================================================


def on_status_code(*codes: int) -> RetryPredicate:
    """Create a predicate that retries on specific HTTP status codes.

    Works with HttpError, aiohttp.ClientResponseError, and similar
    exceptions that expose a `status` attribute.

    Example:
        @retry(on=on_status_code(429, 503))
        async def api_call():
            ...
    """

    def predicate(e: Exception) -> bool:
        status = getattr(e, "status", None)
        return status in codes

    return predicate


def on_exception_message(*patterns: str, case_sensitive: bool = False) -> RetryPredicate:
    """Create a predicate that retries when exception message matches patterns.

    Example:
        @retry(on=on_exception_message("timeout", "connection reset"))
        async def network_call():
            ...
    """

    def predicate(e: Exception) -> bool:
        msg = str(e)
        if not case_sensitive:
            msg = msg.lower()
            return any(p.lower() in msg for p in patterns)
        return any(p in msg for p in patterns)

    return predicate


# =============================================================================
# Combining Predicates
# =============================================================================


def any_of(*predicates: RetryPredicate) -> RetryPredicate:
    """Combine predicates with OR logic (retry if ANY predicate matches).

    Example:
        @retry(on=any_of(
            on_status_code(429, 503),
            on_exception_message("timeout"),
        ))
        async def resilient_call():
            ...
    """

    def combined(e: Exception) -> bool:
        return any(p(e) for p in predicates)

    return combined


def all_of(*predicates: RetryPredicate) -> RetryPredicate:
    """Combine predicates with AND logic (retry only if ALL predicates match).

    Example:
        @retry(on=all_of(
            lambda e: isinstance(e, HTTPError),
            on_status_code(429),
        ))
        async def specific_retry():
            ...
    """

    def combined(e: Exception) -> bool:
        return all(p(e) for p in predicates)

    return combined
