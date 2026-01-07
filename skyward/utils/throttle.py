"""Rate limiting with automatic backpressure."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from functools import wraps
from threading import Lock
from time import monotonic, sleep


class Throttle[**P, R]:
    """Thread-safe rate limiter with automatic backpressure.

    Limits function invocations to at most `calls` per `period` seconds.
    When limit is exceeded, blocks until a slot becomes available.

    Uses sliding window algorithm.

    Example as decorator:
        throttle = Throttle(calls=2, period=1.0)

        @throttle
        def call_api(url: str) -> Response:
            return httpx.get(url)

        # Calls automatically spaced to max 2/second
        for url in urls:
            call_api(url)

    Example as class-level shared throttle:
        class APIClient:
            _throttle = Throttle(calls=2, period=1.0)

            @_throttle
            def _request(self, method: str, url: str) -> Response:
                return httpx.request(method, url)

    Example with acquire():
        throttle = Throttle(calls=2, period=1.0)

        for url in urls:
            throttle.acquire()
            httpx.get(url)
    """

    __slots__ = ("calls", "period", "_timestamps", "_lock")

    def __init__(self, calls: int, period: float) -> None:
        """Initialize throttle.

        Args:
            calls: Maximum invocations allowed per time window.
            period: Time window in seconds.
        """
        self.calls = calls
        self.period = period
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> None:
        """Acquire a slot, blocking if rate limited."""
        while True:
            with self._lock:
                now = monotonic()
                cutoff = now - self.period

                # Evict expired timestamps
                while self._timestamps and self._timestamps[0] < cutoff:
                    self._timestamps.popleft()

                # Check if we have capacity
                if len(self._timestamps) < self.calls:
                    self._timestamps.append(now)
                    return  # Got a slot

                # Calculate wait time
                wait_time = self._timestamps[0] + self.period - now

            # Sleep OUTSIDE the lock, then re-check
            if wait_time > 0:
                sleep(wait_time)

    def __call__(self, fn: Callable[P, R]) -> Callable[P, R]:
        """Use as decorator."""

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self.acquire()
            return fn(*args, **kwargs)

        return wrapper
