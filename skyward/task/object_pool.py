"""Generic thread-safe object pool with health checking."""

import threading
from collections.abc import Callable
from queue import Empty, Queue

from skyward.conc import map_async


class ObjectPool[T]:
    """Thread-safe object pool using Queue.

    Creates all objects in parallel at construction time.
    Optionally runs periodic health checks to validate objects.

    Queue.get() blocks when empty - no polling needed.
    Queue.put() is thread-safe - no locks needed.
    """

    def __init__(
        self,
        size: int,
        create: Callable[[int], T],
        close: Callable[[T], None],
        check: Callable[[T], bool],
        interval: float | None = None,
    ) -> None:
        """Create pool with objects created in parallel.

        Args:
            size: Number of objects to create.
            create: Factory function (index) -> object, called in parallel.
            close: Cleanup function for each object.
            check: Health check function, returns True if object is valid.
            interval: Seconds between health checks. None disables checking.
        """
        self._queue: Queue[T] = Queue()
        self._create = create
        self._close = close
        self._check = check
        self._interval = interval

        # Create all objects in parallel
        self._all = list(map_async(create, list(range(size))))
        for obj in self._all:
            self._queue.put(obj)

        # Start health check thread if interval specified
        self._stop_event = threading.Event()
        self._check_thread: threading.Thread | None = None
        if interval is not None:
            self._check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
            )
            self._check_thread.start()

    def _health_check_loop(self) -> None:
        """Background thread that periodically checks object health."""
        while not self._stop_event.wait(self._interval):
            self._run_health_checks()

    def _run_health_checks(self) -> None:
        """Check all available objects and recreate invalid ones."""
        checked: list[T] = []

        # Drain available objects
        while True:
            try:
                obj = self._queue.get_nowait()
                checked.append(obj)
            except Empty:
                break

        # Check each and recreate if invalid
        for obj in checked:
            try:
                is_valid = self._check(obj)
            except Exception:
                is_valid = False

            if is_valid:
                self._queue.put(obj)
            else:
                # Object invalid - close and recreate
                try:
                    self._close(obj)
                except Exception:
                    pass
                idx = self._all.index(obj)
                new_obj = self._create(idx)
                self._all[idx] = new_obj
                self._queue.put(new_obj)

    def acquire(self) -> T:
        """Acquire object (blocks if empty)."""
        return self._queue.get()

    def release(self, obj: T) -> None:
        """Return object to the pool."""
        self._queue.put(obj)

    def close_all(self) -> None:
        """Stop health checks and close all objects."""
        self._stop_event.set()
        if self._check_thread is not None:
            self._check_thread.join(timeout=1)

        for obj in self._all:
            try:
                self._close(obj)
            except Exception:
                pass

    @property
    def total(self) -> int:
        """Total objects in pool (available + in use)."""
        return len(self._all)

    @property
    def available(self) -> int:
        """Number of available objects."""
        return self._queue.qsize()

