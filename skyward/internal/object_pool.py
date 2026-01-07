"""Generic thread-safe object pool with lazy creation and health checking."""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from queue import Empty, Queue

from loguru import logger


class ObjectPool[T]:
    """Thread-safe object pool with lazy creation.

    Objects are created on-demand when acquired, up to max_size.
    Optionally pre-warms to min_size in background.
    Optionally runs periodic health checks to validate objects.

    Queue.get() blocks when empty and at capacity - no polling needed.
    Queue.put() is thread-safe - no locks needed for queue operations.
    """

    def __init__(
        self,
        create: Callable[[int], T],
        close: Callable[[T], None],
        check: Callable[[T], bool],
        max_size: int = 4,
        min_size: int = 0,
        health_interval: float | None = None,
        max_concurrent: int = 1,
    ) -> None:
        """Create lazy pool.

        Args:
            create: Factory function (index) -> object.
            close: Cleanup function for each object.
            check: Health check function, returns True if object is valid.
            max_size: Maximum number of objects in pool.
            min_size: Minimum objects to pre-create in background (0 = fully lazy).
            health_interval: Seconds between health checks. None disables checking.
            max_concurrent: Max simultaneous object creations. 1 = mutex (default).
        """
        self._queue: Queue[T] = Queue()
        self._create = create
        self._close = close
        self._check = check
        self._max_size = max_size
        self._min_size = min_size
        self._health_interval = health_interval

        # Tracking created objects
        self._all: list[T | None] = []
        self._all_lock = threading.Lock()
        self._creation_sem = threading.Semaphore(max_concurrent)

        # Background threads
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        # Start warmup thread if min_size > 0
        if min_size > 0:
            t = threading.Thread(target=self._warmup_loop, daemon=True, name="pool-warmup")
            t.start()
            self._threads.append(t)

        # Start health check thread if interval specified
        if health_interval is not None:
            t = threading.Thread(
                target=self._health_check_loop, daemon=True, name="pool-health"
            )
            t.start()
            self._threads.append(t)

        logger.debug(
            f"ObjectPool: initialized (max={max_size}, min={min_size}, "
            f"health_interval={health_interval})"
        )

    def _warmup_loop(self) -> None:
        """Background thread that pre-creates objects up to min_size."""
        logger.debug(f"ObjectPool: warmup thread started, target={self._min_size}")
        while not self._stop_event.is_set():
            with self._all_lock:
                current = len(self._all)
                if current >= self._min_size:
                    logger.debug(f"ObjectPool: warmup complete ({current} objects)")
                    return
                idx = current
                # Reserve slot
                self._all.append(None)

            # Create with concurrency limit
            with self._creation_sem:
                try:
                    logger.debug(f"ObjectPool: warmup creating object {idx}")
                    obj = self._create(idx)
                    with self._all_lock:
                        self._all[idx] = obj
                    self._queue.put(obj)
                    logger.debug(f"ObjectPool: warmup created object {idx}")
                except Exception as e:
                    logger.warning(f"ObjectPool: warmup failed to create object {idx}: {e}")
                    # Remove reserved slot on failure
                    with self._all_lock:
                        if idx < len(self._all) and self._all[idx] is None:
                            self._all.pop(idx)

    def _try_create_one(self) -> T | None:
        """Try to create a new object if below max_size.

        Returns:
            New object, or None if at capacity.
        """
        with self._all_lock:
            if len(self._all) >= self._max_size:
                return None
            idx = len(self._all)
            # Reserve slot
            self._all.append(None)

        # Create with concurrency limit
        with self._creation_sem:
            logger.debug(f"ObjectPool: creating object {idx} on-demand")
            try:
                obj = self._create(idx)
                with self._all_lock:
                    self._all[idx] = obj
                logger.debug(f"ObjectPool: created object {idx}")
                return obj
            except Exception as e:
                logger.error(f"ObjectPool: failed to create object {idx}: {e}")
                # Remove reserved slot on failure
                with self._all_lock:
                    if idx < len(self._all) and self._all[idx] is None:
                        self._all.pop(idx)
                raise

    def _health_check_loop(self) -> None:
        """Background thread that periodically checks object health."""
        while not self._stop_event.wait(self._health_interval):
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

        if not checked:
            return

        logger.debug(f"ObjectPool: health check on {len(checked)} objects")

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
                logger.debug("ObjectPool: object failed health check, recreating")
                with suppress(Exception):
                    self._close(obj)

                with self._all_lock:
                    try:
                        idx = self._all.index(obj)
                    except ValueError as e:
                        # Object not in list (shouldn't happen)
                        logger.debug(f"ObjectPool: failed to fetch index: {e}")
                        continue

                try:
                    new_obj = self._create(idx)
                    with self._all_lock:
                        self._all[idx] = new_obj
                    self._queue.put(new_obj)
                except Exception as e:
                    logger.error(f"ObjectPool: failed to recreate object: {e}")
                    # Mark slot as empty
                    with self._all_lock:
                        if idx < len(self._all):
                            self._all[idx] = None

    def acquire(self, timeout: float | None = None) -> T:
        """Acquire object, creating one if needed and below max_size.

        Args:
            timeout: Max seconds to wait if at capacity. None = wait forever.

        Returns:
            Object from pool.

        Raises:
            Empty: If timeout expires while waiting for object.
        """
        # Try to get from queue first (non-blocking)
        try:
            obj = self._queue.get_nowait()
            logger.debug("ObjectPool.acquire: got existing object")
            return obj
        except Empty:
            pass

        # Try to create new one
        obj = self._try_create_one()
        if obj is not None:
            return obj

        # At capacity - must wait for release
        logger.debug(
            f"ObjectPool.acquire: at capacity ({self._max_size}), waiting for release"
        )
        obj = self._queue.get(timeout=timeout)
        logger.debug("ObjectPool.acquire: got object after waiting")
        return obj

    def release(self, obj: T) -> None:
        """Return object to the pool."""
        self._queue.put(obj)

    @contextmanager
    def __call__(self) -> Iterator[T]:
        """Acquire object as context manager."""
        obj = self.acquire()
        try:
            yield obj
        finally:
            self.release(obj)

    def close_all(self) -> None:
        """Stop background threads and close all objects."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=1)

        with self._all_lock:
            for obj in self._all:
                if obj is not None:
                    with suppress(Exception):
                        self._close(obj)
            self._all.clear()

    @property
    def total(self) -> int:
        """Total objects created (may include None slots on failure)."""
        with self._all_lock:
            return sum(1 for obj in self._all if obj is not None)

    @property
    def available(self) -> int:
        """Number of available objects in queue."""
        return self._queue.qsize()

    @property
    def max_size(self) -> int:
        """Maximum pool capacity."""
        return self._max_size
