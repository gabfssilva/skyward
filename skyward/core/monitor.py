"""Generic background monitor for emitting events.

Provides a reusable abstraction for polling-based monitoring that can be
used for preemption detection, health checks, metrics collection, etc.

Example:
    def health_check() -> list[SkywardEvent]:
        return [HealthEvent(...)] if unhealthy else []

    m = monitor(
        name="health",
        interval=30,
        check=health_check,
        emit=emit,
    )
    m.start()
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field
from typing import Callable

from loguru import logger

from skyward.core.events import PoolStopping, SkywardEvent


@dataclass
class Monitor:
    """Generic background monitor that periodically checks and emits events.

    The monitor runs a background thread that:
    1. Waits for `interval` seconds
    2. Calls `check()` to get a list of events
    3. Emits each event via `emit()`
    4. Stops if `stop_event` type is encountered or `stop` is set

    Attributes:
        name: Monitor name (used for thread naming and logging).
        interval: Seconds between checks.
        check: Function that returns list of events to emit.
        emit: Function to emit events (typically skyward.core.callback.emit).
        stop_event: Event type that triggers monitor shutdown.
        stop: Threading event to signal shutdown.
    """

    name: str
    interval: float
    check: Callable[[], list[SkywardEvent]]
    emit: Callable[[SkywardEvent], None]
    stop_event: type[SkywardEvent] = PoolStopping
    stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        """Start monitoring in background thread.

        Uses contextvars.copy_context() to ensure events are emitted
        in the correct callback context.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning(f"Monitor {self.name} already running")
            return

        self.stop.clear()
        ctx = contextvars.copy_context()
        self._thread = threading.Thread(
            target=ctx.run,
            args=(self._loop,),
            daemon=True,
            name=f"monitor-{self.name}",
        )
        self._thread.start()
        logger.debug(f"Monitor {self.name} started (interval={self.interval}s)")

    def shutdown(self) -> None:
        """Stop monitoring and wait for thread to finish."""
        self.stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning(f"Monitor {self.name} did not stop cleanly")
            self._thread = None
        logger.debug(f"Monitor {self.name} stopped")

    def _loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop.wait(self.interval):
            try:
                events = self.check()
                for event in events:
                    self.emit(event)
                    if isinstance(event, self.stop_event):
                        logger.debug(f"Monitor {self.name} received stop event")
                        self.stop.set()
                        return
            except Exception as e:
                logger.warning(f"Monitor {self.name} check failed: {e}")


def monitor(
    *,
    name: str,
    interval: float,
    check: Callable[[], list[SkywardEvent]],
    emit: Callable[[SkywardEvent], None],
    stop_event: type[SkywardEvent] = PoolStopping,
) -> Monitor:
    """Factory function for creating monitors.

    Args:
        name: Monitor name for identification.
        interval: Seconds between checks.
        check: Function that returns events to emit.
        emit: Function to emit events.
        stop_event: Event type that triggers shutdown.

    Returns:
        Configured Monitor instance (not yet started).
    """
    return Monitor(
        name=name,
        interval=interval,
        check=check,
        emit=emit,
        stop_event=stop_event,
    )


__all__ = ["Monitor", "monitor"]
