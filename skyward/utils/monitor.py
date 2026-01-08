import threading
from dataclasses import dataclass, field
from typing import Callable, Any

from skyward import PoolStopping
from skyward.core import SkywardEvent

@dataclass
class Monitor:
    name: str
    thread: threading.Thread = field(init=False)
    interval: int
    check: Callable[[], list[SkywardEvent]]
    emit: Callable[[SkywardEvent], Any]
    stop_event: type[SkywardEvent] = field(default=PoolStopping)
    stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        self.thread = threading.Thread(target=self.loop, args=(), daemon=True)

    def start(self) -> None:
        self.thread.start()

    def loop(self):
        while not self.stop.wait(self.interval):
            events = self.check()

            for event in events:
                if type(event) is self.stop_event:
                    self.stop.set()

                self.emit(event)

def monitor(
    check: Callable[[], list[SkywardEvent]],
    *,
    name: str,
    interval: int,
    emit: Callable[[SkywardEvent], Any],
    stop_event: type[SkywardEvent] = PoolStopping,
) -> Monitor:
    return Monitor(
        name=name,
        interval=interval,
        emit=emit,
        stop_event=stop_event,
        check=check
    )
