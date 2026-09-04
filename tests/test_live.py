"""The Rich console: one fold of the footer's state per event, however many lines the event prints."""

from io import StringIO

import pytest

from skyward.core import live
from skyward.core.view import ComputeView
from skyward.shared.events import ConsoleEvent

pytestmark = pytest.mark.local


def describe_the_live_console() -> None:
    def it_folds_the_footer_state_once_per_event(monkeypatch: pytest.MonkeyPatch) -> None:
        folds = 0
        original = live._state

        def counted(view: ComputeView) -> live._State:
            nonlocal folds
            folds += 1
            return original(view)

        monkeypatch.setattr(live, "_state", counted)
        console = live.RichConsole(StringIO())
        view = ComputeView(id="cmp_1", state="ready")
        console.opened(view)
        folds = 0

        console.event(ConsoleEvent(compute="cmp_1", node="nod_1", content="hello"), view)
        console.closed(view)

        assert folds == 1
