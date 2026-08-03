"""What the pool tells the person watching."""

import io
from collections.abc import AsyncIterator

from skyward.core.console import Console, render, watcher


class _Says:
    """A client with a stream and nothing else — all the log ever asks of one."""

    def __init__(self, *events: tuple[str, bytes]) -> None:
        self._events = events

    async def events(self, compute: str) -> AsyncIterator[tuple[str, bytes]]:
        for event in self._events:
            yield event


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_what_the_code_on_a_node_printed_is_the_point():
    assert render("node.console", {"node": "nod_1", "content": "epoch 3"}) == "nod_1 │ epoch 3"


def test_the_machines_say_where_they_got_to():
    assert render("node.ready", {"node": "nod_1"}) == "nod_1 │ ready"
    assert render("compute.ready", {"compute": "cmp_1"}) == "cmp_1 │ ready"


def test_a_failure_says_why():
    assert render("node.failed", {"node": "nod_1", "error": "no such package: torhc"}) == "nod_1 │ failed no such package: torhc"


def test_a_task_gets_no_line():
    """The caller is holding the result, or the exception. Both say more than a line would."""
    assert render("task.succeeded", {"compute": "cmp_1", "task": "tsk_1"}) is None


def test_a_metric_gets_no_line():
    """A gauge belongs in the pinned panel, not scrolling past in the log."""
    assert render("node.metrics", {"node": "nod_1", "name": "cpu"}) is None


async def test_a_gauge_does_not_stop_the_log():
    """A cost or a metric carries a number, and the log goes on reading past it."""
    out = io.StringIO()
    client = _Says(
        ("compute.cost", b'{"compute": "cmp_1", "cost": 0.12, "nodes": 2, "at": "2026-07-16T00:00:00+00:00"}'),
        ("node.metrics", b'{"compute": "cmp_1", "node": "nod_1", "name": "cpu", "value": 12.5}'),
        ("node.ready", b'{"compute": "cmp_1", "node": "nod_1"}'),
    )

    await Console(client, "cmp_1", out).follow()

    assert out.getvalue() == "nod_1 │ ready\n"


def test_colour_only_when_asked():
    """A redirect gets plain lines; a terminal gets the same line wrapped in escape codes."""
    assert "\033[" not in render("node.ready", {"node": "nod_1"})
    coloured = render("node.ready", {"node": "nod_1"}, color=True)
    assert "\033[" in coloured and "ready" in coloured and coloured.endswith("\033[0m")


def test_watcher_has_only_rich_and_log_modes():
    client = _Says()

    assert type(watcher(client, "cmp_1", _TTY(), mode="rich")).__name__ == "RichConsole"
    assert type(watcher(client, "cmp_1", _TTY(), mode="log")) is Console
