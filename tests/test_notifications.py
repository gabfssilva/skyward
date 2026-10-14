"""What went wrong lately, folded into notifications, and which of them a reader has seen."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import msgspec
import pytest

from skyward.api.v1 import LogEntryResource
from skyward.core import notifications
from skyward.core.client import Client
from skyward.core.notifications import Notification, Seen
from skyward.server.persistence.events import EventStore
from skyward.shared.events import ComputeDegraded, ConsoleEvent, NodeEvent, TaskEvent
from tests.conftest import held

pytestmark = pytest.mark.local

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
URL = "http://127.0.0.1:17590"


def _entry(sequence: int, minutes_ago: int, kind: str, data: dict[str, object]) -> LogEntryResource:
    return msgspec.convert(
        {"sequence": sequence, "type": kind, "at": NOW - timedelta(minutes=minutes_ago), "data": data},
        LogEntryResource,
        strict=False,
    )


def _lost(sequence: int, minutes_ago: int, compute: str = "cmp_a", why: str = "the provider stopped listing it") -> LogEntryResource:
    return _entry(sequence, minutes_ago, "node.lost", {"type": "node.state", "compute": compute, "node": f"nod_{sequence}", "state": "lost", "error": why})


def _failed(sequence: int, minutes_ago: int, why: str | None) -> LogEntryResource:
    return _entry(sequence, minutes_ago, "task.failed", {"type": "task.state", "compute": "cmp_a", "task": f"tsk_{sequence}", "state": "failed", "error": why})


def describe_unseen() -> None:
    def a_failure_repeated_for_the_same_reason_is_one_notification_the_newest_first() -> None:
        folded = notifications.unseen((_lost(9, 1), _lost(5, 10), _lost(2, 30)), frozenset())

        assert folded == (Notification("cmp_a", "node.lost", "the provider stopped listing it", (9, 5, 2), NOW - timedelta(minutes=1)),)

    def a_different_reason_kind_or_compute_is_another_the_newest_group_first() -> None:
        folded = notifications.unseen(
            (_failed(8, 1, "ZeroDivisionError"), _lost(7, 2), _lost(6, 3, why="spot reclaimed"), _lost(5, 4, compute="cmp_b"), _failed(4, 5, "KeyError")),
            frozenset(),
        )

        assert [(n.compute, n.kind, n.reason) for n in folded] == [
            ("cmp_a", "task.failed", "ZeroDivisionError"),
            ("cmp_a", "node.lost", "the provider stopped listing it"),
            ("cmp_a", "node.lost", "spot reclaimed"),
            ("cmp_b", "node.lost", "the provider stopped listing it"),
            ("cmp_a", "task.failed", "KeyError"),
        ]

    def what_was_seen_is_left_out_and_a_group_says_only_what_is_left() -> None:
        folded = notifications.unseen((_lost(9, 1), _lost(5, 10), _failed(4, 12, "ZeroDivisionError")), frozenset({9, 4}))

        assert folded == (Notification("cmp_a", "node.lost", "the provider stopped listing it", (5,), NOW - timedelta(minutes=10)),)

    def a_failure_without_a_reason_still_counts() -> None:
        (only,) = notifications.unseen((_failed(3, 1, None), _failed(2, 2, None)), frozenset())

        assert (only.reason, only.sequences) == (None, (3, 2))


def describe_describe() -> None:
    def a_kind_reads_as_words() -> None:
        assert [notifications.describe(kind) for kind in notifications.FAILURES] == [
            "degraded",
            "deletion failed",
            "node lost",
            "node failed",
            "task failed",
            "task timed out",
        ]


def describe_seen() -> None:
    def nothing_is_seen_before_anything_was_marked(tmp_path: Path) -> None:
        assert Seen(tmp_path / "seen.json", URL).read() == frozenset()

    def what_is_marked_is_read_back_by_another_reader_of_the_same_file(tmp_path: Path) -> None:
        Seen(tmp_path / "seen.json", URL).mark((3, 4), NOW)

        assert Seen(tmp_path / "seen.json", URL).read() == {3, 4}

    def marks_add_up(tmp_path: Path) -> None:
        seen = Seen(tmp_path / "seen.json", URL)
        seen.mark((1, 2), NOW - timedelta(hours=1))

        seen.mark((3,), NOW)

        assert seen.read() == {1, 2, 3}

    def what_was_marked_longer_ago_than_the_window_is_forgotten(tmp_path: Path) -> None:
        seen = Seen(tmp_path / "seen.json", URL)
        seen.mark((1, 2), NOW - notifications.WINDOW - timedelta(minutes=1))

        seen.mark((3,), NOW)

        assert seen.read() == {3}

    def each_daemon_has_its_own(tmp_path: Path) -> None:
        Seen(tmp_path / "seen.json", URL).mark((1,), NOW)
        Seen(tmp_path / "seen.json", "http://10.0.0.5:17590").mark((7,), NOW)

        assert Seen(tmp_path / "seen.json", URL).read() == {1}
        assert Seen(tmp_path / "seen.json", "http://10.0.0.5:17590").read() == {7}


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[Client]:
    opened = await Client.embedded(tmp_path / "skyward.sqlite")
    try:
        yield opened
    finally:
        await opened.close()


def describe_window() -> None:
    async def it_is_the_failures_the_daemon_recorded_the_newest_first(client: Client) -> None:
        await EventStore().record_all(
            (
                NodeEvent(compute="cmp_a", node="nod_1", state="lost", error="gone"),
                ConsoleEvent(compute="cmp_a", node="nod_2", content="hello"),
                NodeEvent(compute="cmp_a", node="nod_2", state="ready"),
                TaskEvent(compute="cmp_b", task="tsk_1", state="failed", error="ZeroDivisionError"),
                ComputeDegraded(compute="cmp_b", error="no offer"),
            ),
        )

        entries = await notifications.window(client, datetime.now(UTC))

        assert [entry.type for entry in entries] == ["compute.degraded", "task.failed", "node.lost"]

    async def it_narrows_to_one_compute(client: Client) -> None:
        compute = await held(client, "training")
        await EventStore().record_all(
            (NodeEvent(compute=compute.id, node="nod_1", state="lost", error="gone"), ComputeDegraded(compute="cmp_b", error="no offer")),
        )

        entries = await notifications.window(client, datetime.now(UTC), compute.id)

        assert [entry.type for entry in entries] == ["node.lost"]

    async def a_failure_older_than_the_window_is_left_out(client: Client) -> None:
        await EventStore().record(NodeEvent(compute="cmp_a", node="nod_1", state="lost", error="gone"))

        assert await notifications.window(client, datetime.now(UTC) + notifications.WINDOW + timedelta(minutes=1)) == ()
