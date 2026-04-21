"""Tests for the per-node idle scale-down detection."""

from __future__ import annotations

import pytest

from skyward.actors.node.actor import _should_announce_idle

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


# ── Pure decision helper ─────────────────────────────────────────


class TestShouldAnnounceIdle:
    def test_empty_inflight_past_threshold_announces(self) -> None:
        assert _should_announce_idle(
            inflight_count=0, last_task_at=100.0, now=200.0,
            threshold=60.0, announced=False,
        )

    def test_not_empty_does_not_announce(self) -> None:
        assert not _should_announce_idle(
            inflight_count=1, last_task_at=100.0, now=200.0,
            threshold=60.0, announced=False,
        )

    def test_within_threshold_does_not_announce(self) -> None:
        assert not _should_announce_idle(
            inflight_count=0, last_task_at=100.0, now=130.0,
            threshold=60.0, announced=False,
        )

    def test_already_announced_does_not_reannounce(self) -> None:
        assert not _should_announce_idle(
            inflight_count=0, last_task_at=100.0, now=300.0,
            threshold=60.0, announced=True,
        )

    def test_exact_threshold_not_announced(self) -> None:
        """Threshold is strictly ``> threshold``; equality must not trigger."""
        assert not _should_announce_idle(
            inflight_count=0, last_task_at=100.0, now=160.0,
            threshold=60.0, announced=False,
        )
