"""Tests for the Textual TUI (Phase 1, simulated source)."""

from __future__ import annotations

import pytest

from skyward.tui.app import SkywardTUI
from skyward.tui.screens import DashboardScreen, NodeScreen
from skyward.tui.sources import SimulatedSource
from skyward.tui.model import UiStatus

pytestmark = pytest.mark.unit


def test_simulation_reaches_full_health() -> None:
    """The scripted timeline brings node-2 up, preempts node-1, recovers it."""
    src = SimulatedSource(seed=7)

    statuses: dict[int, list[UiStatus]] = {1: [], 2: []}
    for _ in range(30):
        src.tick()
        snap = src.snapshot()
        statuses[1].append(snap.nodes[1].status)
        statuses[2].append(snap.nodes[2].status)

    final = src.snapshot()
    assert final.ready_count == 4
    assert final.t == 30
    # node-2 finished bootstrapping, node-1 went through the recovery path
    assert UiStatus.READY in statuses[2]
    assert UiStatus.PREEMPTED in statuses[1]
    assert UiStatus.REPLACING in statuses[1]
    assert final.nodes[1].status is UiStatus.READY
    assert final.nodes[1].started_at is not None
    assert len(final.cluster_log) > 5


def test_metrics_stay_in_bounds() -> None:
    """Jitter never pushes ready-node metrics out of range."""
    src = SimulatedSource(seed=3)
    for _ in range(120):
        src.tick()
    for node in src.snapshot().nodes:
        if node.status is UiStatus.READY:
            assert 0 <= node.gpu <= 99
            assert 0 <= node.mem <= 80
            assert 30 <= node.temp <= 92


async def test_navigation_and_expand() -> None:
    """Mount, open a node, expand on both screens, toggle theme, go back."""
    app = SkywardTUI(SimulatedSource(seed=7), tick_interval=3600.0)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen)

        # dashboard expand path (not covered by the node-screen smoke test)
        await pilot.press("z")
        await pilot.pause()
        assert app.screen.expanded is True
        assert app.screen.query_one("#dash-body").has_class("-expanded")
        await pilot.press("z")
        await pilot.pause()
        assert app.screen.expanded is False

        # navigate to node-2 and open it
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert isinstance(app.screen, NodeScreen)
        assert app.screen.index == 2

        # node screen switches nodes and expands its log
        await pilot.press("up")
        await pilot.pause()
        assert app.screen.index == 1
        await pilot.press("z")
        await pilot.pause()
        assert app.screen.expanded is True

        # escape collapses, then escape returns to the dashboard
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen.expanded is False
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen)
        # the last-viewed node carries back into the dashboard selection
        assert app.screen.selected == 1

        # theme toggle swaps the active theme (default is light, per the mockup)
        assert app.theme == "skyward-light"
        await pilot.press("t")
        await pilot.pause()
        assert app.theme == "skyward-dark"


async def test_copy_log_to_clipboard() -> None:
    """The copy-log action puts the rendered log text on the clipboard."""
    app = SkywardTUI(SimulatedSource(seed=7), tick_interval=3600.0)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        clip = app._clipboard
        assert "cluster bootstrapped" in clip
        assert "node-0" in clip
        assert clip.count("\n") >= 4  # multiple log lines copied


async def test_log_text_is_selectable() -> None:
    """Log lines are real selectable widgets (mouse drag -> selected text)."""
    from skyward.tui.widgets import LogView

    app = SkywardTUI(SimulatedSource(seed=7), tick_interval=3600.0)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        log = app.screen.query_one("#cluster-log", LogView)
        await pilot.triple_click(log, offset=(20, 1))
        await pilot.pause()
        selected = app.screen.get_selected_text()
        assert selected
        assert "cluster bootstrapped" in selected


async def test_log_is_append_only() -> None:
    """Routine ticks append to the log instead of clearing and rewriting it."""
    from skyward.tui.widgets import LogView

    app = SkywardTUI(SimulatedSource(seed=7), tick_interval=3600.0)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        log = app.screen.query_one("#cluster-log", LogView)
        before = log._written
        for _ in range(10):
            app.source.tick()
        app._refresh()
        await pilot.pause()
        after = log._written
        assert after > before  # grew via appends
        assert after == len(app.source.snapshot().cluster_log)


async def test_timer_during_mount_does_not_crash() -> None:
    """A tick/pulse firing while a pushed screen is mid-mount must not crash."""
    app = SkywardTUI(SimulatedSource(seed=7), tick_interval=3600.0)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for _ in range(6):
            await pilot.press("enter")  # push NodeScreen; children mount next cycle
            app._pulse()  # timer fires into the half-mounted screen -> must be guarded
            app._tick()
            await pilot.pause()
            await pilot.press("escape")
            app._pulse()
            await pilot.pause()
        assert app._exception is None
        assert isinstance(app.screen, DashboardScreen)
