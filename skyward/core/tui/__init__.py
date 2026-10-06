"""``sky app``: the fleet on one screen, and one compute under the cursor.

A Textual application over :class:`~skyward.core.fleet.FleetObserver` and the
daemon's API: the fleet screen lists the computes the way the Claude Code panel
does, and the compute screen is one of them with its nodes, its tasks and its
log. The screens draw with the console's own vocabulary
(:mod:`skyward.core.widgets`), so what the app shows and what a pool prints are
the same words in the same colours.
"""

from __future__ import annotations

from skyward.core.tui.app import Dashboard
from skyward.core.tui.cells import REFRESH
from skyward.core.tui.compute import ComputeScreen
from skyward.core.tui.fleet import FleetScreen

__all__ = ["REFRESH", "ComputeScreen", "Dashboard", "FleetScreen"]
