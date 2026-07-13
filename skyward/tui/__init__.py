"""Textual TUI for monitoring Skyward clusters.

Phase 1 is UI-first: :class:`SimulatedSource` drives the whole interface
with a scripted timeline so the look and feel can be tested without a real
session.  Run it with ``python -m skyward.tui``.
"""

from __future__ import annotations

from skyward.tui.app import SkywardTUI, run
from skyward.tui.sources import SimulatedSource

__all__ = ["SimulatedSource", "SkywardTUI", "run"]
