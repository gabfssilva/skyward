"""Themes and palettes for the TUI.

Two sources of colour, kept side by side so they never drift:

* ``Theme`` objects drive widget *chrome* (borders, panel/footer
  backgrounds) via Textual CSS variables.
* ``Palette`` dicts drive *content* rendered as Rich renderables inside
  ``Static``/``RichLog`` widgets, which do not inherit CSS variables.

Both mirror the mockup's ``PALS`` table.  ``NODE_COLORS`` is intentionally
theme-independent, matching the mockup.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.theme import Theme

from skyward.tui.model import UiStatus

__all__ = [
    "DARK",
    "LIGHT",
    "NODE_COLORS",
    "SKYWARD_DARK",
    "SKYWARD_LIGHT",
    "STATUS",
    "Palette",
    "palette_for",
]

NODE_COLORS: tuple[str, ...] = ("#3b82c4", "#2a9d8f", "#9b5fd0", "#cc7a3d")
"""Per-node accent colours, by ``node-<i> % 4``."""

STATUS: dict[UiStatus, tuple[str, str, str]] = {
    UiStatus.READY: ("green", "●", "ready"),
    UiStatus.BOOTSTRAPPING: ("amber", "◐", "booting"),
    UiStatus.PREEMPTED: ("red", "✕", "preempted"),
    UiStatus.REPLACING: ("cyan", "↻", "replacing"),
    UiStatus.WAITING: ("dim", "·", "waiting"),
    UiStatus.SSH: ("amber", "◐", "ssh"),
}
"""``status -> (palette-attr, glyph, label)``."""

PULSING: frozenset[UiStatus] = frozenset({UiStatus.BOOTSTRAPPING, UiStatus.REPLACING})
"""Statuses whose dot/phase pulses."""


@dataclass(frozen=True, slots=True)
class Palette:
    """Concrete hex colours for Rich-rendered content."""

    bg: str
    panel: str
    edge: str
    text: str
    bright: str
    dim: str
    faint: str
    accent: str
    tint: str
    hover: str
    green: str
    amber: str
    red: str
    cyan: str


LIGHT = Palette(
    bg="#fbfbfc", panel="#ffffff", edge="#e3e6ea", text="#1f2328",
    bright="#0b0e12", dim="#6e7781", faint="#b6bcc4", accent="#2f6fdb",
    tint="#eaf1fc", hover="#f1f3f6", green="#2da44e", amber="#bf8700",
    red="#cf222e", cyan="#0a7ea4",
)

DARK = Palette(
    bg="#0c0f15", panel="#11151d", edge="#222a35", text="#c9d1d9",
    bright="#f0f3f6", dim="#6e7787", faint="#2c3440", accent="#4d9bf0",
    tint="#15202f", hover="#161c26", green="#3fb950", amber="#d29922",
    red="#f85149", cyan="#39c5cf",
)


def _theme(name: str, p: Palette, *, dark: bool) -> Theme:
    return Theme(
        name=name,
        dark=dark,
        primary=p.accent,
        accent=p.accent,
        foreground=p.text,
        background=p.bg,
        surface=p.panel,
        panel=p.panel,
        success=p.green,
        warning=p.amber,
        error=p.red,
        variables={
            "edge": p.edge,
            "dim": p.dim,
            "faint": p.faint,
            "tint": p.tint,
            "hover": p.hover,
            "bright": p.bright,
            "cyan": p.cyan,
        },
    )


SKYWARD_LIGHT = _theme("skyward-light", LIGHT, dark=False)
SKYWARD_DARK = _theme("skyward-dark", DARK, dark=True)


def palette_for(theme_name: str) -> Palette:
    """Return the content :class:`Palette` matching a Textual theme name."""
    return LIGHT if theme_name == "skyward-light" else DARK
