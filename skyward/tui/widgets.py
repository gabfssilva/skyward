"""Reusable widgets for the TUI."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rich.console import RenderableType
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

__all__ = ["LogView", "Panel"]

_MAX_LINES = 500


class LogView(VerticalScroll):
    """Append-only, selectable log pane.

    Each log line is its own ``Static`` so the text is selectable with the
    mouse and copyable (``ctrl+c`` / ``cmd+c``) — ``RichLog`` is faster but its
    content cannot be selected.  Lines are appended (existing ones untouched)
    on routine updates so an in-progress selection survives; a full rebuild
    happens only when the rendering context changes (theme, different node).
    Scrolling is *sticky* — new lines only follow the bottom when the view is
    already there, so the user can scroll up to select older lines.

    ``can_focus`` is disabled so arrow keys reach the screen-level bindings;
    mouse-wheel scrolling and text selection work regardless of focus.
    """

    can_focus = False

    def __init__(self, *, id: str | None = None) -> None:  # noqa: A002 - Textual widget id
        super().__init__(id=id)
        self._ctx: str | None = None
        self._written = 0

    def feed(
        self,
        total: int,
        ctx: str,
        render_slice: Callable[[int], Sequence[RenderableType]],
    ) -> None:
        """Append rows ``[self._written, total)``, or rebuild on context change.

        Parameters
        ----------
        total : int
            Number of source rows available.
        ctx : str
            Token identifying the rendering context; a change forces a full
            rebuild (e.g. theme or selected node changed).
        render_slice : Callable[[int], Sequence[RenderableType]]
            Builds the renderables for source rows starting at the given index.
        """
        if ctx != self._ctx or total < self._written:
            self.remove_children()
            self._ctx = ctx
            self._written = 0
        if total == self._written:
            return
        at_bottom = self.scroll_offset.y >= self.max_scroll_y
        self.mount(*(Static(row, markup=False) for row in render_slice(self._written)))
        self._written = total
        if (excess := len(self.children) - _MAX_LINES) > 0:
            for child in list(self.children)[:excess]:
                child.remove()
        if at_bottom:
            self.call_after_refresh(self.scroll_end, animate=False)


class Panel(Vertical):
    """A bordered container with a top-left title and top-right subtitle.

    Maps the mockup's panel chips onto Textual border titles.  The border
    itself is styled in ``app.tcss``.

    Parameters
    ----------
    title : str
        Text shown in the top-left of the border.
    subtitle : str
        Text shown in the top-right of the border.
    id : str | None
        Widget id.
    """

    def __init__(
        self, *, title: str = "", subtitle: str = "", id: str | None = None,  # noqa: A002
    ) -> None:
        super().__init__(id=id)
        self._title = title
        self._subtitle = subtitle

    def on_mount(self) -> None:
        """Apply the configured border titles once mounted."""
        self.border_title = self._title
        self.border_subtitle = self._subtitle
