"""Screens for the TUI: the dashboard and the per-node view.

Both share a top bar and the native ``Footer``; only the body differs.
Screens own the navigation/expand state and push rendered content into
thin widgets via :mod:`skyward.tui.render`.  The "expanded log" mode is a
reactive that hides the side widgets and lets the log grow to fill.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Static

from skyward.tui import render
from skyward.tui.model import UiCluster, UiLog
from skyward.tui.theme import palette_for
from skyward.tui.widgets import LogView, Panel

if TYPE_CHECKING:
    from skyward.tui.app import SkywardTUI

__all__ = ["DashboardScreen", "NodeScreen"]


class _Base[R](Screen[R]):
    """Shared lifecycle for screens that render a :class:`UiCluster`."""

    expanded: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._cluster: UiCluster | None = None

    @property
    def _ui(self) -> SkywardTUI:
        return cast("SkywardTUI", self.app)

    def on_mount(self) -> None:
        """Render the first frame from the current snapshot."""
        self.refresh_data(self._ui.source.snapshot())

    def refresh_data(self, cluster: UiCluster) -> None:  # noqa: B027 - overridden
        """Render a full frame for ``cluster`` (overridden by subclasses)."""

    def apply_pulse(self) -> None:  # noqa: B027 - overridden
        """Re-render only the pulsing widgets (overridden by subclasses)."""

    def _copy_log(self, logs: Sequence[UiLog], *, show_node: bool) -> None:
        self.app.copy_to_clipboard(render.log_text(logs, show_node=show_node))
        self.app.notify(f"Copied {len(logs)} log lines", title="clipboard", timeout=2)


class DashboardScreen(_Base[None]):
    """Cluster overview: summary header, node tree, and cluster log."""

    BINDINGS = [
        Binding("down", "nav_down", "select", key_display="↑↓"),
        Binding("up", "nav_up", "select", show=False),
        Binding("right", "tree_open", "open", show=False),
        Binding("left", "tree_close", "close", show=False),
        Binding("enter", "open_node", "open", key_display="↵"),
        Binding("c", "copy_log", "copy log"),
        Binding("z", "toggle_expand", "expand log"),
        Binding("escape", "collapse", "collapse", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.selected = 0
        self.root_open = True
        self.open_nodes: set[int] = set()

    def compose(self) -> ComposeResult:
        yield Static(id="topbar")
        with Vertical(id="dash-body"):
            yield Static(id="summary")
            with Horizontal(id="dash-cols"):
                with Panel(title="nodes", id="nodes-panel"):
                    yield Static(id="tree")
                with Panel(title="cluster log", subtitle="⤢ expand", id="log-panel"):
                    yield LogView(id="cluster-log")
        yield Footer()

    def refresh_data(self, cluster: UiCluster) -> None:
        self._cluster = cluster
        pal = palette_for(self.app.theme)
        self.query_one("#topbar", Static).update(render.topbar(cluster, pal))
        self.query_one("#summary", Static).update(render.summary(cluster, pal))
        self._render_tree()
        logs = cluster.cluster_log
        self.query_one("#cluster-log", LogView).feed(
            len(logs), self.app.theme,
            lambda start: render.log_rows(logs[start:], pal, show_node=True),
        )
        self.query_one("#log-panel", Panel).border_subtitle = (
            "⤡ collapse" if self.expanded else "⤢ expand"
        )

    def apply_pulse(self) -> None:
        self._render_tree()

    def _render_tree(self) -> None:
        if self._cluster is None:
            return
        self.selected = max(0, min(len(self._cluster.nodes) - 1, self.selected))
        self.query_one("#tree", Static).update(
            render.tree(
                self._cluster,
                palette_for(self.app.theme),
                selected=self.selected,
                open_nodes=frozenset(self.open_nodes),
                root_open=self.root_open,
                pulse_on=self._ui.pulse_on,
            )
        )

    def watch_expanded(self, value: bool) -> None:
        if not self.is_mounted:
            return
        self.query_one("#dash-body").set_class(value, "-expanded")
        if self._cluster is not None:
            self.refresh_data(self._cluster)

    def action_nav_down(self) -> None:
        self.selected += 1
        self._render_tree()

    def action_nav_up(self) -> None:
        self.selected -= 1
        self._render_tree()

    def action_tree_open(self) -> None:
        self.open_nodes.add(self.selected)
        self._render_tree()

    def action_tree_close(self) -> None:
        self.open_nodes.discard(self.selected)
        self._render_tree()

    def action_copy_log(self) -> None:
        if self._cluster is not None:
            self._copy_log(self._cluster.cluster_log, show_node=True)

    def action_open_node(self) -> None:
        self.app.push_screen(NodeScreen(self.selected), self._node_closed)

    def _node_closed(self, index: int | None) -> None:
        if index is not None:
            self.selected = index
            self._render_tree()

    def action_toggle_expand(self) -> None:
        self.expanded = not self.expanded

    def action_collapse(self) -> None:
        self.expanded = False


class NodeScreen(_Base[int]):
    """Per-node detail: breadcrumb, bootstrap phases, metrics, node log."""

    BINDINGS = [
        Binding("down", "next_node", "node", key_display="↑↓"),
        Binding("up", "prev_node", "node", show=False),
        Binding("c", "copy_log", "copy log"),
        Binding("z", "toggle_expand", "expand log"),
        Binding("escape", "back", "back"),
    ]

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def compose(self) -> ComposeResult:
        yield Static(id="topbar")
        with Vertical(id="node-body"):
            yield Static(id="breadcrumb")
            yield Static(id="phasebar")
            with Horizontal(id="node-cols"):
                with Panel(title="metrics", id="metrics-panel"):
                    yield Static(id="metrics")
                with Panel(title="log", id="node-log-panel"):
                    yield LogView(id="node-log")
        yield Footer()

    def refresh_data(self, cluster: UiCluster) -> None:
        self._cluster = cluster
        self.index = max(0, min(len(cluster.nodes) - 1, self.index))
        node = cluster.nodes[self.index]
        pal = palette_for(self.app.theme)

        self.query_one("#topbar", Static).update(render.topbar(cluster, pal))
        self.query_one("#breadcrumb", Static).update(
            render.breadcrumb(cluster, node, pal, self._ui.pulse_on)
        )

        phasebar = self.query_one("#phasebar", Static)
        if node.phase_idx < len(node.phases):
            phasebar.remove_class("-hidden")
            phasebar.update(render.phase_bar(node, pal, self._ui.pulse_on))
        else:
            phasebar.add_class("-hidden")

        self.query_one("#metrics", Static).update(render.metrics(node, pal, self._ui.pulse_on))
        logs = node.logs
        self.query_one("#node-log", LogView).feed(
            len(logs), f"{self.app.theme}:{self.index}",
            lambda start: render.log_rows(logs[start:], pal, show_node=False),
        )
        panel = self.query_one("#node-log-panel", Panel)
        panel.border_title = f"log · {node.id}"
        panel.border_subtitle = "⤡ collapse" if self.expanded else "⤢ expand"

    def apply_pulse(self) -> None:
        if self._cluster is None:
            return
        node = self._cluster.nodes[self.index]
        pal = palette_for(self.app.theme)
        self.query_one("#breadcrumb", Static).update(
            render.breadcrumb(self._cluster, node, pal, self._ui.pulse_on)
        )
        if node.phase_idx < len(node.phases):
            self.query_one("#phasebar", Static).update(
                render.phase_bar(node, pal, self._ui.pulse_on)
            )

    def watch_expanded(self, value: bool) -> None:
        if not self.is_mounted:
            return
        self.query_one("#node-body").set_class(value, "-expanded")
        if self._cluster is not None:
            self.refresh_data(self._cluster)

    def action_copy_log(self) -> None:
        if self._cluster is not None:
            self._copy_log(self._cluster.nodes[self.index].logs, show_node=False)

    def action_next_node(self) -> None:
        self.index += 1
        if self._cluster is not None:
            self.refresh_data(self._cluster)

    def action_prev_node(self) -> None:
        self.index -= 1
        if self._cluster is not None:
            self.refresh_data(self._cluster)

    def action_toggle_expand(self) -> None:
        self.expanded = not self.expanded

    def action_back(self) -> None:
        if self.expanded:
            self.expanded = False
        else:
            self.dismiss(self.index)
