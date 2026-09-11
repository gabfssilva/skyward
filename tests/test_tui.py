"""The renderables ``sky app`` draws, against the views they are drawn from."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import pytest
from textual.app import App
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widgets import DataTable

from skyward.core import tui
from skyward.core.client import Client
from skyward.core.fleet import Fleet, FleetObserver
from skyward.core.tui import ComputeScreen, Detail, FleetScreen, tail
from skyward.core.view import ComputeView, NodeView


class _Fleet(FleetObserver):
    def __init__(self, client: Client, views: Mapping[str, ComputeView]) -> None:
        super().__init__(client)
        self.current: Fleet = dict(views)

    @property
    def views(self) -> Fleet:
        return self.current


class _Host(App[None]):
    def __init__(self, screen: Screen[None]) -> None:
        super().__init__()
        self._first = screen

    def on_mount(self) -> None:
        self.push_screen(self._first)


@asynccontextmanager
async def _fleet(*views: ComputeView) -> AsyncIterator[_Fleet]:
    client = await Client.remote("http://127.0.0.1:9")
    try:
        yield _Fleet(client, {view.id: view for view in views})
    finally:
        await client.close()


def _compute(compute_id: str, name: str, price: float = 0.0, nodes: int = 1) -> ComputeView:
    return ComputeView(
        id=compute_id,
        name=name,
        state="ready",
        nodes=tuple(NodeView(f"{compute_id}_n{rank}", rank=rank, state="ready", price_per_hour=price) for rank in range(nodes)),
    )


def _keys(table: DataTable[object]) -> list[str]:
    return [str(row.key.value) for row in table.ordered_rows]


def _names(table: DataTable[object]) -> list[str]:
    return [str(getattr(table.get_row_at(index)[0], "plain", "")) for index in range(table.row_count)]


async def _refreshed(pilot: Pilot[None]) -> None:
    await pilot.pause(tui.REFRESH * 8)


def describe_the_output_pane() -> None:
    def the_last_lines_come_from_every_node_in_the_order_spoken() -> None:
        view = ComputeView(
            id="cmp_1",
            nodes=(NodeView("nod_a", rank=0), NodeView("nod_b", rank=1)),
            tail=(("nod_a", "a1"), ("nod_b", "b1"), ("nod_a", "a2"), ("nod_b", "b2"), ("nod_a", "a3")),
        )

        assert tail(view, 3).plain == "[node 0] a2\n[node 1] b2\n[node 0] a3"


def describe_a_covered_screen() -> None:
    async def it_stops_repainting_the_fleet_while_another_screen_is_on_top(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tui, "REFRESH", 0.01)
        async with _fleet(_compute("cmp_a", "alpha")) as fleet:
            screen = FleetScreen(fleet, "http://127.0.0.1:17590")
            app = _Host(screen)
            async with app.run_test() as pilot:
                await _refreshed(pilot)
                table = screen.query_one("#computes", DataTable)
                assert _keys(table) == ["cmp_a"]

                await app.push_screen(Screen())
                await pilot.pause()
                fleet.current = {"cmp_a": _compute("cmp_a", "alpha"), "cmp_b": _compute("cmp_b", "beta")}
                await _refreshed(pilot)

                assert _keys(table) == ["cmp_a"]

    async def it_repaints_the_fleet_as_soon_as_it_is_shown_again(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tui, "REFRESH", 3600.0)
        async with _fleet(_compute("cmp_a", "alpha")) as fleet:
            screen = FleetScreen(fleet, "http://127.0.0.1:17590")
            app = _Host(screen)
            async with app.run_test() as pilot:
                await pilot.pause()
                table = screen.query_one("#computes", DataTable)
                await app.push_screen(Screen())
                await pilot.pause()
                fleet.current = {"cmp_a": _compute("cmp_a", "alpha"), "cmp_b": _compute("cmp_b", "beta")}

                app.pop_screen()
                await pilot.pause()

                assert sorted(_keys(table)) == ["cmp_a", "cmp_b"]

    async def it_resumes_the_timer_once_shown_again(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tui, "REFRESH", 0.01)
        async with _fleet(_compute("cmp_a", "alpha")) as fleet:
            screen = FleetScreen(fleet, "http://127.0.0.1:17590")
            app = _Host(screen)
            async with app.run_test() as pilot:
                await _refreshed(pilot)
                table = screen.query_one("#computes", DataTable)
                await app.push_screen(Screen())
                await pilot.pause()
                app.pop_screen()
                await _refreshed(pilot)

                fleet.current = {"cmp_a": _compute("cmp_a", "alpha"), "cmp_b": _compute("cmp_b", "beta")}
                await _refreshed(pilot)

                assert sorted(_keys(table)) == ["cmp_a", "cmp_b"]

    async def it_stops_repainting_one_compute_while_covered_and_repaints_when_shown(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tui, "REFRESH", 0.01)
        async with _fleet(_compute("cmp_a", "alpha", nodes=1)) as fleet:
            screen = ComputeScreen(fleet, "cmp_a")
            app = _Host(screen)
            async with app.run_test() as pilot:
                await _refreshed(pilot)
                nodes = screen.query_one(Detail).query_one("#nodes", DataTable)
                assert nodes.row_count == 1

                await app.push_screen(Screen())
                await pilot.pause()
                fleet.current = {"cmp_a": _compute("cmp_a", "alpha", nodes=3)}
                await _refreshed(pilot)
                assert nodes.row_count == 1

                app.pop_screen()
                await pilot.pause()

                assert nodes.row_count == 3


def describe_repainting_the_fleet_table() -> None:
    async def it_updates_cells_in_place_when_the_same_computes_keep_their_order(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tui, "REFRESH", 0.01)
        async with _fleet(_compute("cmp_a", "alpha", price=2.0), _compute("cmp_b", "beta", price=1.0)) as fleet:
            screen = FleetScreen(fleet, "http://127.0.0.1:17590")
            app = _Host(screen)
            async with app.run_test() as pilot:
                await _refreshed(pilot)
                table = screen.query_one("#computes", DataTable)
                before = dict(table.rows)

                fleet.current = {"cmp_a": _compute("cmp_a", "renamed", price=2.0), "cmp_b": _compute("cmp_b", "beta", price=1.0)}
                await _refreshed(pilot)

                assert _names(table) == ["renamed", "beta"]
                assert all(table.rows[key] is row for key, row in before.items())

    async def it_rebuilds_the_table_when_the_order_changes(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tui, "REFRESH", 0.01)
        async with _fleet(_compute("cmp_a", "alpha", price=2.0), _compute("cmp_b", "beta", price=1.0)) as fleet:
            screen = FleetScreen(fleet, "http://127.0.0.1:17590")
            app = _Host(screen)
            async with app.run_test() as pilot:
                await _refreshed(pilot)
                table = screen.query_one("#computes", DataTable)
                assert _keys(table) == ["cmp_a", "cmp_b"]
                before = dict(table.rows)

                fleet.current = {"cmp_a": _compute("cmp_a", "alpha", price=1.0), "cmp_b": _compute("cmp_b", "beta", price=3.0)}
                await _refreshed(pilot)

                assert _keys(table) == ["cmp_b", "cmp_a"]
                assert _names(table) == ["beta", "alpha"]
                assert all(table.rows[key] is not row for key, row in before.items())

    async def it_rebuilds_the_table_when_a_compute_joins(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tui, "REFRESH", 0.01)
        async with _fleet(_compute("cmp_a", "alpha", price=2.0)) as fleet:
            screen = FleetScreen(fleet, "http://127.0.0.1:17590")
            app = _Host(screen)
            async with app.run_test() as pilot:
                await _refreshed(pilot)
                table = screen.query_one("#computes", DataTable)

                fleet.current = {"cmp_a": _compute("cmp_a", "alpha", price=2.0), "cmp_b": _compute("cmp_b", "beta", price=1.0)}
                await _refreshed(pilot)

                assert _keys(table) == ["cmp_a", "cmp_b"]
                assert _names(table) == ["alpha", "beta"]


def describe_repainting_the_node_table() -> None:
    async def it_updates_node_cells_in_place_when_the_nodes_are_the_same(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tui, "REFRESH", 0.01)
        async with _fleet(_compute("cmp_a", "alpha", nodes=2)) as fleet:
            screen = ComputeScreen(fleet, "cmp_a")
            app = _Host(screen)
            async with app.run_test() as pilot:
                await _refreshed(pilot)
                nodes = screen.query_one(Detail).query_one("#nodes", DataTable)
                before = dict(nodes.rows)
                moved = _compute("cmp_a", "alpha", nodes=2)
                fleet.current = {
                    "cmp_a": ComputeView(
                        id="cmp_a",
                        name="alpha",
                        state="ready",
                        nodes=tuple(NodeView(node.id, rank=node.rank, state="ready", address=f"10.0.0.{node.rank}") for node in moved.nodes),
                    )
                }
                await _refreshed(pilot)

                assert [str(getattr(nodes.get_row_at(index)[4], "plain", "")) for index in range(2)] == ["10.0.0.0", "10.0.0.1"]
                assert all(nodes.rows[key] is row for key, row in before.items())

    async def it_rebuilds_the_node_table_when_the_nodes_change(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tui, "REFRESH", 0.01)
        async with _fleet(_compute("cmp_a", "alpha", nodes=2)) as fleet:
            screen = ComputeScreen(fleet, "cmp_a")
            app = _Host(screen)
            async with app.run_test() as pilot:
                await _refreshed(pilot)
                nodes = screen.query_one(Detail).query_one("#nodes", DataTable)

                fleet.current = {"cmp_a": _compute("cmp_a", "alpha", nodes=3)}
                await _refreshed(pilot)

                assert _keys(nodes) == ["cmp_a_n0", "cmp_a_n1", "cmp_a_n2"]
