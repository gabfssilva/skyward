"""The screens ``sky app`` draws, against the views and the daemon they are drawn from."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import msgspec
import pytest
from textual.app import App
from textual.dom import DOMNode
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widgets import Static

from skyward.api.v1 import ComputeResource, NodeResource, TaskResource
from skyward.core import history
from skyward.core.client import Client
from skyward.core.fleet import Fleet, FleetObserver
from skyward.core.tui import ComputeScreen, FleetScreen
from skyward.core.tui.dialogs import Ask, Confirm, Scale
from skyward.core.tui.log import Log
from skyward.core.tui.tasks import Tasks
from skyward.core.view import ComputeView, NodeView
from tests.conftest import held, machine

pytestmark = pytest.mark.local

URL = "http://127.0.0.1:17590"
NOW = datetime(2026, 9, 30, 12, 0, 10, tzinfo=UTC)


class _Fleet(FleetObserver):
    def __init__(self, client: Client, views: Mapping[str, ComputeView]) -> None:
        super().__init__(client)
        self.client = client
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


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[Client]:
    opened = await Client.embedded(tmp_path / "skyward.sqlite")
    try:
        yield opened
    finally:
        await opened.close()


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every timer at a pace a test can wait for."""
    monkeypatch.setattr("skyward.core.tui.fleet.REFRESH", 0.01)
    monkeypatch.setattr("skyward.core.tui.fleet.RECENT_POLL", 0.05)
    monkeypatch.setattr("skyward.core.tui.compute.REFRESH", 0.01)
    monkeypatch.setattr("skyward.core.tui.compute.POLL", 0.05)
    monkeypatch.setattr("skyward.core.tui.tasks.POLL", 0.05)
    monkeypatch.setattr("skyward.core.tui.log.POLL", 0.05)


@asynccontextmanager
async def _shown(screen: Screen[None]) -> AsyncIterator[Pilot[None]]:
    async with _Host(screen).run_test(size=(160, 50)) as pilot:
        await _settled(pilot)
        yield pilot


async def _settled(pilot: Pilot[None]) -> None:
    """Long enough for several repaints and reads: what a test waits before saying nothing changed."""
    await pilot.pause(0.4)


async def _until(pilot: Pilot[None], condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(20):
        while not condition():
            await pilot.pause(0.02)


@asynccontextmanager
async def _opened(screen: ComputeScreen) -> AsyncIterator[Pilot[None]]:
    """The compute screen, once its first read is on it."""
    async with _shown(screen) as pilot:
        await _until(pilot, lambda: "loading" not in _said(screen, "#header"))
        yield pilot


async def _read_until(pilot: Pilot[None], client: Client, compute_id: str, condition: Callable[[ComputeResource], bool]) -> ComputeResource:
    async with asyncio.timeout(20):
        while not condition(compute := await client.call("GET", f"/v1/computes/{compute_id}", ComputeResource)):
            await pilot.pause(0.02)
    return compute


_LOAD = {
    "cpu": {"at": 1, "value": 12.0},
    "gpu_util": {"at": 1, "value": 40.0},
    "mem_used_mb": {"at": 1, "value": 1024.0},
    "mem_total_mb": {"at": 1, "value": 8192.0},
    "disk_used_pct": {"at": 1, "value": 10.0},
    "net_rx_kbps": {"at": 1, "value": 120.0},
    "net_tx_kbps": {"at": 1, "value": 30.0},
    "gpu_mem_mb": {"at": 1, "value": 2048.0},
    "gpu_mem_total_mb": {"at": 1, "value": 16384.0},
    "gpu_temp_c": {"at": 1, "value": 61.0},
    "gpu_power_w": {"at": 1, "value": 180.0},
}
"""What a busy node reports, as the API tells it."""


def _text(widget: Static) -> str:
    return str(getattr(widget.visual, "plain", ""))


def _said(screen: DOMNode, selector: str) -> str:
    return _text(screen.query_one(selector, Static))


def _heads(screen: Screen[None]) -> list[str]:
    """Each node block's first line, reduced to its rank and state: ``#0 ready`` — without the spinner a moving state carries."""
    return [f"#{found[1]} {found[2]}" for found in re.finditer(r"^[▸▾]  #(\d+)  is (?:\S )?(\S+)", _said(screen, "#nodes"), re.MULTILINE)]


async def _click(pilot: Pilot[None], line: Static, label: str, row: int | None = None) -> None:
    """Click the word ``label`` where ``line`` draws it — on its ``row``, else the first row that has it — once laid out."""
    lines = _text(line).split("\n")
    at = next(index for index, drawn in enumerate(lines) if label in drawn) if row is None else row
    await _until(pilot, lambda: line.region.height > at)
    await pilot.pause()
    await pilot.click(line, offset=(lines[at].index(label), at))


def _canned(monkeypatch: pytest.MonkeyPatch, template: ComputeResource) -> dict[str, tuple[NodeResource, ...]]:
    """Make every snapshot the screen reads carry the nodes held under ``"nodes"``, which a test replaces."""
    held_nodes: dict[str, tuple[NodeResource, ...]] = {"nodes": ()}

    async def snapshot(client: Client, compute_id: str, now: datetime) -> history.Snapshot:
        return history.Snapshot(msgspec.structs.replace(template, nodes=held_nodes["nodes"]), {}, {}, None, now)

    monkeypatch.setattr(history, "snapshot", snapshot)
    return held_nodes


def _task(task_id: str, name: str, state: str, error: str | None = None) -> TaskResource:
    """One task as the API tells it, run once on ``nod_a`` a minute before ``NOW``."""
    failure = None if error is None else {"code": "task_failed", "message": error, "retryable": False, "request_id": None, "details": None}
    execution = {
        "id": f"exe_{task_id}",
        "rank": 0,
        "ordinal": 1,
        "state": state,
        "node_id": "nod_a",
        "retry_of": None,
        "result_sha256": None,
        "error": failure,
        "started_at": NOW - timedelta(minutes=1),
        "finished_at": NOW,
        "deadline_at": None,
        "stopping": False,
    }
    return msgspec.convert(
        {
            "id": task_id,
            "compute": {"id": "cmp_any", "name": "training"},
            "generation": 1,
            "function": {"sha256": "0" * 64, "name": name, "version": 1},
            "args_sha256": "1" * 64,
            "dispatch": "one",
            "state": state,
            "retry": None,
            "executions": (execution,),
            "submitted_at": NOW - timedelta(minutes=1),
            "finished_at": NOW,
            "rank": None,
            "correlation_id": None,
            "queue_timeout_seconds": None,
            "run_timeout_seconds": None,
            "result_sha256": None,
        },
        TaskResource,
    )


def _paged(monkeypatch: pytest.MonkeyPatch, items: tuple[TaskResource, ...]) -> None:
    """Make every page of tasks the section reads be ``items``."""

    async def tasks(client: Client, compute_id: str, **query: object) -> history.TaskPage:
        return history.TaskPage(items, len(items), None, {})

    monkeypatch.setattr(history, "tasks", tasks)


def _view(compute_id: str, name: str, price: float = 0.0, nodes: int = 1, minute: int = 0, error: str | None = None) -> ComputeView:
    return ComputeView(
        id=compute_id,
        name=name,
        state="ready",
        created_at=datetime(2026, 9, 30, 12, minute, tzinfo=UTC),
        cost=0.5,
        nodes=tuple(NodeView(f"{compute_id}_n{rank}", rank=rank, state="ready", price_per_hour=price) for rank in range(nodes)),
        errors=(error,) if error else (),
    )


def _listed(screen: FleetScreen) -> list[str]:
    return _said(screen, "#computes").split("\n")


def describe_the_list_of_computes() -> None:
    async def a_live_compute_is_a_line_with_its_state_shape_price_and_age(client: Client) -> None:
        fleet = _Fleet(client, {"cmp_a": _view("cmp_a", "alpha", price=1.25, nodes=2)})
        screen = FleetScreen(fleet, URL, client)
        async with _shown(screen):
            header, rule, alpha, *_ = _listed(screen)

            assert header.split() == ["NAME", "STATE", "GPU", "PROVIDER", "$/H", "COST", "AGE"]
            assert set(rule) == {"─"}
            assert alpha.split() == ["alpha", "●", "2/2", "—", "?", "2.50", "0.50", alpha.split()[-1]]
            assert alpha.split()[-1].endswith(("m", "h", "d"))

    async def the_newest_comes_first_and_the_total_closes_the_live_ones(client: Client) -> None:
        fleet = _Fleet(client, {"cmp_a": _view("cmp_a", "alpha", price=2.0, minute=1), "cmp_b": _view("cmp_b", "beta", price=1.0, minute=5)})
        screen = FleetScreen(fleet, URL, client)
        async with _shown(screen):
            names = [line.split()[0] for line in _listed(screen)[2:] if line.strip() and not set(line) <= {"─"}]

            assert names == ["beta", "alpha", "total"]
            assert [line for line in _listed(screen) if line.startswith("total")][0].split() == ["total", "3.00", "1.00"]

    async def an_error_is_said_on_the_line_after_the_compute(client: Client) -> None:
        fleet = _Fleet(client, {"cmp_a": _view("cmp_a", "alpha", error="the machine went away")})
        screen = FleetScreen(fleet, URL, client)
        async with _shown(screen):
            lines = _listed(screen)
            at = next(index for index, line in enumerate(lines) if line.startswith("alpha"))

            assert lines[at + 1] == "  the machine went away"

    async def it_repaints_as_the_fleet_changes(client: Client) -> None:
        fleet = _Fleet(client, {"cmp_a": _view("cmp_a", "alpha")})
        screen = FleetScreen(fleet, URL, client)
        async with _shown(screen) as pilot:
            fleet.current = {"cmp_a": _view("cmp_a", "renamed"), "cmp_b": _view("cmp_b", "beta")}

            await _until(pilot, lambda: "renamed" in _said(screen, "#computes") and "beta" in _said(screen, "#computes"))

    async def with_nothing_live_it_says_so(client: Client) -> None:
        screen = FleetScreen(_Fleet(client, {}), URL, client)
        async with _shown(screen) as pilot:
            await _until(pilot, lambda: "no live computes" in _said(screen, "#computes"))

    async def the_summary_names_the_daemon(client: Client) -> None:
        screen = FleetScreen(_Fleet(client, {}), URL, client)
        async with _shown(screen):
            assert _said(screen, "#summary").startswith("@ :17590 · updated ")


def describe_a_covered_fleet() -> None:
    async def it_stops_repainting_while_another_screen_is_on_top(client: Client) -> None:
        fleet = _Fleet(client, {"cmp_a": _view("cmp_a", "alpha")})
        screen = FleetScreen(fleet, URL, client)
        async with _shown(screen) as pilot:
            await pilot.app.push_screen(Screen())
            await pilot.pause()
            fleet.current = {"cmp_a": _view("cmp_a", "alpha"), "cmp_b": _view("cmp_b", "beta")}
            await _settled(pilot)

            assert "beta" not in _said(screen, "#computes")

    async def it_repaints_once_shown_again(client: Client) -> None:
        fleet = _Fleet(client, {"cmp_a": _view("cmp_a", "alpha")})
        screen = FleetScreen(fleet, URL, client)
        async with _shown(screen) as pilot:
            await pilot.app.push_screen(Screen())
            await pilot.pause()
            fleet.current = {"cmp_a": _view("cmp_a", "alpha"), "cmp_b": _view("cmp_b", "beta")}

            pilot.app.pop_screen()

            await _until(pilot, lambda: "beta" in _said(screen, "#computes"))


def describe_the_computes_that_ended() -> None:
    async def they_are_listed_under_the_live_ones_as_recent(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        ended = await held(client, "finished")

        async def recent(client: Client, limit: int = history.RECENT) -> tuple[ComputeResource, ...]:
            return (ended,)

        monkeypatch.setattr(history, "recent", recent)
        screen = FleetScreen(_Fleet(client, {"cmp_a": _view("cmp_a", "alpha")}), URL, client)
        async with _shown(screen) as pilot:
            await _until(pilot, lambda: "finished" in _said(screen, "#computes"))
            lines = _listed(screen)

            assert lines.index("recent") > next(index for index, line in enumerate(lines) if line.startswith("total"))
            assert lines[lines.index("recent") + 1].split()[:2] == ["finished", "○"]

    async def one_that_is_also_live_is_listed_once(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        both = await held(client, "training")

        async def recent(client: Client, limit: int = history.RECENT) -> tuple[ComputeResource, ...]:
            return (both,)

        monkeypatch.setattr(history, "recent", recent)
        screen = FleetScreen(_Fleet(client, {both.id: _view(both.id, "training")}), URL, client)
        async with _shown(screen):
            assert sum(line.startswith("training") for line in _listed(screen)) == 1
            assert "recent" not in _listed(screen)

    async def a_daemon_that_cannot_be_read_gives_the_list_way_to_saying_so(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        async def recent(client: Client, limit: int = history.RECENT) -> tuple[ComputeResource, ...]:
            return await client.call("GET", "/v1/computes/cmp_nobody", tuple[ComputeResource, ...])

        monkeypatch.setattr(history, "recent", recent)
        screen = FleetScreen(_Fleet(client, {"cmp_a": _view("cmp_a", "alpha")}), URL, client)
        async with _shown(screen) as pilot:
            await _until(pilot, lambda: "daemon request failed at" in _said(screen, "#computes"))

            assert "alpha" not in _said(screen, "#computes")
            assert "sky server start" in _said(screen, "#computes")


def describe_walking_the_list() -> None:
    async def enter_opens_the_compute_under_the_cursor_and_the_arrows_move_it(client: Client) -> None:
        first = await held(client, "first")
        second = await held(client, "second")
        views = {first.id: _view(first.id, "first", minute=5), second.id: _view(second.id, "second", minute=1)}
        screen = FleetScreen(_Fleet(client, views), URL, client)
        async with _shown(screen) as pilot:
            await pilot.press("down", "enter")

            await _until(pilot, lambda: isinstance(pilot.app.screen, ComputeScreen) and "second" in _said(pilot.app.screen, "#header"))

    async def a_name_is_a_link_to_its_compute(client: Client) -> None:
        compute = await held(client, "training")
        screen = FleetScreen(_Fleet(client, {compute.id: _view(compute.id, "training")}), URL, client)
        async with _shown(screen) as pilot:
            await _click(pilot, screen.query_one("#computes", Static), "training")

            await _until(pilot, lambda: isinstance(pilot.app.screen, ComputeScreen) and "training" in _said(pilot.app.screen, "#header"))

    async def the_way_back_is_a_link(client: Client) -> None:
        compute = await held(client, "training")
        screen = FleetScreen(_Fleet(client, {compute.id: _view(compute.id, "training")}), URL, client)
        async with _shown(screen) as pilot:
            await pilot.press("enter")
            await _until(pilot, lambda: isinstance(pilot.app.screen, ComputeScreen))

            await _click(pilot, pilot.app.screen.query_one("#actions", Static), "← computes")

            await _until(pilot, lambda: pilot.app.screen is screen)


def describe_the_compute_screen() -> None:
    async def it_says_what_the_compute_is(client: Client) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen):
            header = _said(screen, "#header")

            assert "training" in header
            assert "nodes 2–8" in header
            assert compute.id in header

    async def it_stops_reading_while_covered_and_reads_again_once_shown(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        compute = await held(client, "training")
        renamed = msgspec.structs.replace(compute, name="renamed")

        async def snapshot(client: Client, compute_id: str, now: datetime) -> history.Snapshot:
            return history.Snapshot(renamed, {}, {}, None, now)

        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            await pilot.app.push_screen(Screen())
            await pilot.pause()
            monkeypatch.setattr(history, "snapshot", snapshot)
            await _settled(pilot)
            assert "renamed" not in _said(screen, "#header")

            pilot.app.pop_screen()
            await _until(pilot, lambda: "renamed" in _said(screen, "#header"))

    async def a_read_that_fails_keeps_what_is_drawn_and_says_so(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:

            async def snapshot(client: Client, compute_id: str, now: datetime) -> history.Snapshot:
                return await client.call("GET", "/v1/computes/cmp_nobody", history.Snapshot)

            monkeypatch.setattr(history, "snapshot", snapshot)

            await _until(pilot, lambda: "daemon request failed" in _said(screen, "#header"))
            assert "training" in _said(screen, "#header")


def describe_the_nodes_of_a_compute() -> None:
    async def a_block_says_what_the_node_is_and_carries_its_load_under_it(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_a", 0, NOW, market="spot", price_per_hour=0.4, metrics=_LOAD),)
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen) as pilot:
            head, load = _said(screen, "#nodes").split("\n")
            assert re.fullmatch(r"▸  #0  is ready up [\dhm ]+ @ \$0\.40/h \(spot\)  logs  kill", head), head
            assert load == "    cpu 12%  gpu 40%  memory 1.0/8.0 GB  disk 10%"

            nodes["nodes"] = (machine("nod_a", 0, NOW, state="draining", market="spot", price_per_hour=0.4, metrics=_LOAD),)

            await _until(pilot, lambda: _heads(screen) == ["#0 draining"])

    async def one_that_was_replaced_is_listed_after_the_ones_alive(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_old", 0, NOW, state="deleted", terminated_at=NOW), machine("nod_new", 0, NOW), machine("nod_b", 1, NOW))
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen):
            assert _heads(screen) == ["#0 ready", "#1 ready", "#0 deleted"]
            assert _said(screen, "#groups") == "Nodes (2 active · 1 stopped)"

    async def g_walks_the_groups_that_have_nodes(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_old", 0, NOW, state="deleted", terminated_at=NOW), machine("nod_new", 0, NOW))
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen) as pilot:
            await pilot.press("g")
            assert _heads(screen) == ["#0 ready"]
            await pilot.press("g")
            assert _heads(screen) == ["#0 deleted"]
            await pilot.press("g")
            assert _heads(screen) == ["#0 ready", "#0 deleted"]

    async def enter_opens_the_node_under_the_cursor_on_its_trends_and_it_stays_open_across_reads(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_a", 0, NOW, address="10.0.0.5", metrics=_LOAD),)
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen) as pilot:
            assert "nod_a" not in _said(screen, "#nodes")

            await pilot.press("enter")
            lines = _said(screen, "#nodes").split("\n")
            assert lines[0].startswith("▾")
            assert lines[2].startswith("    cpu ") and "net" in lines[2]
            assert lines[3].startswith("    gpu ") and "vram 2.0/16.0 GB" in lines[3] and "temp 61°C" in lines[3] and "power 180 W" in lines[3]
            assert lines[4] == "    10.0.0.5 · nod_a"

            nodes["nodes"] = (machine("nod_a", 0, NOW, state="draining", address="10.0.0.5", metrics=_LOAD),)
            await _until(pilot, lambda: _heads(screen) == ["#0 draining"])
            assert "nod_a" in _said(screen, "#nodes")

            await pilot.press("enter")
            assert "nod_a" not in _said(screen, "#nodes")

    async def the_arrows_move_the_cursor_and_k_asks_before_killing_the_node_under_it(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_a", 0, NOW), machine("nod_b", 1, NOW))
        drained: list[tuple[str, str]] = []

        async def drain(client: Client, compute_id: str, node_id: str) -> None:
            drained.append((compute_id, node_id))

        monkeypatch.setattr("skyward.core.writes.drain", drain)
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen) as pilot:
            await pilot.press("k")
            await _until(pilot, lambda: isinstance(pilot.app.screen, Confirm))
            await pilot.press("enter")
            await _until(pilot, lambda: pilot.app.screen is screen)
            await _settled(pilot)
            assert drained == []

            await pilot.press("down", "k")
            await _until(pilot, lambda: isinstance(pilot.app.screen, Confirm))
            await pilot.press("tab", "enter")
            await _until(pilot, lambda: drained == [("cmp_any", "nod_b")])

    async def k_does_nothing_on_a_node_that_is_gone(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_old", 0, NOW, state="deleted", terminated_at=NOW),)
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen) as pilot:
            await pilot.press("k")
            await pilot.pause()

            assert pilot.app.screen is screen


def describe_writing_from_the_compute_screen() -> None:
    async def s_scales_to_the_bounds_typed(client: Client) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            await pilot.press("s")
            await _until(pilot, lambda: isinstance(pilot.app.screen, Scale))
            await pilot.press("3", "tab", "5", "enter")

            await _until(pilot, lambda: "nodes 3–5" in _said(screen, "#header"))
            scaled = await client.call("GET", f"/v1/computes/{compute.id}", ComputeResource)
            assert (scaled.spec.nodes.initial, scaled.spec.nodes.min, scaled.spec.nodes.max) == (4, 3, 5)

    async def a_bound_that_is_not_a_number_keeps_the_dialog_open(client: Client) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            await pilot.press("s")
            await _until(pilot, lambda: isinstance(pilot.app.screen, Scale))
            await pilot.press("9", "enter")

            await _until(pilot, lambda: "min <= max" in _said(pilot.app.screen, "#problem"))
            assert isinstance(pilot.app.screen, Scale)

    async def d_deletes_once_confirmed(client: Client) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            await pilot.press("d")
            await _until(pilot, lambda: isinstance(pilot.app.screen, Confirm))
            await pilot.press("enter")
            await _until(pilot, lambda: pilot.app.screen is screen)
            await _settled(pilot)
            kept = await client.call("GET", f"/v1/computes/{compute.id}", ComputeResource)
            assert history.deletable(kept)

            await pilot.press("d")
            await _until(pilot, lambda: isinstance(pilot.app.screen, Confirm))
            await pilot.press("tab", "enter")

            await _read_until(pilot, client, compute.id, lambda compute: not history.deletable(compute))


def describe_the_tasks_section() -> None:
    async def a_compute_with_no_tasks_says_so(client: Client) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            tasks = screen.query_one(Tasks)

            await _until(pilot, lambda: _said(tasks, "#title").startswith("Tasks (0)"))
            assert _said(tasks, "#rows") == "no tasks"

    async def a_task_is_a_line_with_its_function_and_its_error_under_it(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_a", 0, NOW),)
        _paged(monkeypatch, (_task("tsk_1", "nap", "succeeded"), _task("tsk_2", "boom", "failed", error="kaboom\nsecond line")))
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen) as pilot:
            tasks = screen.query_one(Tasks)

            await _until(pilot, lambda: "boom" in _said(tasks, "#rows"))
            first, second, error = _said(tasks, "#rows").split("\n", 2)
            assert re.fullmatch(r"✓  #0  nap  succeeded in [\dhms ]+ · [\dhms ]+ ago  logs", first), first
            assert re.fullmatch(r"✗  #0  boom  failed after [\dhms ]+ · [\dhms ]+ ago  logs", second), second
            assert error == "       kaboom\nsecond line"

    async def the_arrows_walk_the_lines_and_l_opens_the_log_on_the_one_under_the_cursor(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_a", 0, NOW),)
        _paged(monkeypatch, (_task("tsk_1", "nap", "succeeded"), _task("tsk_2", "boom", "failed")))
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen) as pilot:
            tasks = screen.query_one(Tasks)
            log = screen.query_one(Log)
            await _until(pilot, lambda: "boom" in _said(tasks, "#rows"))

            tasks.query_one("#rows").focus()
            await pilot.press("down", "l")
            await _until(pilot, lambda: log.task_id == "tsk_2")
            assert "logs ✕" in _said(tasks, "#rows").split("\n")[1]

            await pilot.press("l")
            await _until(pilot, lambda: log.task_id is None)
            assert "logs ✕" not in _said(tasks, "#rows")

    async def the_function_and_the_logs_of_a_task_are_links_on_its_line(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_a", 0, NOW),)
        _paged(monkeypatch, (_task("tsk_1", "nap", "succeeded"), _task("tsk_2", "boom", "failed")))
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen) as pilot:
            tasks = screen.query_one(Tasks)
            rows = tasks.query_one("#rows", Static)
            log = screen.query_one(Log)
            await _until(pilot, lambda: "boom" in _said(tasks, "#rows"))

            await _click(pilot, rows, "logs", row=1)
            await _until(pilot, lambda: log.task_id == "tsk_2")

            await _click(pilot, rows, "boom")
            await _until(pilot, lambda: "function: boom" in _said(tasks, "#status"))


def describe_a_daemon_slower_than_the_poll() -> None:
    async def its_tasks_are_still_heard(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        compute = await held(client, "training")
        read = history.tasks

        async def slow(client: Client, compute_id: str, **query: object) -> history.TaskPage:
            await asyncio.sleep(0.3)
            return await read(client, compute_id)

        monkeypatch.setattr(history, "tasks", slow)
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            await _until(pilot, lambda: _said(screen.query_one(Tasks), "#title").startswith("Tasks (0)"))

    async def its_log_is_still_heard(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        compute = await held(client, "training")
        read = history.log

        async def slow(client: Client, compute_id: str, **query: object) -> history.LogPage:
            await asyncio.sleep(0.3)
            return await read(client, compute_id)

        monkeypatch.setattr(history, "log", slow)
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            await _until(pilot, lambda: "compute." in _said(screen.query_one(Log), "#lines"))


def describe_the_log_section() -> None:
    async def it_lists_what_the_daemon_recorded(client: Client) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            log = screen.query_one(Log)

            await _until(pilot, lambda: "compute." in _said(log, "#lines"))
            assert "live" in _said(log, "#title")

    async def a_filter_is_named_on_the_title_and_narrows_the_lines(client: Client) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            log = screen.query_one(Log)

            log.show_task("tsk_nobody")

            await _until(pilot, lambda: _said(log, "#lines") == "no matching events")
            assert log.task_id == "tsk_nobody"
            assert "task tsk_nobody" in _said(log, "#title")

            log.query_one("#body").focus()
            await pilot.press("x")

            await _until(pilot, lambda: "compute." in _said(log, "#lines"))
            assert log.task_id is None


def describe_clicking_the_compute_screen() -> None:
    async def scale_and_delete_are_links_to_their_dialogs(client: Client) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            await _click(pilot, screen.query_one("#actions", Static), "scale")
            await _until(pilot, lambda: isinstance(pilot.app.screen, Scale))
            await pilot.press("escape")
            await _until(pilot, lambda: pilot.app.screen is screen)

            await _click(pilot, screen.query_one("#actions", Static), "delete")
            await _until(pilot, lambda: isinstance(pilot.app.screen, Confirm))

    async def a_group_is_filtered_on_by_clicking_it_and_dropped_by_clicking_it_again(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_old", 0, NOW, state="deleted", terminated_at=NOW), machine("nod_new", 0, NOW))
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen) as pilot:
            groups = screen.query_one("#groups", Static)

            await _click(pilot, groups, "1 stopped")
            await _until(pilot, lambda: _heads(screen) == ["#0 deleted"])

            await _click(pilot, groups, "1 stopped ✕")
            await _until(pilot, lambda: _heads(screen) == ["#0 ready", "#0 deleted"])

    async def a_node_is_opened_logged_and_killed_from_its_own_block(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_a", 0, NOW), machine("nod_b", 1, NOW))
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen) as pilot:
            blocks = screen.query_one("#nodes", Static)
            log = screen.query_one(Log)

            await _click(pilot, blocks, "▸", row=1)
            await _until(pilot, lambda: _text(blocks).split("\n")[1].startswith("▾") and "nod_b" in _text(blocks))
            assert _text(blocks).startswith("▸")

            await _click(pilot, blocks, "logs", row=1)
            await _until(pilot, lambda: log.node_id == "nod_b" and "node #1 ✕" in _said(log, "#title"))
            assert "logs ✕" in _text(blocks).split("\n")[1]

            await _click(pilot, log.query_one("#title", Static), "node #1 ✕")
            await _until(pilot, lambda: log.node_id is None)

            await _click(pilot, blocks, "kill", row=1)
            await _until(pilot, lambda: isinstance(pilot.app.screen, Confirm))
            assert "kill #1?" in _said(pilot.app.screen, ".question")

    async def a_node_that_is_gone_offers_no_kill(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
        nodes = _canned(monkeypatch, await held(client, "training"))
        nodes["nodes"] = (machine("nod_old", 0, NOW, state="deleted", terminated_at=NOW),)
        screen = ComputeScreen(client, "cmp_any")
        async with _opened(screen):
            assert "logs" in _said(screen, "#nodes")
            assert "kill" not in _said(screen, "#nodes")

    async def the_sort_of_the_tasks_is_a_link(client: Client) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            tasks = screen.query_one(Tasks)
            await _until(pilot, lambda: "sort: state" in _said(tasks, "#status"))

            await _click(pilot, tasks.query_one("#status", Static), "sort: state")

            await _until(pilot, lambda: "sort: submitted" in _said(tasks, "#status"))

    async def the_search_of_the_log_is_a_link(client: Client) -> None:
        compute = await held(client, "training")
        screen = ComputeScreen(client, compute.id)
        async with _opened(screen) as pilot:
            log = screen.query_one(Log)
            await _until(pilot, lambda: "search" in _said(log, "#title"))

            await _click(pilot, log.query_one("#title", Static), "search")
            await _until(pilot, lambda: isinstance(pilot.app.screen, Ask))
            await pilot.press("n", "o", "p", "e", "enter")

            await _until(pilot, lambda: '"nope" ✕' in _said(log, "#title") and _said(log, "#lines") == "no matching events")
