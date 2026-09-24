"""The daemon's wakeup bus: identical payloads collapse, different ones do not."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from litestar.events import listener

from skyward.server.http.emitter import Listener, ReconcilingEventEmitter
from skyward.shared.observability import entries
from tests.conftest import logfile

pytestmark = pytest.mark.local


class Colliding:
    """A payload whose hash says nothing about its identity."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __hash__(self) -> int:
        return 7

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Colliding) and other.label == self.label


def describe_coalescing() -> None:
    async def two_payloads_that_hash_alike_are_two_wakeups() -> None:
        seen: list[str] = []

        @listener("compute.changed")
        async def on_change(payload: Any) -> None:
            seen.append(payload.label)

        async with ReconcilingEventEmitter([on_change]) as emitter:
            emitter.emit("compute.changed", Colliding("a"))
            emitter.emit("compute.changed", Colliding("b"))
            await asyncio.sleep(0.05)

        assert sorted(seen) == ["a", "b"]

    async def the_same_payload_twice_is_one_wakeup_in_flight_and_one_after() -> None:
        seen: list[str] = []
        release = asyncio.Event()

        @listener("compute.changed")
        async def on_change(payload: Any) -> None:
            seen.append(payload.label)
            await release.wait()

        async with ReconcilingEventEmitter([on_change]) as emitter:
            emitter.emit("compute.changed", Colliding("a"))
            await asyncio.sleep(0.01)
            emitter.emit("compute.changed", Colliding("a"))
            emitter.emit("compute.changed", Colliding("a"))
            release.set()
            await asyncio.sleep(0.05)

        assert seen == ["a", "a"], "the duplicates emitted mid-flight collapse into one run afterwards"


def describe_failing_listener() -> None:
    async def it_is_logged_with_the_ids_it_was_called_for_and_nothing_reaches_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "skyward.log"

        @Listener("node.connect")
        async def on_node_connect(compute_id: str, node_id: str) -> None:
            raise RuntimeError("cannot reach member")

        with logfile(path):
            async with ReconcilingEventEmitter([on_node_connect]) as emitter:
                emitter.emit("node.connect", compute_id="cmp_a", node_id="nod_a")
                await asyncio.sleep(0.05)

        (failed,) = entries(path)
        assert (failed.component, failed.compute, failed.node, failed.fields) == ("emitter", "cmp_a", "nod_a", {"listener": "on_node_connect"})
        assert failed.exception is not None
        assert (failed.exception.type, failed.exception.message) == ("RuntimeError", "cannot reach member")
        assert capsys.readouterr().err == ""
