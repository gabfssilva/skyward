"""The daemon's wakeup bus: identical payloads collapse, different ones do not."""

import asyncio
from typing import Any

import pytest
from litestar.events import listener

from skyward.server.http.emitter import ReconcilingEventEmitter

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
