"""What skyward adds over casty's collections, on one node: a synchronous face, and keys that come back as they went in.

A key is stored as its msgpack encoding, so the same key is the same entry on every
node. msgpack has one kind of array, and a key written as a tuple would come back a
list — which is no key at all in Python — unless it is made a tuple again on the way
out.
"""

import asyncio
from collections.abc import AsyncIterator

import casty
import pytest

from skyward.worker import distributed

pytestmark = pytest.mark.local


@pytest.fixture
async def bound() -> AsyncIterator[None]:
    async with casty.ActorSystem() as system:
        distributed.bind(system, asyncio.get_running_loop())
        try:
            yield
        finally:
            distributed.unbind()


def describe_the_keys_of_a_shared_map() -> None:
    async def a_tuple_comes_back_a_tuple(bound: None) -> None:
        def use() -> tuple[list[tuple[tuple[str, int], str]], str]:
            table: distributed.Dict[tuple[str, int], str] = distributed.dict("pairs")
            table[("a", 1)] = "first"
            return table.items(), table[("a", 1)]

        items, value = await asyncio.to_thread(use)

        assert items == [(("a", 1), "first")]
        assert value == "first"

    async def a_registry_lists_the_names_it_was_given(bound: None) -> None:
        def use() -> list[tuple[str, tuple[int, bool]] | str]:
            models: distributed.DistributedRegistry[tuple[str, tuple[int, bool]] | str, bytes] = distributed.registry("models")
            models.register(("run", (3, True)), b"weights")
            models.register("latest", b"weights")
            return models.list()

        listed = await asyncio.to_thread(use)

        assert sorted(listed, key=repr) == sorted([("run", (3, True)), "latest"], key=repr)

    async def an_absent_key_is_absent_and_not_an_error(bound: None) -> None:
        def use() -> tuple[str | None, bool, bool]:
            table: distributed.Dict[str, str] = distributed.dict("sparse")
            return table.get("nobody"), "nobody" in table, table.pop("nobody")

        assert await asyncio.to_thread(use) == (None, False, False)
