"""F3: pool actor persists Node rows when spawning node actors.

Pool is the single writer for the ``nodes`` table row creation; node
status transitions belong to the node actor (F4).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestPersistNodeWaiting:
    """``_persist_node_waiting`` writes an initial ``NodeWaiting`` row."""

    @pytest.mark.asyncio
    async def test_writes_waiting_row_and_event(self) -> None:
        from skyward.actors.pool.actor import _persist_node_waiting
        from skyward.server.host.domain import NodeWaiting
        from skyward.server.host.store import Store

        store = Store(":memory:")
        await store.open()
        try:
            class _Instance:
                id = "i-abc"

            class _Provider:
                name = "aws"

            await _persist_node_waiting(
                store=store,
                pool_name="train",
                nid=0,
                instance=_Instance(),
                provider=_Provider(),
                created_at=datetime.now(UTC),
            )

            nodes = await store.list_nodes(compute="train")
            assert len(nodes) == 1
            assert nodes[0].id == "train-0"
            assert nodes[0].instance_id == "i-abc"
            assert nodes[0].provider_name == "aws"
            assert isinstance(nodes[0].status, NodeWaiting)

            events = []
            async for e in store.tail_events(
                since_id=0, aggregate_like="compute:train",
            ):
                events.append(e)
            assert any(e.type == "Node.Waiting" for e in events)
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_provider_without_name_uses_classname(self) -> None:
        from skyward.actors.pool.actor import _persist_node_waiting
        from skyward.server.host.store import Store

        store = Store(":memory:")
        await store.open()
        try:
            class _Instance:
                id = "i-xyz"

            class MyProvider:
                pass

            await _persist_node_waiting(
                store=store,
                pool_name="pool",
                nid=3,
                instance=_Instance(),
                provider=MyProvider(),
                created_at=datetime.now(UTC),
            )

            nodes = await store.list_nodes(compute="pool")
            assert nodes[0].provider_name == "MyProvider"
        finally:
            await store.close()
