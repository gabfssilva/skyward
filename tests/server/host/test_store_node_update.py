"""F4: ``Store.update_node_status`` applies partial node row updates."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.mark.asyncio
async def test_update_node_status_preserves_immutable_columns() -> None:
    from skyward.server.host.domain import (
        Node,
        NodeLost,
        NodeReady,
        NodeWaiting,
    )
    from skyward.server.host.store import Store

    store = Store(":memory:")
    await store.open()
    try:
        created_at = datetime.now(UTC)
        await store.put_node(
            Node(
                id="pool-0",
                compute="pool",
                instance_id="i-abc",
                provider_name="aws",
                head_addr=None,
                status=NodeWaiting(),
                created_at=created_at,
            ),
        )

        now = datetime.now(UTC)
        await store.update_node_status("pool-0", NodeReady(since=now))
        (row,) = await store.list_nodes(compute="pool")
        assert row.instance_id == "i-abc"
        assert row.provider_name == "aws"
        assert isinstance(row.status, NodeReady)

        await store.update_node_status(
            "pool-0", NodeLost(at=now, reason="preempted"),
        )
        (row,) = await store.list_nodes(compute="pool")
        assert row.instance_id == "i-abc"
        assert isinstance(row.status, NodeLost)
        assert row.status.reason == "preempted"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_update_node_status_sets_head_addr_when_provided() -> None:
    from skyward.server.host.domain import Node, NodeReady, NodeWaiting
    from skyward.server.host.store import Store

    store = Store(":memory:")
    await store.open()
    try:
        await store.put_node(
            Node(
                id="pool-1",
                compute="pool",
                instance_id="i-1",
                provider_name="aws",
                head_addr=None,
                status=NodeWaiting(),
                created_at=datetime.now(UTC),
            ),
        )
        await store.update_node_status(
            "pool-1",
            NodeReady(since=datetime.now(UTC)),
            head_addr="10.0.0.1",
        )
        (row,) = await store.list_nodes(compute="pool")
        assert row.head_addr == "10.0.0.1"
    finally:
        await store.close()
