"""F5: ``Store.update_execution_status`` transitions executions atomically."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.mark.asyncio
async def test_update_execution_status_preserves_immutable_columns() -> None:
    from skyward.server.host.domain import (
        Dispatching,
        Queued,
        Run,
        SucceededExec,
        TaskExecution,
    )
    from skyward.server.host.store import Store

    store = Store(":memory:")
    await store.open()
    try:
        await store.put_execution(
            TaskExecution(
                id="01H999",
                task=("mymod", "mytask"),
                compute="train",
                kind=Run(),
                payload=1,
                timeout=timedelta(seconds=60),
                client="client-1",
                submitted_at=datetime.now(UTC),
                status=Queued(),
            ),
        )

        await store.update_execution_status("01H999", Dispatching())
        row = await store.get_execution("01H999")
        assert row is not None
        assert row.task == ("mymod", "mytask")
        assert row.compute == "train"
        assert row.payload == 1
        assert isinstance(row.status, Dispatching)

        finished = datetime.now(UTC)
        await store.update_execution_status(
            "01H999", SucceededExec(finished_at=finished),
        )
        row = await store.get_execution("01H999")
        assert row is not None
        assert isinstance(row.status, SucceededExec)
        assert row.payload == 1
    finally:
        await store.close()
