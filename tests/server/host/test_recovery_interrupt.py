"""I1: ``PoolHost._interrupt_in_flight`` transitions non-``Queued`` rows."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.mark.asyncio
async def test_interrupts_dispatching_and_running_but_not_queued(
    tmp_path: Path,
) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.domain import (
        Dispatching,
        InterruptedExec,
        Queued,
        Run,
        RunningExec,
        TaskExecution,
    )
    from skyward.server.host.pool_host import PoolHost
    from skyward.server.host.store import Store

    store = Store(str(tmp_path / "state.db"))
    await store.open()

    try:
        now = datetime.now(UTC)
        rows = [
            TaskExecution(
                id=f"exec-{i}",
                task=("m", "q"),
                compute="pool",
                kind=Run(),
                payload=1,
                timeout=timedelta(seconds=60),
                client=None,
                submitted_at=now,
                status=st,
            )
            for i, st in enumerate(
                [Queued(), Dispatching(), RunningExec()],
            )
        ]
        for row in rows:
            await store.put_execution(row)

        blobs = Blobs(store=store, root=tmp_path / "blobs")
        async with PoolHost(store, blobs, tmp_path / "logs") as host:
            final = {e.id: e for e in await host.store.list_executions()}
            assert isinstance(final["exec-0"].status, Queued)
            assert isinstance(final["exec-1"].status, InterruptedExec)
            assert isinstance(final["exec-2"].status, InterruptedExec)
            assert final["exec-1"].status.reason == "server_restart"
    finally:
        if store._write is not None:
            await store.close()
