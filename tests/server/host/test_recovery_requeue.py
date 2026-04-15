"""I3: queued executions on failed computes are interrupted."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _compute_spec() -> Any:
    from skyward.api.spec import Image, Nodes, Spec
    from skyward.providers.container.config import Container
    from skyward.server.host.domain import ComputeSpec

    return ComputeSpec(
        specs=(Spec(provider=Container(), image=Image(metrics=None)),),
        selection="cheapest",
        nodes=Nodes(desired=1),
        allocation="spot-if-available",
        ttl=timedelta(hours=1),
    )


@pytest.mark.asyncio
async def test_queued_on_ready_preserved_and_on_failed_interrupted(
    tmp_path: Path,
) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.domain import (
        Compute,
        Failed,
        InterruptedExec,
        Queued,
        Ready,
        Run,
        TaskExecution,
    )
    from skyward.server.host.pool_host import PoolHost
    from skyward.server.host.store import Store

    store = Store(str(tmp_path / "state.db"))
    await store.open()
    try:
        now = datetime.now(UTC)
        spec = _compute_spec()
        await store.put_compute(
            Compute(
                name="alive", spec=spec, created_at=now,
                status=Ready(
                    started_at=now, chosen=spec.specs[0],
                    nodes_ready=1, last_activity_at=now,
                ),
            ),
        )
        await store.put_compute(
            Compute(
                name="dead", spec=spec, created_at=now,
                status=Failed(failed_at=now, reason="previous_crash"),
            ),
        )
        for compute_name in ("alive", "dead"):
            await store.put_execution(
                TaskExecution(
                    id=f"exec-{compute_name}",
                    task=("m", "q"), compute=compute_name, kind=Run(),
                    payload=1, timeout=timedelta(seconds=60),
                    client=None, submitted_at=now, status=Queued(),
                ),
            )

        blobs = Blobs(store=store, root=tmp_path / "blobs")
        async with PoolHost(store, blobs, tmp_path / "logs") as host:
            final = {e.id: e for e in await host.store.list_executions()}
            # Workstream C: Ready without live rehydration is marked Failed,
            # so the queued execution on it is also interrupted (the pool
            # is gone).
            assert isinstance(final["exec-alive"].status, InterruptedExec)
            assert (
                final["exec-alive"].status.reason == "compute_failed_on_restart"
            )
            assert isinstance(final["exec-dead"].status, InterruptedExec)
            assert (
                final["exec-dead"].status.reason == "compute_failed_on_restart"
            )
    finally:
        if store._write is not None:
            await store.close()
