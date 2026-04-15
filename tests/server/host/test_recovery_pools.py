"""I2: transitional computes close on restart; ``Ready`` must rehydrate.

Workstream C changed the invariant: ``Ready`` no longer silently persists
across restart. When :meth:`PoolHost._recover_ready_computes` can't
rebuild the pool (e.g. no surviving ``NodeReady`` rows), the compute is
transitioned to ``Failed`` so queued executions don't hang.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

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
async def test_provisioning_and_stopping_closed_on_restart(
    tmp_path: Path,
) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.domain import (
        Compute,
        Failed,
        Provisioning,
        Ready,
        Stopped,
        Stopping,
    )
    from skyward.server.host.pool_host import PoolHost
    from skyward.server.host.store import Store

    store = Store(str(tmp_path / "state.db"))
    await store.open()
    try:
        now = datetime.now(UTC)
        await store.put_compute(
            Compute(
                name="a", spec=_compute_spec(), created_at=now,
                status=Provisioning(started_at=now),
            ),
        )
        await store.put_compute(
            Compute(
                name="b", spec=_compute_spec(), created_at=now,
                status=Stopping(started_at=now, stopping_since=now),
            ),
        )
        ready_spec = _compute_spec().specs[0]
        await store.put_compute(
            Compute(
                name="c", spec=_compute_spec(), created_at=now,
                status=Ready(
                    started_at=now, chosen=ready_spec,
                    nodes_ready=1, last_activity_at=now,
                ),
            ),
        )

        blobs = Blobs(store=store, root=tmp_path / "blobs")
        async with PoolHost(store, blobs, tmp_path / "logs") as host:
            final = {c.name: c for c in await host.store.list_compute()}
            assert isinstance(final["a"].status, Failed)
            assert final["a"].status.reason == "server_restart"
            assert isinstance(final["b"].status, Stopped)
            assert isinstance(final["c"].status, Failed)
            assert "recovery_failed" in final["c"].status.reason
    finally:
        if store._write is not None:
            await store.close()


from typing import Any  # noqa: E402  (used only in helper annotation)
