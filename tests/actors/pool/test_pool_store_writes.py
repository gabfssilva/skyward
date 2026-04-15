"""F2: pool actor persists Compute lifecycle transitions via store.tx()."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

import pytest
from casty import ActorSystem

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@dataclass
class _StubOffer:
    """Minimal offer that provider.prepare/provision will accept."""

    instance_type: Any = None


def _make_spec(nodes: int = 1):
    from skyward.api.spec import Image, Nodes, PoolSpec

    return PoolSpec(
        nodes=Nodes(desired=nodes, min=nodes, max=nodes),
        accelerator=None,
        region="",
        image=Image(python="3.12"),
        ttl=3600,
    )


def _make_compute_spec(pool_spec: Any) -> Any:
    from datetime import timedelta

    from skyward.server.host.domain import ComputeSpec

    return ComputeSpec(
        specs=(),
        selection="first",
        nodes=pool_spec.nodes,
        allocation=pool_spec.allocation,
        ttl=timedelta(seconds=float(pool_spec.ttl)),
    )


class TestPoolStoreWrites:
    @pytest.mark.asyncio
    async def test_start_pool_writes_provisioning(self) -> None:
        """On StartPool, pool actor writes Compute(Provisioning) + event."""
        from skyward.actors.pool.actor import pool_actor
        from skyward.actors.pool.messages import (
            ProvisionFailed,
            StartPool,
        )
        from skyward.server.host.domain import Provisioning
        from skyward.server.host.store import Store

        store = Store(":memory:")
        await store.open()
        try:
            system = ActorSystem("test-f2-start")
            async with system:
                reply_ref = MagicMock()
                reply_ref.tell = MagicMock()
                spec = _make_spec()
                provider = MagicMock()

                fut: asyncio.Future[Any] = asyncio.Future()
                async def _never_completes():
                    return await fut

                provider.prepare = MagicMock(
                    side_effect=lambda *_a, **_k: _never_completes(),
                )

                ref = system.spawn(
                    pool_actor(pool_name="train", store=store),
                    "pool-train",
                )
                ref.tell(StartPool(
                    spec=spec,
                    provider_config=MagicMock(),
                    provider=provider,
                    offers=(_StubOffer(),),
                    compute_spec=_make_compute_spec(spec),
                    chosen_spec=MagicMock(),
                    reply_to=reply_ref,
                ))
                await asyncio.sleep(0.2)

                compute = await store.get_compute("train")
                assert compute is not None
                assert compute.name == "train"
                assert isinstance(compute.status, Provisioning)

                events = []
                async for e in store.tail_events(
                    since_id=0, aggregate_like="compute:train",
                ):
                    events.append(e)
                    if len(events) >= 1:
                        break
                assert any(e.type == "Pool.Provisioning" for e in events)
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_start_pool_then_stop_writes_failed_on_provision_error(self) -> None:
        """Unrecoverable provision failure writes Compute(Failed)."""
        from skyward.actors.pool.actor import pool_actor
        from skyward.actors.pool.messages import ProvisionFailed, StartPool
        from skyward.server.host.domain import Failed, Provisioning
        from skyward.server.host.store import Store

        store = Store(":memory:")
        await store.open()
        try:
            system = ActorSystem("test-f2-failed")
            async with system:
                reply_ref = MagicMock()
                reply_ref.tell = MagicMock()
                spec = _make_spec()
                provider = MagicMock()

                async def _fail():
                    raise RuntimeError("boom")

                provider.prepare = MagicMock(side_effect=lambda *_a, **_k: _fail())
                provider.terminate = MagicMock(
                    side_effect=lambda *_a, **_k: asyncio.sleep(0),
                )
                provider.teardown = MagicMock(
                    side_effect=lambda *_a, **_k: asyncio.sleep(0),
                )

                ref = system.spawn(
                    pool_actor(pool_name="train", store=store),
                    "pool-train",
                )
                ref.tell(StartPool(
                    spec=spec,
                    provider_config=MagicMock(),
                    provider=provider,
                    offers=(_StubOffer(),),
                    compute_spec=_make_compute_spec(spec),
                    chosen_spec=MagicMock(),
                    reply_to=reply_ref,
                ))
                await asyncio.sleep(0.3)

                compute = await store.get_compute("train")
                assert compute is not None
                assert isinstance(compute.status, Failed)
                assert "boom" in compute.status.reason
        finally:
            await store.close()
