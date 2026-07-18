"""The cost of a compute is derived from its node rows, not accumulated.

The pure part — rounding elapsed time up to the provider's billing unit — is
tested against constructed nodes. The wiring — launch stamps the purchase, the
meter reads it back and publishes a gauge — runs against a real SQLite, same as
the rest of the persistence suite.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import msgspec
import pytest

from skyward.application.metering import Meter, accrued
from skyward.application.provider import Machine
from skyward.persistence.computes import ComputeStore
from skyward.persistence.db import connect
from skyward.persistence.events import EventStore
from skyward.persistence.nodes import NodeStore
from skyward.protocol.schemas import (
    BillingUnit,
    ComputeCreate,
    ComputeSpec,
    ComputeStatus,
    Node,
    NodeBounds,
    Offer,
    ProviderRef,
    Spec,
)

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)


def node(
    price: float | None = 3.6,
    unit: BillingUnit | None = "second",
    launched_at: datetime | None = NOW - timedelta(seconds=100),
    terminated_at: datetime | None = None,
) -> Node:
    return Node(
        id="nod_1",
        compute_id="cmp_1",
        generation=1,
        rank=0,
        revision=1,
        desired="present",
        state="ready",
        provider_binding={},
        created_at=NOW - timedelta(seconds=130),
        price_per_hour=price,
        billing_unit=unit,
        launched_at=launched_at,
        terminated_at=terminated_at,
    )


def test_cost_is_elapsed_time_at_the_hourly_price():
    assert accrued(node(price=3.6, unit="second"), at=NOW) == pytest.approx(0.1)


def test_elapsed_time_rounds_up_to_the_billing_unit():
    held_61s = node(launched_at=NOW - timedelta(seconds=61))
    assert accrued(msgspec.structs.replace(held_61s, billing_unit="minute"), at=NOW) == pytest.approx(3.6 * 120 / 3600)
    assert accrued(msgspec.structs.replace(held_61s, billing_unit="hour"), at=NOW) == pytest.approx(3.6)


def test_a_terminated_node_stops_costing():
    gone = node(terminated_at=NOW - timedelta(seconds=50))
    assert accrued(gone, at=NOW) == accrued(gone, at=NOW + timedelta(hours=1))


def test_a_node_with_no_machine_or_no_price_counts_as_zero():
    assert accrued(node(launched_at=None), at=NOW) == 0.0
    assert accrued(node(price=None), at=NOW) == 0.0


SPEC = ComputeSpec(
    specs=(Spec(provider=ProviderRef(kind="container"), cpus=2, memory_gb=2),),
    nodes=NodeBounds(desired=1),
)

OFFER = Offer(
    id="ofr_1",
    provider_id="prv_1",
    provider_name="fake",
    kind="fake",
    instance_type="fake.small",
    accelerator="A100",
    accelerator_count=1,
    cpus=2,
    memory_gb=2,
    spot_price=1.0,
    on_demand_price=2.0,
    billing_unit="minute",
    fetched_at=NOW,
    expires_at=NOW + timedelta(hours=1),
)


@pytest.fixture
async def stores(tmp_path: Path) -> tuple[ComputeStore, NodeStore, EventStore]:
    await connect(tmp_path / "skyward.sqlite")
    return ComputeStore(), NodeStore(), EventStore()


async def test_launching_stamps_the_purchase(stores: tuple[ComputeStore, NodeStore, EventStore]):
    computes, nodes, _ = stores
    created, _ = await computes.create(ComputeCreate(spec=SPEC, name=None), idempotency_key="k1")

    requested = await nodes.request(created.id, generation=1)
    await nodes.launched(requested.id, Machine(id="i-1", state="running", host="10.0.0.1"), offer=OFFER, market="spot")

    launched = await nodes.get(created.id, requested.id)
    assert launched.price_per_hour == 1.0, "the spot market sold, so the spot price is what bills"
    assert launched.market == "spot"
    assert launched.billing_unit == "minute"
    assert launched.accelerator == "a100", "the offer's accelerator arrives already normalized"
    assert launched.launched_at is not None


async def test_the_meter_publishes_what_a_live_compute_has_accrued(stores: tuple[ComputeStore, NodeStore, EventStore]):
    computes, nodes, events = stores
    created, _ = await computes.create(ComputeCreate(spec=SPEC, name=None), idempotency_key="k1")
    await computes.observe(created.id, ComputeStatus(state="ready", observed_generation=1, nodes_ready=1, nodes_total=1))

    requested = await nodes.request(created.id, generation=1)
    await nodes.launched(requested.id, Machine(id="i-1", state="running", host="10.0.0.1"), offer=OFFER, market="on_demand")

    samples: list[bytes] = []

    async def subscribe() -> None:
        async for _, _, payload in events.stream(None, created.id, None, ("compute.cost",)):
            samples.append(payload)
            return

    async with asyncio.TaskGroup() as group:
        follower = group.create_task(subscribe())
        await asyncio.sleep(0.05)
        await Meter(computes, nodes, events).sample()
        await asyncio.wait_for(follower, timeout=5)

    reading = msgspec.json.decode(samples[0])
    assert reading["compute"] == created.id
    assert reading["nodes"] == 1
    assert reading["cost"] == pytest.approx(2.0 / 60, abs=1e-4), "a just-launched node owes one on-demand minute"
