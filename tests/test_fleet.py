"""The fleet: every live compute the daemon holds, and none that it does not."""

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path

import msgspec
import pytest

from skyward.core.client import Client
from skyward.core.fleet import FleetObserver
from skyward.server.application.mock import SPEC
from skyward.shared.schemas import Compute, ComputeCreate

pytestmark = pytest.mark.local


async def _create(client: Client, name: str) -> Compute:
    return await client.call(
        "POST",
        "/v1/computes",
        Compute,
        body=msgspec.json.encode(ComputeCreate(spec=SPEC, name=name)),
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )


async def _until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.05)


def describe_following_the_fleet() -> None:
    async def it_shows_what_was_live_before_it_looked_and_what_is_created_after(tmp_path: Path) -> None:
        client = await Client.embedded(tmp_path / "skyward.sqlite")
        try:
            before = await _create(client, "before")
            fleet = FleetObserver(client)
            following = asyncio.create_task(fleet.follow())

            await _until(lambda: before.id in fleet.views and fleet.views[before.id].name == "before")
            after = await _create(client, "after")
            await _until(lambda: after.id in fleet.views and fleet.views[after.id].name == "after")

            assert {view.name for view in fleet.views.values()} == {"before", "after"}
            following.cancel()
        finally:
            await client.close()

    async def a_compute_that_is_deleted_leaves_the_fleet(tmp_path: Path) -> None:
        client = await Client.embedded(tmp_path / "skyward.sqlite")
        try:
            compute = await _create(client, "brief")
            fleet = FleetObserver(client)
            following = asyncio.create_task(fleet.follow())
            await _until(lambda: compute.id in fleet.views)

            current = await client.call("GET", f"/v1/computes/{compute.id}", Compute)
            await client.call(
                "DELETE",
                f"/v1/computes/{compute.id}",
                Compute,
                headers={"If-Match": f'"{current.revision}"', "Idempotency-Key": uuid.uuid4().hex},
            )

            await _until(lambda: compute.id not in fleet.views, timeout=30.0)
            following.cancel()
        finally:
            await client.close()
