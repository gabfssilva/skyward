"""The whole thing, through the API, against real containers.

Nothing here reaches inside. A compute is created over HTTP, and the machines,
the SSH, the bootstrap, the casty cluster and the worker all happen because the
control plane decided they should. The only thing the test knows how to do is
what a user knows how to do: post, and then read.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import msgspec
import pytest
from litestar.testing import AsyncTestClient

from skyward.server.persistence.db import connect
from skyward.shared import codec
from skyward.shared.codec import digest
from skyward.shared.schemas import (
    Compute,
    ComputeCreate,
    ComputeSpec,
    Image,
    NodeBounds,
    ProviderRef,
    Spec,
    Task,
    TaskCreate,
)
from skyward.server.http.app import create_app, services

pytestmark = pytest.mark.e2e

SPEC = ComputeSpec(
    specs=(Spec(provider=ProviderRef(kind="container"), cpus=1, memory_gb=1),),
    nodes=NodeBounds(desired=2),
    image=Image(python="3.13", skyward="local"),
)


@pytest.fixture
async def client(tmp_path: Path):
    await connect(tmp_path / "skyward.sqlite")

    app = create_app(services())
    async with AsyncTestClient(app=app) as client:
        await client.post(
            "/v1/providers",
            json={"name": "local", "kind": "container", "credentials": {}, "config": {}},
        )
        yield client


async def until[T](what: Callable[[], Awaitable[T | None]], timeout: float = 600) -> T:
    async with asyncio.timeout(timeout):
        while True:
            if (value := await what()) is not None:
                return value
            await asyncio.sleep(0.5)


async def test_a_posted_compute_becomes_machines_that_run_python(client: AsyncTestClient):
    created = await client.post(
        "/v1/computes",
        content=msgspec.json.encode(ComputeCreate(spec=SPEC, name=None)),
        headers={"Idempotency-Key": uuid.uuid4().hex, "Content-Type": "application/json"},
    )
    assert created.status_code == 201, created.text
    compute = msgspec.json.decode(created.content, type=Compute)
    assert compute.status.state == "requested", "creating does not wait; the reconciler does the work"

    async def ready() -> Compute | None:
        response = await client.get(f"/v1/computes/{compute.id}")
        current = msgspec.json.decode(response.content, type=Compute)
        assert current.status.state != "failed", current.status.last_error
        return current if current.status.state == "ready" else None

    live = await until(ready)
    assert live.status.nodes_ready == 2, "both machines came up, and the compute says so"

    nodes = await client.get(f"/v1/computes/{compute.id}/nodes")
    assert nodes.status_code == 200

    def double(x: int) -> int:
        return x * 2

    code = await codec.payload.encode(double)
    args = await codec.payload.encode(((21,), {}))
    function = await digest(code)

    registered = await client.put(
        f"/v1/functions/{function}",
        content=code,
        headers={"Content-Type": "application/vnd.skyward.blob", "X-Skyward-Function-Name": "double"},
    )
    assert registered.status_code in (200, 201), registered.text

    submitted = await client.post(
        "/v1/tasks",
        content=msgspec.json.encode(
            TaskCreate(compute=compute.id, function=function, dispatch="one", args_inline=args),
        ),
        headers={"Idempotency-Key": uuid.uuid4().hex, "Content-Type": "application/json"},
    )
    assert submitted.status_code == 201, submitted.text
    task = msgspec.json.decode(submitted.content, type=Task)

    result = await client.get(f"/v1/tasks/{task.id}/result", params={"wait": 120})
    assert result.status_code == 200, result.text
    assert await codec.payload.decode(result.content) == 42

    latest = msgspec.json.decode((await client.get(f"/v1/computes/{compute.id}")).content, type=Compute)
    deleted = await client.delete(
        f"/v1/computes/{compute.id}",
        headers={"If-Match": f'"{latest.revision}"', "Idempotency-Key": uuid.uuid4().hex},
    )
    assert deleted.status_code == 202

    async def gone() -> bool | None:
        response = await client.get(f"/v1/computes/{compute.id}")
        current = msgspec.json.decode(response.content, type=Compute)
        return True if current.status.state == "deleted" else None

    assert await until(gone, timeout=120), "deleted means the provider confirmed the machines are gone"
