from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer
from casty.sharding import ClusteredActorSystem

from skyward.infra.worker import create_app, worker_behavior
from tests.conftest import get_free_port


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def worker_system(event_loop):
    async def _start():
        system = ClusteredActorSystem(
            name="test-http",
            host="127.0.0.1",
            port=get_free_port(),
            node_id="test-0",
        )
        await system.__aenter__()
        return system

    system = event_loop.run_until_complete(_start())
    yield system, event_loop
    event_loop.run_until_complete(system.__aexit__(None, None, None))


def test_health_endpoint(worker_system):
    system, loop = worker_system

    async def _test():
        local_ref = system.spawn(worker_behavior(0), "worker-health")
        broadcast_ref = local_ref
        app = create_app(system, local_ref, broadcast_ref, num_nodes=1)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ready"

    loop.run_until_complete(_test())
