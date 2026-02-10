from __future__ import annotations

import asyncio
import threading

import pytest
from casty.sharding import ClusteredActorSystem

from skyward.distributed.proxies import set_system_loop
from skyward.distributed.registry import DistributedRegistry
from tests.conftest import get_free_port


@pytest.fixture(scope="session")
def _cluster_system():
    loop = asyncio.new_event_loop()
    system: ClusteredActorSystem | None = None

    async def _start() -> ClusteredActorSystem:
        s = ClusteredActorSystem(
            name="test",
            host="127.0.0.1",
            port=get_free_port(),
        )
        await s.__aenter__()
        return s

    ready = threading.Event()

    def _run_loop():
        nonlocal system
        asyncio.set_event_loop(loop)
        system = loop.run_until_complete(_start())
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=_run_loop, daemon=True)
    t.start()
    ready.wait()

    set_system_loop(loop)

    yield system, loop

    future = asyncio.run_coroutine_threadsafe(
        system.__aexit__(None, None, None), loop
    )
    future.result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)


@pytest.fixture
def registry(_cluster_system):
    system, loop = _cluster_system
    reg = DistributedRegistry(system, loop=loop)
    yield reg
    reg.cleanup()
