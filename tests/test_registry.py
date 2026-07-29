"""Distributed consistency and the named registry, over an in-process cluster.

``casty.local()`` is a whole cluster of one member, which is all the collections
need — the replica count folds to one and every write acks itself. Binding it into
``skyward.distributed`` is exactly what the worker does, minus the network.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import casty
import pytest

from skyward import distributed
from skyward.runtime import worker


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SKYWARD_NODE", "n0")
    monkeypatch.setenv("SKYWARD_COMPUTE", "c0")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "n0")

    loop = asyncio.new_event_loop()
    threading.Thread(
        target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()),
        daemon=True,
    ).start()

    async def _make() -> casty.ActorSystem:
        return casty.local()

    system = asyncio.run_coroutine_threadsafe(_make(), loop).result()
    distributed.bind(system, loop)

    yield

    distributed.unbind()
    asyncio.run_coroutine_threadsafe(system.close(), loop).result()
    loop.call_soon_threadsafe(loop.stop)


def test_one_node_folds_to_one_replica(bound: None) -> None:
    assert distributed.cluster().replicas == 1


def test_eventual_consistency_round_trips(bound: None) -> None:
    d = distributed.dict("scratch", consistency="eventual")
    d["loss"] = {"value": 0.5, "step": 3}
    assert d["loss"] == {"value": 0.5, "step": 3}

    c = distributed.counter("processed", consistency="eventual")
    c.add(5)
    assert c.get() == 5


def test_strong_is_the_default(bound: None) -> None:
    d = distributed.dict("model")
    assert d._params == {"consistency": "strong"}


def test_registry_registers_and_looks_up(bound: None) -> None:
    reg = distributed.registry("checkpoints")
    assert reg.lookup("step1") is None

    reg.register("step1", {"weights": [1, 2, 3]})
    assert reg.lookup("step1") == {"weights": [1, 2, 3]}
    assert reg.list() == ["step1"]

    assert reg.unregister("step1") is True
    assert reg.lookup("step1") is None


async def test_standalone_worker_does_not_expose_distributed_collections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYWARD_NODE", "n0")
    monkeypatch.setenv("SKYWARD_COMPUTE", "c0")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "n0")
    monkeypatch.setenv("SKYWARD_CLUSTER", "0")
    system = casty.local()
    try:
        worker.bind_distributed(system)

        with pytest.raises(RuntimeError):
            distributed.cluster()
    finally:
        distributed.unbind()
        await system.close()
