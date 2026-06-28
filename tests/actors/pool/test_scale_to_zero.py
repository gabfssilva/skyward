"""Scale-to-zero: a pool can sit ``ready`` with zero nodes (lazy start) and
wake on demand. ``desired=0`` skips provisioning on enter; the first
``RequestScaleUp`` provisions from zero.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from casty import ActorContext, Behavior, Behaviors

from skyward.actors.messages import RequestScaleUp
from skyward.actors.pool.actor import pool_actor
from skyward.actors.pool.messages import PoolStarted, StartPool
from skyward.api.spec import Nodes, PoolSpec
from skyward.core.model import Cluster

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class _FakeClusterClient:
    def __init__(self, **_kwargs: Any) -> None: ...

    async def __aenter__(self) -> _FakeClusterClient:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


def _spec(nodes: Nodes) -> PoolSpec:
    return PoolSpec(
        nodes=nodes,
        accelerator=None,
        region="test",
        cluster=False,
        provision_retry_delay=0.0,
        max_provision_attempts=5,
    )


def _cluster(spec: PoolSpec) -> Cluster[Any]:
    return Cluster(
        id="c1",
        status="ready",
        spec=spec,
        offer=MagicMock(),
        ssh_key_path="/key",
        ssh_user="root",
        use_sudo=False,
        shutdown_command="shutdown",
        specific=MagicMock(),
        instances=(),
        prebaked=False,
        mount_plan=None,
    )


def _install_fakes(monkeypatch) -> dict[int, list[object]]:
    records: dict[int, list[object]] = {}

    def fake_node_actor(node_id: int, pool: Any, **_kwargs: Any) -> Behavior:
        async def receive(ctx: ActorContext, msg: object) -> Behavior:
            records.setdefault(node_id, []).append(msg)
            return Behaviors.same()
        return Behaviors.receive(receive)

    monkeypatch.setattr("skyward.actors.pool.actor.node_actor", fake_node_actor)
    monkeypatch.setattr("skyward.actors.pool.actor.ClusterClient", _FakeClusterClient)
    monkeypatch.setattr(
        "skyward.actors.pool.actor._build_pool_info_json",
        lambda *_a, **_k: "{}",
    )
    monkeypatch.setattr("skyward.infra.tls.ensure_ca", lambda: object())
    monkeypatch.setattr("skyward.infra.tls.issue_client_config", lambda _ca: None)
    return records


@pytest.mark.asyncio
async def test_lazy_start_provisions_zero_nodes_and_emits_pool_started(monkeypatch):
    _install_fakes(monkeypatch)

    spec = _spec(Nodes(desired=0, min=0, max=8))
    cluster = _cluster(spec)

    provision_calls: list[int] = []

    async def fake_provision(_cluster: Any, count: int) -> tuple[Any, tuple[Any, ...]]:
        provision_calls.append(count)
        return cluster, ()

    provider = MagicMock()
    provider.prepare = _async_return(cluster)
    provider.provision = fake_provision

    started: list[object] = []
    reply_to = MagicMock()
    reply_to.tell = lambda msg: started.append(msg)

    from casty import ActorSystem

    async with ActorSystem("pool-lazy-start") as system:
        ref = system.spawn(pool_actor(), "pool")
        ref.tell(StartPool(
            spec=spec, provider_config=MagicMock(), provider=provider,
            offers=(MagicMock(),), reply_to=reply_to,  # type: ignore[arg-type]
        ))
        await asyncio.sleep(0.3)

    assert provision_calls == [], (
        f"lazy start must not provision, got counts={provision_calls}"
    )
    pool_started = [m for m in started if isinstance(m, PoolStarted)]
    assert len(pool_started) == 1
    assert pool_started[0].instances == ()


@pytest.mark.asyncio
async def test_wake_up_from_zero_provisions_on_request_scale_up(monkeypatch):
    _install_fakes(monkeypatch)

    spec = _spec(Nodes(desired=0, min=0, max=8))
    cluster = _cluster(spec)

    provision_calls: list[int] = []

    async def fake_provision(_cluster: Any, count: int) -> tuple[Any, tuple[Any, ...]]:
        provision_calls.append(count)
        return cluster, (MagicMock(name="inst0"),)

    provider = MagicMock()
    provider.prepare = _async_return(cluster)
    provider.provision = fake_provision

    started: list[object] = []
    reply_to = MagicMock()
    reply_to.tell = lambda msg: started.append(msg)

    from casty import ActorSystem

    async with ActorSystem("pool-wake-up") as system:
        ref = system.spawn(pool_actor(), "pool")
        ref.tell(StartPool(
            spec=spec, provider_config=MagicMock(), provider=provider,
            offers=(MagicMock(),), reply_to=reply_to,  # type: ignore[arg-type]
        ))
        await asyncio.sleep(0.3)
        assert any(isinstance(m, PoolStarted) for m in started)
        provision_calls.clear()

        ref.tell(RequestScaleUp(count=1))
        await asyncio.sleep(0.3)

    assert provision_calls == [1], (
        f"wake-up must provision exactly one node, got {provision_calls}"
    )


def _async_return(value: Any):
    async def _fn(*_a: Any, **_k: Any) -> Any:
        return value
    return _fn
