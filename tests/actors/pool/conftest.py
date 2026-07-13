"""Shared fakes for Pool tests: fake nodes, providers, and clients."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from skyward.actors.messages import NodeInstance
from skyward.api.spec import Nodes, PoolSpec
from skyward.core.model import Cluster, Instance


class FakeNode:
    """Stands in for ``Node``: adoptable, provisions instantly on demand.

    ``auto_ready`` drives the full callback sequence (connected → ready →
    activated) on the event loop, mimicking a healthy node lifecycle.
    """

    auto_ready = True

    def __init__(self, node_id: int, pool: Any, **_kwargs: Any) -> None:
        self.node_id = node_id
        self._pool = pool
        self.adopted = False
        self.provisioned = False
        self.stopped = False
        self.head_info = None
        self.joined: list[tuple[Any, ...]] = []
        self.ni: NodeInstance | None = None

    def provision(self, cluster: Any, provider: Any, instance: Any) -> None:
        self.provisioned = True
        self._start(cluster, instance)

    def adopt(self, cluster: Any, provider: Any, instance: Any) -> None:
        self.adopted = True
        self._start(cluster, instance)

    def _start(self, cluster: Any, instance: Any) -> None:
        self.ni = NodeInstance(
            instance=instance, node=self.node_id, provider="container",
            ssh_user="root", ssh_key_path="/key",
        )
        if self.auto_ready:
            asyncio.get_running_loop().create_task(self._lifecycle())

    async def _lifecycle(self) -> None:
        assert self.ni is not None
        self._pool.node_connected(self.node_id, self.ni)
        await self._pool.node_ready(
            self, self.ni, local_port=10000 + self.node_id,
            private_ip=f"10.0.0.{self.node_id}", transport=MagicMock(),
        )

    async def join(self, client: Any, pool_info_json: str, env: Any,
                   hooks: Any = (), process_hooks: Any = ()) -> None:
        self.joined.append((client, pool_info_json))
        self._pool.node_activated(self, slots=1)

    def set_head_info(self, info: Any) -> None:
        self.head_info = info

    async def execute(self, fn: Any, args: tuple, kwargs: dict, *,
                      timeout: float = 600.0, task_id: str = "") -> Any:
        return "ok"

    async def file_op(self, op: Any, path: str, content: bytes, timeout: float) -> Any:
        from skyward.actors.messages import NodeFileResult

        return NodeFileResult(node_id=self.node_id, success=True, listing="listing")

    async def stop(self) -> None:
        self.stopped = True


class FakeClusterClient:
    def __init__(self, **_kwargs: Any) -> None:
        self.closed = False

    async def __aenter__(self) -> FakeClusterClient:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        self.closed = True
        return False


def make_spec(nodes: Nodes, **kw: Any) -> PoolSpec:
    defaults: dict[str, Any] = {
        "accelerator": None,
        "region": "test",
        "cluster": False,
        "provision_retry_delay": 0.0,
        "max_provision_attempts": 3,
        "reconcile_tick_interval": 3600.0,
        "autoscale_cooldown": 3600.0,
    }
    defaults.update(kw)
    return PoolSpec(nodes=nodes, **defaults)


def make_cluster(spec: PoolSpec) -> Cluster[Any]:
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


def make_instance(iid: str) -> Instance:
    offer = MagicMock()
    return Instance(id=iid, ip="10.0.0.1", status="provisioned", offer=offer)


class FakeProvider:
    def __init__(self, cluster: Cluster[Any], *, per_call: int | None = None) -> None:
        self.cluster = cluster
        self.per_call = per_call
        self.provision_calls: list[int] = []
        self.terminated: list[tuple[str, ...]] = []
        self.teardown_calls = 0
        self._counter = 0

    async def prepare(self, _spec: Any, _offer: Any) -> Cluster[Any]:
        return self.cluster

    async def provision(
        self, cluster: Cluster[Any], count: int,
    ) -> tuple[Cluster[Any], tuple[Instance, ...]]:
        self.provision_calls.append(count)
        n = count if self.per_call is None else min(self.per_call, count)
        instances = tuple(
            make_instance(f"i-{self._counter + i}") for i in range(n)
        )
        self._counter += n
        return cluster, instances

    async def get_instance(self, cluster: Any, iid: str) -> tuple[Any, Instance]:
        return cluster, make_instance(iid)

    async def terminate(self, _cluster: Any, iids: tuple[str, ...]) -> None:
        self.terminated.append(tuple(iids))

    async def teardown(self, _cluster: Any) -> None:
        self.teardown_calls += 1


@pytest.fixture
def patch_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skyward.actors.pool.pool.ClusterClient", FakeClusterClient)
    monkeypatch.setattr(
        "skyward.actors.pool.pool._build_pool_info_json", lambda *_a, **_k: "{}",
    )
    monkeypatch.setattr("skyward.infra.tls.ensure_ca", lambda: object())
    monkeypatch.setattr("skyward.infra.tls.issue_client_config", lambda _ca: None)


@pytest.fixture
def patch_nodes(monkeypatch: pytest.MonkeyPatch, patch_pool: None) -> dict[int, FakeNode]:
    created: dict[int, FakeNode] = {}

    def factory(node_id: int, pool: Any, **kwargs: Any) -> FakeNode:
        node = FakeNode(node_id, pool, **kwargs)
        created[node_id] = node
        return node

    monkeypatch.setattr("skyward.actors.pool.pool.Node", factory)
    return created
