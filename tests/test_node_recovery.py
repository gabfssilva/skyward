"""Tests for node failure recovery.

Two scenarios:

1. **Startup failure** — a container dies during provisioning (get_instance
   returns data then container is removed). The instance actor detects SSH
   failure, fires InstanceDied, and the node replaces itself.

2. **Active failure** — a container dies after the cluster is fully
   operational and has already executed tasks. Uses a _KillSwitch that the
   test triggers at a controlled point. The node detects the failure,
   provisions a replacement, and the cluster continues working.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import pytest

import skyward as sky
from skyward.api.model import Cluster, Instance, Offer
from skyward.api.spec import PoolSpec
from skyward.providers.container.cli import run
from skyward.providers.container.config import Container
from skyward.providers.container.provider import ContainerProvider, ContainerSpecific

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.integration,
    pytest.mark.timeout(180),
    pytest.mark.xdist_group("pool"),
]


class _NodeKillerProvider:
    """Wraps ContainerProvider, killing the Nth provisioned container.

    On the first get_instance() call for the marked container, returns valid
    data then immediately kills the container. The instance actor gets the IP,
    moves to connecting, SSH fails (port gone), InstanceDied fires, and the
    node provisions a replacement.
    """

    def __init__(
        self,
        inner: ContainerProvider,
        kill_nth: int,
        binary: str,
        container_prefix: str,
    ) -> None:
        self._inner = inner
        self._kill_nth = kill_nth
        self._bin = binary
        self._prefix = container_prefix
        self._provision_count = 0
        self._kill_id: str | None = None
        self._killed = False

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        async for offer in self._inner.offers(spec):
            yield offer

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[ContainerSpecific]:
        return await self._inner.prepare(spec, offer)

    async def provision(
        self, cluster: Cluster[ContainerSpecific], count: int,
    ) -> tuple[Cluster[ContainerSpecific], Sequence[Instance]]:
        result = await self._inner.provision(cluster, count)
        cluster_out, instances = result
        for inst in instances:
            self._provision_count += 1
            if self._provision_count == self._kill_nth and not self._killed:
                self._kill_id = inst.id
        return result

    async def get_instance(
        self, cluster: Cluster[ContainerSpecific], instance_id: str,
    ) -> tuple[Cluster[ContainerSpecific], Instance | None]:
        if instance_id == self._kill_id and not self._killed:
            self._killed = True
            result = await self._inner.get_instance(cluster, instance_id)
            container_name = f"{self._prefix}-{instance_id}"
            await run(self._bin, "rm", "-f", container_name)
            return result
        return await self._inner.get_instance(cluster, instance_id)

    async def terminate(
        self, cluster: Cluster[ContainerSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[ContainerSpecific]:
        return await self._inner.terminate(cluster, instance_ids)

    async def teardown(
        self, cluster: Cluster[ContainerSpecific],
    ) -> Cluster[ContainerSpecific]:
        return await self._inner.teardown(cluster)


@dataclass(frozen=True, slots=True)
class _NodeKillerContainer(Container):
    """Container config that kills the Nth provisioned container.

    kill_nth: 1-indexed. E.g. kill_nth=4 kills the 4th container provisioned.
    """

    kill_nth: int = 4

    async def create_provider(self) -> _NodeKillerProvider:  # type: ignore[override]
        inner = await ContainerProvider.create(self)
        return _NodeKillerProvider(
            inner,
            kill_nth=self.kill_nth,
            binary=self.binary,
            container_prefix=self.container_prefix or "skyward",
        )


class TestNodeFailureRecovery:
    def test_killed_node_is_replaced_and_cluster_recovers(self) -> None:
        """Pool replaces a dead node and all 5 nodes become operational."""
        with sky.App(console=False), sky.ComputePool(
            provider=_NodeKillerContainer(
                network="skyward",
                container_prefix="skyward-recovery",
                kill_nth=4,
            ),
            nodes=5,
            vcpus=0.5,
            memory_gb=0.5,
            ssh_timeout=10,
            ssh_retry_interval=2,
        ) as pool:

            @sky.compute
            def whoami() -> int:
                info = sky.instance_info()
                return info.node if info else -1

            nodes = whoami() @ pool
            assert sorted(nodes) == [0, 1, 2, 3, 4]

    def test_tasks_execute_on_replacement_node(self) -> None:
        """Tasks dispatched after recovery reach all nodes including the replacement."""
        with sky.App(console=False), sky.ComputePool(
            provider=_NodeKillerContainer(
                network="skyward",
                container_prefix="skyward-recovery-tasks",
                kill_nth=3,
            ),
            nodes=3,
            vcpus=0.5,
            memory_gb=0.5,
            ssh_timeout=10,
            ssh_retry_interval=2,
        ) as pool:

            @sky.compute
            def ping() -> str:
                return "pong"

            results = ping() @ pool
            assert results == ["pong", "pong", "pong"]

            @sky.compute
            def add(a: int, b: int) -> int:
                return a + b

            result = add(2, 3) >> pool
            assert result == 5


# ---------------------------------------------------------------------------
# Kill switch infrastructure — kills containers from test code at any time
# ---------------------------------------------------------------------------


class _KillSwitch:
    """Thread-safe kill switch for controlling container lifecycle from tests.

    The provider registers each provisioned instance. The test can then kill
    any container by its provision index (0-based) at any point.
    """

    def __init__(self, binary: str, container_prefix: str) -> None:
        self._bin = binary
        self._prefix = container_prefix
        self._lock = threading.Lock()
        self._instance_ids: list[str] = []

    def register(self, instance_id: str) -> None:
        with self._lock:
            self._instance_ids.append(instance_id)

    @property
    def registered_count(self) -> int:
        with self._lock:
            return len(self._instance_ids)

    def kill(self, index: int) -> None:
        """Kill the container at provision index (0-based)."""
        with self._lock:
            iid = self._instance_ids[index]
        name = f"{self._prefix}-{iid}"
        subprocess.run(
            [self._bin, "rm", "-f", name],
            check=True,
            capture_output=True,
        )


class _KillSwitchProvider:
    """Wraps ContainerProvider, registers all instances with the kill switch."""

    def __init__(
        self,
        inner: ContainerProvider,
        kill_switch: _KillSwitch,
    ) -> None:
        self._inner = inner
        self._switch = kill_switch

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        async for offer in self._inner.offers(spec):
            yield offer

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[ContainerSpecific]:
        return await self._inner.prepare(spec, offer)

    async def provision(
        self, cluster: Cluster[ContainerSpecific], count: int,
    ) -> tuple[Cluster[ContainerSpecific], Sequence[Instance]]:
        cluster_out, instances = await self._inner.provision(cluster, count)
        for inst in instances:
            self._switch.register(inst.id)
        return cluster_out, instances

    async def get_instance(
        self, cluster: Cluster[ContainerSpecific], instance_id: str,
    ) -> tuple[Cluster[ContainerSpecific], Instance | None]:
        return await self._inner.get_instance(cluster, instance_id)

    async def terminate(
        self, cluster: Cluster[ContainerSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[ContainerSpecific]:
        return await self._inner.terminate(cluster, instance_ids)

    async def teardown(
        self, cluster: Cluster[ContainerSpecific],
    ) -> Cluster[ContainerSpecific]:
        return await self._inner.teardown(cluster)


@dataclass(frozen=True, slots=True)
class _KillSwitchContainer(Container):
    """Container config with an external kill switch for test control.

    The kill switch is created by the test and passed in — after the pool
    is up the test can call kill_switch.kill(index) to remove any container.
    """

    kill_switch: _KillSwitch | None = field(default=None, hash=False)

    async def create_provider(self) -> _KillSwitchProvider:  # type: ignore[override]
        inner = await ContainerProvider.create(self)
        switch = self.kill_switch or _KillSwitch(
            self.binary, self.container_prefix or "skyward",
        )
        return _KillSwitchProvider(inner, switch)


@pytest.mark.timeout(300)
class TestActiveNodeRecovery:
    def test_active_node_killed_and_cluster_recovers(self) -> None:
        """Kill a node after the cluster is fully operational — all nodes
        including the replacement should work after recovery."""
        kill_switch = _KillSwitch(binary="docker", container_prefix="skyward-killswitch")

        with sky.App(console=False), sky.ComputePool(
            provider=_KillSwitchContainer(
                network="skyward",
                container_prefix="skyward-killswitch",
                kill_switch=kill_switch,
            ),
            nodes=5,
            vcpus=0.5,
            memory_gb=0.5,
            ssh_timeout=10,
            ssh_retry_interval=2,
            default_compute_timeout=15,
        ) as pool:

            @sky.compute
            def ping() -> str:
                return "pong"

            # 1. Verify all 5 nodes are operational
            results = ping() @ pool
            assert results == ["pong"] * 5

            # 2. Kill node 3's container (0-indexed)
            kill_switch.kill(3)

            @sky.compute
            def whoami() -> int:
                info = sky.instance_info()
                return info.node if info else -1

            # 3. Retry broadcast until recovery completes and all 5 nodes respond.
            #    First attempt hits the dead node (times out after default_compute_timeout),
            #    which triggers replacement. Subsequent attempts succeed once the
            #    replacement is ready.
            import time

            deadline = time.monotonic() + 120
            nodes: list[int] | tuple[int, ...] = []
            while time.monotonic() < deadline:
                try:
                    nodes = whoami() @ pool
                    if len(nodes) == 5 and all(isinstance(n, int) for n in nodes):
                        break
                except (RuntimeError, TimeoutError):
                    time.sleep(2)
            else:
                pytest.fail(f"Recovery did not complete within 120s, last result: {nodes}")

            assert len(set(nodes)) == 5, f"Expected 5 unique nodes, got {nodes}"
