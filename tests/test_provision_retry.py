"""Tests for provision retry logic in the pool actor.

Verifies that when provider.provision() returns fewer instances than
requested, the pool retries until all nodes are provisioned or exhausts
max_provision_attempts.

Uses real Docker containers via a PartialContainerProvider wrapper that
delegates to the real ContainerProvider but limits how many instances
are returned per provision() call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import pytest

import skyward as sky
from skyward.api.model import Cluster, Instance
from skyward.api.spec import PoolSpec
from skyward.providers.container.config import Container
from skyward.providers.container.provider import ContainerProvider, ContainerSpecific

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


class _PartialContainerProvider:
    """Wraps ContainerProvider but limits provision count per call.

    Simulates a cloud provider that can't fulfill the full instance
    request on every attempt — the scenario that triggers provision retry.
    """

    def __init__(
        self,
        inner: ContainerProvider,
        batch_limits: tuple[int, ...],
    ) -> None:
        self._inner = inner
        self._limits = batch_limits
        self._call = 0

    async def offers(self, spec: PoolSpec) -> AsyncIterator[sky.Offer]:
        async for offer in self._inner.offers(spec):
            yield offer

    async def prepare(self, spec: PoolSpec, offer: sky.Offer) -> Cluster[ContainerSpecific]:
        return await self._inner.prepare(spec, offer)

    async def provision(
        self, cluster: Cluster[ContainerSpecific], count: int,
    ) -> tuple[Cluster[ContainerSpecific], Sequence[Instance]]:
        limit = (
            self._limits[self._call]
            if self._call < len(self._limits)
            else count
        )
        self._call += 1
        actual = min(count, limit)
        if actual <= 0:
            return (cluster, ())
        return await self._inner.provision(cluster, actual)

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
class _PartialContainer(Container):
    """Container config that creates a PartialContainerProvider.

    batch_limits controls how many instances provision() returns per call.
    E.g. (1, 1) means first call returns 1, second returns 1, then passthrough.
    """

    batch_limits: tuple[int, ...] = ()

    async def create_provider(self) -> _PartialContainerProvider:  # type: ignore[override]
        inner = await ContainerProvider.create(self)
        return _PartialContainerProvider(inner, self.batch_limits)


class TestProvisionRetryPartial:
    def test_partial_then_remaining(self) -> None:
        """Provider returns 1 of 2 on first try, remaining on retry."""
        with sky.App(console=False), sky.ComputePool(
            provider=_PartialContainer(
                network="skyward",
                container_prefix="skyward-retry-partial",
                batch_limits=(1,),
            ),
            nodes=2,
            vcpus=0.5,
            memory_gb=0.5,
            provision_retry_delay=1.0,
            max_provision_attempts=5,
        ) as pool:
            @sky.compute
            def ping() -> str:
                return "pong"

            results = ping() @ pool
            assert results == ["pong", "pong"]


class TestProvisionRetryExhausted:
    def test_zero_instances_exhausts_retries(self) -> None:
        """Provider returns 0 instances every time, pool raises after max attempts."""
        with (
            pytest.raises(RuntimeError, match="provisioning failed"),
            sky.App(console=False),
            sky.ComputePool(
                provider=_PartialContainer(
                    network="skyward",
                    container_prefix="skyward-retry-exhausted",
                    batch_limits=(0, 0, 0),
                ),
                nodes=2,
                vcpus=0.5,
                memory_gb=0.5,
                provision_retry_delay=0.5,
                max_provision_attempts=3,
            ),
        ):
            pass


class TestProvisionRetryGradual:
    def test_one_per_attempt_accumulates(self) -> None:
        """Provider returns 1 instance per attempt, pool accumulates until complete."""
        with sky.App(console=False), sky.ComputePool(
            provider=_PartialContainer(
                network="skyward",
                container_prefix="skyward-retry-gradual",
                batch_limits=(1, 1, 1),
            ),
            nodes=3,
            vcpus=0.5,
            memory_gb=0.5,
            provision_retry_delay=1.0,
            max_provision_attempts=5,
        ) as pool:
            @sky.compute
            def node_id() -> int:
                info = sky.instance_info()
                return info.node if info else -1

            results = sorted(node_id() @ pool)
            assert results == [0, 1, 2]
