from collections.abc import AsyncIterator, Sequence
from typing import Protocol, Self, runtime_checkable

from skyward.api import Cluster, Instance, PoolSpec
from skyward.api.model import Offer


@runtime_checkable
class Provider[C, S](Protocol):
    """Interface for cloud provider operations.

    Every method that receives a Cluster returns an (optionally updated)
    Cluster, allowing the immutable context to evolve across the lifecycle.
    """

    @classmethod
    async def create(cls, config: C) -> Self:
        """Create a new instance of the provider."""
        ...

    def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        """Query available offers matching the spec.

        Yields
        ------
        Offer
            Available machine offers with pricing.
        """
        ...

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[S]:
        """Provision cluster-level infrastructure.

        Sets up everything needed before instances can be launched:
        SSH key registration, VPC/security groups (AWS), overlay
        networks (VastAI), GPU type resolution, etc.

        Returns
        -------
        Cluster[S]
            Immutable cluster context for subsequent calls.
        """
        ...

    async def provision(
        self, cluster: Cluster[S], count: int,
    ) -> tuple[Cluster[S], Sequence[Instance]]:
        """Launch compute instances.

        Returns
        -------
        tuple[Cluster[S], Sequence[Instance]]
            Updated cluster and launched instances in "provisioning" state.
        """
        ...

    async def get_instance(
        self, cluster: Cluster[S], instance_id: str,
    ) -> tuple[Cluster[S], Instance | None]:
        """Query current state of an instance.

        Returns
        -------
        tuple[Cluster[S], Instance | None]
            Updated cluster and current instance status.
        """
        ...

    async def terminate(self, cluster: Cluster[S], instance_ids: tuple[str, ...]) -> Cluster[S]:
        """Terminate one or more instances."""
        ...

    async def teardown(self, cluster: Cluster[S]) -> Cluster[S]:
        """Clean up cluster-level resources."""
        ...


@runtime_checkable
class WarmableProvider[C, S](Provider[C, S], Protocol):
    async def save(self, cluster: Cluster[S]) -> Cluster[S]:
        """Save a prebaked image from the current cluster state."""
        ...
