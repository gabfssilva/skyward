from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable

from skyward.api import Cluster, Instance, PoolSpec


@runtime_checkable
class CloudProvider[C, S](Protocol):
    """Stateless interface for cloud provider operations.

    Implementations hold only immutable config (API keys, region, etc.).
    All lifecycle state lives in the actors that call these methods.
    The Cluster object flows through every call as the shared context.
    """

    @classmethod
    async def create(cls, config: C) -> Self:
        """Create a new instance of the provider.

        Parameters
        ...
        """
        ...

    async def prepare(self, spec: PoolSpec) -> Cluster[S]:
        """Provision cluster-level infrastructure.

        Sets up everything needed before instances can be launched:
        SSH key registration, VPC/security groups (AWS), overlay
        networks (VastAI), GPU type resolution, etc.

        Parameters
        ----------
        spec
            User-defined pool specification (accelerator, nodes, image, etc.).

        Returns
        -------
        Cluster
            Immutable context with cluster ID, SSH credentials, and
            provider-specific infrastructure references.
        """
        ...

    async def provision(self, cluster: Cluster[S], count: int) -> Sequence[Instance]:
        """Launch compute instances.

        Creates instances using the provider's native mechanism:
        EC2 Fleet (AWS), create_pod (RunPod), marketplace offer
        search (VastAI), or create_instance (Verda).

        Parameters
        ----------
        cluster
            Cluster context returned by prepare.
        count
            Number of instances to launch. Providers that support
            batch creation (AWS Fleet, RunPod Instant Cluster) launch
            all at once. Others launch individually.

        Returns
        -------
        Sequence[Instance]
            Launched instances in "provisioning" state. IP may not
            be available yet — poll with get_instance until ready.
        """
        ...

    async def get_instance(self, cluster: Cluster[S], instance_id: str) -> Instance | None:
        """Query current state of an instance.

        Parameters
        ----------
        cluster
            Cluster context returned by prepare.
        instance_id
            Provider-specific instance identifier.

        Returns
        -------
        Instance | None
            Current instance status with IP, SSH port, and metadata
            when available. None if the instance no longer exists.
        """
        ...

    async def terminate(self, instance_ids: tuple[str, ...]) -> None:
        """Terminate one or more instances.

        Parameters
        ----------
        instance_ids
            Provider-specific instance identifiers to destroy.
        """
        ...

    async def teardown(self, cluster: Cluster[S]) -> None:
        """Clean up cluster-level resources.

        Destroys infrastructure created during prepare: overlay
        networks, startup scripts, security groups, etc.

        Parameters
        ----------
        cluster
            Cluster context returned by prepare.
        """
        ...
