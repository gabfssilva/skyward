"""InstancePool - manages a pool of cloud instances with lifecycle operations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyward.core.callback import emit
from skyward.core.events import (
    BootstrapCompleted,
    BootstrapStarting,
    InstanceStopping,
    ProviderName,
    ProvisioningCompleted,
    ProvisioningStarted,
)
from skyward.utils.conc import for_each_async
from skyward.providers.common import make_provisioned

if TYPE_CHECKING:
    from skyward.types import ComputeSpec, ExitedInstance, Instance, Provider

@dataclass
class InstancePool:
    """Manages a pool of cloud instances with lifecycle operations.

    InstancePool holds provider, compute spec, and instances.
    Use as a context manager for automatic cleanup.

    Attributes:
        provider: Cloud provider instance (AWS, DigitalOcean, etc.).
        compute: Compute specification (nodes, accelerator, image, etc.).
        instances: Tuple of provisioned instances (empty until provision()).
    """

    provider: Provider
    compute: ComputeSpec
    instances: tuple[Instance, ...] = ()

    def __iter__(self) -> Iterator[Instance]:
        """Iterate over current instances."""
        return iter(self.instances)

    def __len__(self) -> int:
        """Number of instances."""
        return len(self.instances)

    def provision(self) -> None:
        """Provision instances and store internally.

        Calls provider.provision() and emits provisioning events:
        - ProvisioningStarted: Before provisioning begins
        - InstanceProvisioned: For each instance (emitted by provider)
        - ProvisioningCompleted: After all instances are provisioned

        Example:
            pool.provision()
            pool.setup()
        """
        emit(ProvisioningStarted())

        self.instances = self.provider.provision(self.compute)

        provider_name = _get_provider_name(self.provider)
        provisioned_instances = tuple(
            make_provisioned(inst, provider_name)
            for inst in self.instances
        )

        region = self.instances[0].get_meta("region", "unknown") if self.instances else "unknown"
        emit(
            ProvisioningCompleted(
                instances=provisioned_instances,
                provider=provider_name,
                region=region,
            )
        )

    def setup(self, timeout: int = 300) -> None:
        """Bootstrap all instances in parallel.

        Waits for SSH connectivity and installs skyward on each instance.
        Emits bootstrap events for progress tracking.

        Args:
            timeout: Maximum seconds to wait for SSH per instance.

        Emits:
            BootstrapStarting: When bootstrap begins on each instance.
            BootstrapCompleted: When bootstrap finishes on each instance.
        """
        if not self.instances:
            return

        provider_name = _get_provider_name(self.provider)
        use_systemd = provider_name != ProviderName.VastAI

        def bootstrap_instance(inst: Instance) -> None:
            provisioned = make_provisioned(inst, provider_name)
            emit(BootstrapStarting(instance=provisioned))

            # Wait for SSH port to be ready
            inst.wait_for_ssh(timeout)

            # Wait for bootstrap script to complete (.ready file)
            inst.wait_for_ready(timeout)

            # Install skyward wheel (skips if not "local" source)
            inst.install_skyward(self.compute, use_systemd=use_systemd)

            emit(BootstrapCompleted(instance=provisioned))

        for_each_async(bootstrap_instance, self.instances)

    def shutdown(self) -> tuple[ExitedInstance, ...]:
        """Shutdown all instances.

        Destroys each instance and returns ExitedInstance objects.

        Returns:
            Tuple of ExitedInstance objects with exit information.

        Emits:
            InstanceStopping: For each instance being stopped.
        """
        from skyward.types import ExitedInstance

        if not self.instances:
            return ()

        provider_name = _get_provider_name(self.provider)
        exited: list[ExitedInstance] = []

        for inst in self.instances:
            provisioned = make_provisioned(inst, provider_name)
            emit(InstanceStopping(instance=provisioned))
            inst.destroy()
            exited.append(
                ExitedInstance(
                    instance=inst,
                    exit_code=0,
                    exit_reason="shutdown",
                )
            )

        return tuple(exited)

    def __enter__(self) -> InstancePool:
        """Enter context manager.

        Provisions and sets up instances automatically.

        Returns:
            Self with instances provisioned and bootstrapped.
        """
        self.provision()
        self.setup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager.

        Shuts down all instances, regardless of exceptions.
        """
        self.shutdown()


def _get_provider_name(provider: Provider) -> ProviderName:
    """Map provider.name to ProviderName enum."""
    match provider.name:
        case "aws":
            return ProviderName.AWS
        case "digitalocean":
            return ProviderName.DigitalOcean
        case "verda":
            return ProviderName.Verda
        case "vastai":
            return ProviderName.VastAI
        case _:
            return ProviderName.AWS  # Default fallback

__all__ = ["InstancePool"]
