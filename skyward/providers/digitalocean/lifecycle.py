"""Lifecycle operations (provision, setup, shutdown) for DigitalOcean Droplets."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, cast

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_fixed,
)

from skyward.accelerator import Accelerator
from skyward.events import (
    BootstrapCompleted,
    BootstrapStarting,
    Error,
    EventCallback,
    InfraCreated,
    InfraCreating,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
)
from skyward.types import ComputeSpec, ExitedInstance, Instance

if TYPE_CHECKING:
    from skyward.providers.digitalocean import DigitalOcean
    from skyward.providers.digitalocean.types import Droplet

logger = logging.getLogger("skyward.digitalocean.lifecycle")


def provision(
    provider: DigitalOcean,
    compute: ComputeSpec,
    on_event: EventCallback = None,
) -> tuple[Instance, ...]:
    """Provision DigitalOcean Droplets.

    Creates droplets and waits for them to become active.

    Args:
        provider: DigitalOcean provider instance.
        compute: Compute specification.
        on_event: Optional callback for progress events.

    Returns:
        Tuple of provisioned Instance objects.
    """
    try:
        from skyward.providers.digitalocean.client import get_client
        from skyward.providers.digitalocean.droplet_types import (
            get_gpu_image,
            get_standard_image,
            select_droplet_type,
        )
        from skyward.providers.digitalocean.types import Droplet

        cluster_id = str(uuid.uuid4())[:8]

        if on_event:
            on_event(InfraCreating())

        _ = provider.ssh_fingerprint  # Trigger SSH key registration

        if on_event:
            on_event(InfraCreated(region=provider.region))

        client = get_client(provider.api_token)

        # Select droplet size
        memory_mb = 1024  # Default 1GB
        droplet_spec = select_droplet_type(
            cpu=1,
            memory_mb=memory_mb,
            accelerator=cast(Accelerator, compute.accelerator),
        )

        # Select image
        if compute.accelerator:
            image = get_gpu_image(
                cast(Accelerator, compute.accelerator), droplet_spec.accelerator_count
            )
            username = "ubuntu"
        else:
            image = get_standard_image()
            username = "root"

        object.__setattr__(provider, "_username", username)

        # Create user_data script
        from skyward.providers.digitalocean.bootstrap import create_user_data_script

        user_data = create_user_data_script(compute, provider.instance_timeout)

        if on_event:
            on_event(InstanceLaunching(count=compute.nodes, instance_type=droplet_spec.slug))

        # Create droplets
        droplets: list[Droplet] = []
        for i in range(compute.nodes):
            droplet_name = f"skyward-{cluster_id}-{i}"

            create_data: dict[str, Any] = {
                "name": droplet_name,
                "region": provider.region,
                "size": droplet_spec.slug,
                "image": image,
                "user_data": user_data,
                "tags": ["skyward", f"skyward-cluster-{cluster_id}"],
                "ssh_keys": [provider.ssh_fingerprint],
            }

            resp = client.droplets.create(body=create_data)
            droplet_data = resp["droplet"]
            droplets.append(
                Droplet(
                    id=droplet_data["id"],
                    name=droplet_name,
                    ip="",
                )
            )
            logger.info(f"Created droplet {droplet_name} (id={droplet_data['id']})")

        # Wait for droplets to be active
        _wait_for_active(client, droplets, timeout=300)

        # Store droplets in provider
        object.__setattr__(provider, "_droplets", droplets)

        # Build Instance objects
        instances: list[Instance] = []
        for i, droplet in enumerate(droplets):
            instance = Instance(
                id=str(droplet.id),
                provider=provider,
                spot=False,  # DigitalOcean doesn't have spot
                private_ip=droplet.private_ip or droplet.ip,
                public_ip=droplet.ip,
                node=i,
                metadata=frozenset(
                    [
                        ("cluster_id", cluster_id),
                        ("droplet_id", droplet.id),
                        ("droplet_ip", droplet.ip),
                        ("username", username),
                        ("accelerator_count", str(droplet_spec.accelerator_count)),
                    ]
                ),
            )
            instances.append(instance)

            if on_event:
                on_event(InstanceProvisioned(
                    instance_id=str(droplet.id),
                    node=i,
                    spot=False,
                    ip=droplet.ip,
                    instance_type=droplet_spec.slug,
                ))

        return tuple(instances)

    except Exception as e:
        if on_event:
            on_event(Error(message=f"Provision failed: {e}"))
        raise


def setup(
    provider: DigitalOcean,
    instances: tuple[Instance, ...],
    compute: ComputeSpec,
    on_event: EventCallback = None,
) -> None:
    """Setup instances (wait for bootstrap to complete).

    Args:
        provider: DigitalOcean provider instance.
        instances: Instances from provision phase.
        compute: Compute specification.
        on_event: Optional callback for bootstrap progress events.
    """
    try:
        from skyward.providers.digitalocean.bootstrap import (
            install_skyward_wheel,
            wait_for_bootstrap,
        )

        for inst in instances:
            if on_event:
                on_event(BootstrapStarting(instance_id=inst.id))

        # Wait for bootstrap on all droplets
        wait_for_bootstrap(instances, on_event, timeout=300)

        # Install skyward wheel on all droplets
        install_skyward_wheel(instances)

        for inst in instances:
            if on_event:
                on_event(BootstrapCompleted(instance_id=inst.id))

    except Exception as e:
        if on_event:
            on_event(Error(message=f"Setup failed: {e}"))
        raise


def shutdown(
    provider: DigitalOcean,
    instances: tuple[Instance, ...],
    compute: ComputeSpec,
    on_event: EventCallback = None,
) -> tuple[ExitedInstance, ...]:
    """Shutdown Droplets (destroy them).

    Args:
        provider: DigitalOcean provider instance.
        instances: Instances to shut down.
        compute: Compute specification.
        on_event: Optional callback for shutdown events.

    Returns:
        Tuple of ExitedInstance objects.
    """
    from skyward.providers.digitalocean.client import get_client

    client = get_client(provider.api_token)

    exited: list[ExitedInstance] = []
    for inst in instances:
        droplet_id = inst.get_meta("droplet_id")
        if droplet_id:
            if on_event:
                on_event(InstanceStopping(instance_id=inst.id))

            try:
                client.droplets.destroy(droplet_id=droplet_id)
                logger.info(f"Destroyed droplet {droplet_id}")
            except Exception as e:
                logger.warning(f"Error destroying droplet {droplet_id}: {e}")

        exited.append(
            ExitedInstance(
                instance=inst,
                exit_code=0,
                exit_reason="normal",
            )
        )

    object.__setattr__(provider, "_droplets", [])

    return tuple(exited)


class _DropletPendingError(Exception):
    """Droplet not yet active - retry."""


def _wait_for_active(
    client: Any,
    droplets: list[Droplet],
    timeout: float,
) -> None:
    """Wait for all droplets to become active and get IPs."""

    def _poll_droplet(droplet: Droplet) -> None:
        """Poll a single droplet until active."""

        @retry(
            stop=stop_after_delay(timeout),
            wait=wait_fixed(5),
            retry=retry_if_exception_type(_DropletPendingError),
            reraise=True,
        )
        def _check() -> None:
            resp = client.droplets.get(droplet_id=droplet.id)
            data = resp["droplet"]

            if data["status"] != "active":
                raise _DropletPendingError()

            for network in data["networks"]["v4"]:
                if network["type"] == "public":
                    droplet.ip = network["ip_address"]
                elif network["type"] == "private":
                    droplet.private_ip = network["ip_address"]

            if not droplet.ip:
                raise _DropletPendingError()

            logger.info(f"Droplet {droplet.id} active at {droplet.ip}")

        try:
            _check()
        except RetryError as e:
            raise TimeoutError(
                f"Droplet {droplet.id} did not become active within {timeout}s"
            ) from e

    for droplet in droplets:
        _poll_droplet(droplet)
