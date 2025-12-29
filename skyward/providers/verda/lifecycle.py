"""Lifecycle operations (provision, setup, shutdown) for Verda instances."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
)
from verda.constants import Actions

from skyward.accelerator import Accelerator
from skyward.callback import emit
from skyward.events import (
    BootstrapCompleted,
    BootstrapStarting,
    Error,
    InfraCreated,
    InfraCreating,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    ProviderName,
    ProvisioningCompleted,
)
from skyward.types import ComputeSpec, ExitedInstance, Instance

if TYPE_CHECKING:
    from skyward.providers.verda import Verda
    from skyward.providers.verda.types import VerdaInstance


def provision(
    provider: Verda,
    compute: ComputeSpec,
) -> tuple[Instance, ...]:
    """Provision Verda GPU instances.

    Creates instances and waits for them to become running.

    Args:
        provider: Verda provider instance.
        compute: Compute specification.

    Returns:
        Tuple of provisioned Instance objects.
    """
    try:
        from skyward.providers.verda.client import get_client
        from skyward.providers.verda.instance_types import (
            get_gpu_image,
            get_standard_image,
            select_instance_type,
        )
        from skyward.providers.verda.ssh import get_or_create_ssh_key
        from skyward.providers.verda.types import VerdaInstance

        cluster_id = str(uuid.uuid4())[:8]

        emit(InfraCreating())

        client = get_client(provider.client_id_resolved, provider.client_secret_resolved)

        # Register SSH key
        ssh_key_info = get_or_create_ssh_key(client)
        object.__setattr__(provider, "_resolved_ssh_key_id", ssh_key_info.id)

        emit(InfraCreated(region=provider.region))

        # Select instance type
        memory_mb = 1024  # Default 1GB
        instance_spec = select_instance_type(
            cpu=1,
            memory_mb=memory_mb,
            accelerator=cast(Accelerator, compute.accelerator),
        )

        # Select image
        if compute.accelerator:
            image = get_gpu_image(
                cast(Accelerator, compute.accelerator), instance_spec.accelerator_count
            )
            username = "ubuntu"
        else:
            image = get_standard_image()
            username = "ubuntu"  # Verda uses ubuntu for all images

        object.__setattr__(provider, "_username", username)

        # Create startup script
        from skyward.providers.verda.bootstrap import create_user_data_script

        user_data = create_user_data_script(compute, provider.instance_timeout)

        # Register startup script with Verda
        script_name = f"skyward-bootstrap-{cluster_id}"
        startup_script = client.startup_scripts.create(name=script_name, script=user_data)
        startup_script_id = startup_script.id

        emit(InstanceLaunching(count=compute.nodes, instance_type=instance_spec.slug, provider=ProviderName.Verda))

        # Create instances
        verda_instances: list[VerdaInstance] = []
        for i in range(compute.nodes):
            instance_name = f"skyward-{cluster_id}-{i}"

            created = client.instances.create(
                instance_type=instance_spec.slug,
                image=image,
                ssh_key_ids=[ssh_key_info.id],
                hostname=instance_name,
                description=f"Skyward managed instance - cluster {cluster_id}",
                startup_script_id=startup_script_id,
                location_code=provider.region,
                is_spot=compute.spot if hasattr(compute, "spot") and compute.spot else False,
            )

            verda_instances.append(
                VerdaInstance(
                    id=created.id,
                    name=instance_name,
                    ip="",
                    status="provisioning",
                )
            )

        # Wait for instances to be running
        _wait_for_running(client, verda_instances, timeout=300)

        # Store instances in provider
        object.__setattr__(provider, "_instances", verda_instances)
        object.__setattr__(provider, "_cluster_id", cluster_id)

        # Build Instance objects
        instances: list[Instance] = []
        for i, vinst in enumerate(verda_instances):
            instance = Instance(
                id=str(vinst.id),
                provider=provider,
                spot=bool(compute.spot) if hasattr(compute, "spot") else False,
                private_ip=vinst.private_ip or vinst.ip,
                public_ip=vinst.ip,
                node=i,
                metadata=frozenset(
                    [
                        ("cluster_id", cluster_id),
                        ("instance_id", vinst.id),
                        ("instance_ip", vinst.ip),
                        ("username", username),
                        ("accelerator_count", str(instance_spec.accelerator_count)),
                    ]
                ),
            )
            instances.append(instance)

            emit(
                InstanceProvisioned(
                    instance_id=str(vinst.id),
                    node=i,
                    spot=bool(compute.spot) if hasattr(compute, "spot") else False,
                    ip=vinst.ip,
                    instance_type=instance_spec.slug,
                    provider=ProviderName.Verda,
                )
            )

        emit(
            ProvisioningCompleted(
                spot=len([i for i in instances if i.spot]),
                on_demand=len([i for i in instances if not i.spot]),
                provider=ProviderName.Verda,
                region=provider.region,
                instances=list(map(lambda inst: inst.id, instances)),
            )
        )

        return tuple(instances)

    except Exception as e:
        emit(Error(message=f"Provision failed: {e}"))
        raise


def setup(
    provider: Verda,
    instances: tuple[Instance, ...],
    compute: ComputeSpec,
) -> None:
    """Setup instances (wait for bootstrap to complete).

    Args:
        provider: Verda provider instance.
        instances: Instances from provision phase.
        compute: Compute specification.
    """
    try:
        from skyward.providers.verda.bootstrap import (
            install_skyward_wheel,
            wait_for_bootstrap,
        )

        for inst in instances:
            emit(BootstrapStarting(instance_id=inst.id))

        # Wait for bootstrap on all instances
        wait_for_bootstrap(instances, timeout=300)

        # Install skyward wheel on all instances
        install_skyward_wheel(instances)

        for inst in instances:
            emit(BootstrapCompleted(instance_id=inst.id))

    except Exception as e:
        emit(Error(message=f"Setup failed: {e}"))
        raise


def shutdown(
    provider: Verda,
    instances: tuple[Instance, ...],
    compute: ComputeSpec,
) -> tuple[ExitedInstance, ...]:
    """Shutdown instances (delete them).

    Args:
        provider: Verda provider instance.
        instances: Instances to shut down.
        compute: Compute specification.

    Returns:
        Tuple of ExitedInstance objects.
    """
    from skyward.providers.verda.client import get_client

    client = get_client(provider.client_id_resolved, provider.client_secret_resolved)

    exited: list[ExitedInstance] = []
    for inst in instances:
        instance_id = inst.get_meta("instance_id") or inst.id
        if instance_id:
            emit(InstanceStopping(instance_id=inst.id))

            try:
                client.instances.action(instance_id, Actions.DELETE)
            except Exception:
                pass  # Ignore delete errors

        exited.append(
            ExitedInstance(
                instance=inst,
                exit_code=0,
                exit_reason="normal",
            )
        )

    object.__setattr__(provider, "_instances", [])

    return tuple(exited)


class _InstancePendingError(Exception):
    """Instance not yet running - retry."""


def _wait_for_running(
    client: Any,
    instances: list[VerdaInstance],
    timeout: float,
) -> None:
    """Wait for all instances to become running and get IPs."""

    def _poll_instance(vinst: VerdaInstance) -> None:
        """Poll a single instance until running."""

        @retry(
            stop=stop_after_delay(timeout),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(_InstancePendingError),
            reraise=True,
        )
        def _check() -> None:
            # Get all instances and find ours
            all_instances = client.instances.get()
            data = None
            for inst in all_instances:
                if inst.id == vinst.id:
                    data = inst
                    break

            if data is None:
                raise _InstancePendingError(f"Instance {vinst.id} not found")

            vinst.status = data.status

            if data.status != "running":
                raise _InstancePendingError(f"Instance {vinst.id} status: {data.status}")

            # Get IP address
            if hasattr(data, "ip") and data.ip:
                vinst.ip = data.ip
            else:
                raise _InstancePendingError(f"Instance {vinst.id} has no IP yet")

        try:
            _check()
        except RetryError as e:
            raise TimeoutError(
                f"Instance {vinst.id} did not become running within {timeout}s"
            ) from e

    for vinst in instances:
        _poll_instance(vinst)
