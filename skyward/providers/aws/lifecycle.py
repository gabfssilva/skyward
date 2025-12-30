"""Lifecycle operations (provision, setup, shutdown) for AWS EC2 instances."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, cast

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
    from skyward.providers.aws import AWS


def provision(
    provider: AWS,
    compute: ComputeSpec,
) -> tuple[Instance, ...]:
    """Provision EC2 instances.

    Creates AWS infrastructure, acquires instances from pool,
    and waits for them to be ready with SSM connectivity.

    Args:
        provider: AWS provider instance.
        compute: Compute specification.

    Returns:
        Tuple of provisioned Instance objects.
    """
    from skyward.accelerator import Accelerator
    from skyward.image import Image
    from skyward.providers.aws.instance_types import (
        get_instance_spec,
        get_instance_types_for_accelerators,
        select_instance_type,
    )
    from skyward.providers.aws.wheel import ensure_skyward_wheel
    from skyward.accelerator import AcceleratorSpec

    try:
        cluster_id = str(uuid.uuid4())[:8]

        emit(InfraCreating())

        provider._get_resources()

        emit(InfraCreated(region=provider.region))

        # Resolve AMI and username
        ami_id = provider._resolve_ami(compute)
        username = provider._resolve_username(ami_id)

        # Determine if using EC2 Fleet (list of accelerators)
        use_fleet = isinstance(compute.accelerator, list)

        if use_fleet:
            # EC2 Fleet: convert accelerator list to instance types
            accelerator_list = cast(list[str], compute.accelerator)
            instance_types = get_instance_types_for_accelerators(accelerator_list)
            instance_type = instance_types[0]  # Primary type for display
        else:
            # Get accelerator spec (AcceleratorSpec or Accelerator string)
            acc_spec = AcceleratorSpec.from_value(compute.accelerator)
            accelerator_type = acc_spec.accelerator if acc_spec else None
            requested_gpu_count = acc_spec.count if acc_spec else 1

            # Standard: select single instance type
            memory_mb = 512  # Default memory for instance type selection
            instance_type = select_instance_type(
                cpu=1,  # Default
                memory_mb=memory_mb,
                accelerator=cast(Accelerator, accelerator_type),
                accelerator_count=requested_gpu_count,
            )
            instance_types = [instance_type]

        instance_spec = get_instance_spec(instance_type)
        accelerator_count = instance_spec.accelerator_count if instance_spec else 0

        # Create image spec and upload requirements
        image = Image(
            python=compute.python,
            pip=list(compute.pip),
            pip_extra_index_url=compute.pip_extra_index_url,
            apt=list(compute.apt),
            env=dict(compute.env),
        )
        requirements_hash = image.requirements_hash()
        requirements_key = f"requirements/{requirements_hash}.txt"

        if not provider._store_instance.exists(requirements_key):
            provider._store_instance.put(
                requirements_key, image.to_requirements_txt().encode()
            )

        # Ensure skyward wheel is uploaded
        skyward_wheel_key = ensure_skyward_wheel(provider._store_instance)

        # Prepare apt packages and python version
        apt_packages = " ".join(image.apt) if image.apt else ""
        python_version = image.python or "3.13"

        # Acquire instances from pool
        pool = provider._get_pool()

        if use_fleet:
            emit(
                InstanceLaunching(
                    count=compute.nodes,
                    instance_type=", ".join(instance_types),
                    provider=ProviderName.AWS,
                )
            )

            pool_instances = pool.acquire_fleet(
                n=compute.nodes,
                instance_types=instance_types,
                ami_id=ami_id,
                requirements_hash=requirements_hash,
                apt_packages=apt_packages,
                python_version=python_version,
                spot=compute.spot,
                skyward_wheel_key=skyward_wheel_key,
                username=username,
                volumes=compute.volumes,
                pip_extra_index_url=compute.pip_extra_index_url,
                instance_timeout=provider.instance_timeout,
                allocation_strategy=provider.allocation_strategy,
            )
        else:
            emit(InstanceLaunching(count=compute.nodes, instance_type=instance_type, provider=ProviderName.AWS))

            pool_instances = pool.acquire(
                n=compute.nodes,
                instance_type=instance_type,
                ami_id=ami_id,
                requirements_hash=requirements_hash,
                apt_packages=apt_packages,
                python_version=python_version,
                spot=compute.spot,
                skyward_wheel_key=skyward_wheel_key,
                username=username,
                volumes=compute.volumes,
                pip_extra_index_url=compute.pip_extra_index_url,
                instance_timeout=provider.instance_timeout,
            )

        # Convert pool instances to Instance objects
        instances: list[Instance] = []
        for i, pool_inst in enumerate(pool_instances):
            # Use real instance_type from AWS (may differ in Fleet with multiple types)
            real_instance_type = pool_inst.instance_type or instance_type
            real_spec = get_instance_spec(real_instance_type)
            real_accelerator_count = (
                real_spec.accelerator_count if real_spec else accelerator_count
            )

            instance = Instance(
                id=pool_inst.id,
                provider=provider,
                spot=pool_inst.spot,  # Use actual spot status from AWS
                private_ip=pool_inst.private_ip,
                node=i,
                metadata=frozenset(
                    [
                        ("cluster_id", cluster_id),
                        ("region", provider.region),
                        ("instance_type", real_instance_type),
                        ("requirements_hash", requirements_hash),
                        ("accelerator_count", str(real_accelerator_count)),
                    ]
                ),
            )
            instances.append(instance)

            emit(
                InstanceProvisioned(
                    instance_id=instance.id,
                    node=i,
                    spot=instance.spot,
                    instance_type=real_instance_type,
                    provider=ProviderName.AWS
                )
            )

        spot_count = sum(1 for inst in instances if inst.spot)
        emit(
            ProvisioningCompleted(
                spot=spot_count,
                on_demand=len(instances) - spot_count,
                provider=ProviderName.AWS,
                region=provider.region,
                instances=list(map(lambda it: it.id, instances)),
            )
        )

        return tuple(instances)

    except Exception as e:
        emit(Error(message=f"Provision failed: {e}"))
        raise


def setup(
    provider: AWS,
    instances: tuple[Instance, ...],
    compute: ComputeSpec,
) -> None:
    """Setup instances (bootstrap, install dependencies).

    Waits for bootstrap to complete on all instances in parallel.
    Bootstrap is handled by user-data script on first boot.

    Args:
        provider: AWS provider instance.
        instances: Instances from provision phase.
        compute: Compute specification.
    """
    from skyward.conc import for_each_async
    from skyward.providers.aws.bootstrap import wait_for_bootstrap

    try:
        ssm_session = provider._get_ssm_session()

        def bootstrap_instance(inst: Instance) -> None:
            """Wait for bootstrap on a single instance."""
            try:
                emit(BootstrapStarting(instance_id=inst.id))

                wait_for_bootstrap(
                    ssm_session=ssm_session,
                    instance_id=inst.id,
                    verified_instances=provider._verified_instances,
                )

                emit(BootstrapCompleted(instance_id=inst.id))
            except Exception as e:
                emit(
                    Error(
                        message=f"Bootstrap failed on {inst.id}: {e}",
                        instance_id=inst.id,
                    )
                )
                raise

        for_each_async(bootstrap_instance, instances)

    except Exception as e:
        emit(Error(message=f"Setup failed: {e}"))
        raise


def shutdown(
    provider: AWS,
    instances: tuple[Instance, ...],
    compute: ComputeSpec,
) -> tuple[ExitedInstance, ...]:
    """Shutdown instances (stop on-demand, terminate spot).

    Args:
        provider: AWS provider instance.
        instances: Instances to shut down.
        compute: Compute specification.

    Returns:
        Tuple of ExitedInstance objects.
    """
    from skyward.providers.aws.pool import Instance as PoolInstance

    # Convert instances back to pool format
    pool_instances = [
        PoolInstance(
            id=inst.id,
            private_ip=inst.private_ip,
            state="running",
        )
        for inst in instances
    ]

    # Release instances back to pool
    pool = provider._get_pool()
    pool.release(pool_instances, spot=compute.spot)

    exited: list[ExitedInstance] = []
    for inst in instances:
        emit(InstanceStopping(instance_id=inst.id))

        exited.append(
            ExitedInstance(
                instance=inst,
                exit_code=0,
                exit_reason="normal",
            )
        )

    # Cleanup SSM session
    if provider._ssm_session is not None:
        provider._ssm_session.cleanup()

    # Clear verified instances
    provider._verified_instances.clear()

    return tuple(exited)
