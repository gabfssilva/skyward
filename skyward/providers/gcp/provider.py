"""GCP Compute Engine provider for Skyward.

Implements the Provider[GCP, GCPSpecific] protocol using sync GCP clients
dispatched to a dedicated thread pool. Provisions instances via instance
templates and bulk_insert for fleet-style provisioning.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from skyward.accelerators import Accelerator
from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import get_local_ssh_key, get_ssh_key_path

from .config import GCP
from .instances import (
    ResolvedMachine,
    default_n1_for_gpus,
    estimate_vram,
    is_guest_attachable,
    is_tpu_accelerator,
    match_accelerator_name,
    parse_builtin_gpu_count,
    resolve_tpu_type,
    select_image_family,
)

log = logger.bind(provider="gcp")


@dataclass(frozen=True, slots=True)
class GCPSpecific:
    """GCP-specific cluster data flowing through Cluster[GCPSpecific]."""

    project: str
    zone: str
    template_name: str
    firewall_rule: str | None
    machine_type: str
    image: str
    uses_guest_accelerators: bool
    accelerator_type: str
    gpu_count: int = 0
    gpu_model: str = ""
    vcpus: int = 0
    memory_gb: float = 0.0
    gpu_vram_gb: int = 0


class GCPProvider(Provider[GCP, GCPSpecific]):
    """Stateless GCP provider. Holds only immutable config + sync clients."""

    def __init__(
        self,
        config: GCP,
        instances_client: object,
        templates_client: object,
        firewalls_client: object,
        machines_client: object,
        accelerators_client: object,
        images_client: object,
        tpu_client: object | None,
        project: str,
        thread_pool: ThreadPoolExecutor,
    ) -> None:
        self._config = config
        self._instances = instances_client
        self._templates = templates_client
        self._firewalls = firewalls_client
        self._machines = machines_client
        self._accelerators = accelerators_client
        self._images = images_client
        self._tpu = tpu_client
        self._project = project
        self._pool = thread_pool

    async def _run[T](self, fn: Callable[..., T], *args: object, **kwargs: object) -> T:
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                self._pool, lambda: fn(*args, **kwargs),
            )
        return await loop.run_in_executor(self._pool, fn, *args)

    @classmethod
    async def create(cls, config: GCP) -> GCPProvider:
        from google.cloud import compute_v1  # type: ignore[reportMissingImports]

        project = _resolve_project(config.project)
        log.info("Resolved GCP project: {project}", project=project)

        thread_pool = ThreadPoolExecutor(
            max_workers=config.thread_pool_size,
            thread_name_prefix="gcp-io",
        )

        tpu_client = None
        try:
            from google.cloud import tpu_v2  # type: ignore[reportMissingImports]

            tpu_client = tpu_v2.TpuClient()
            log.debug("TPU client initialized")
        except ImportError:
            log.debug("google-cloud-tpu not installed, TPU support disabled")

        return cls(
            config=config,
            instances_client=compute_v1.InstancesClient(),
            templates_client=compute_v1.InstanceTemplatesClient(),
            firewalls_client=compute_v1.FirewallsClient(),
            machines_client=compute_v1.MachineTypesClient(),
            accelerators_client=compute_v1.AcceleratorTypesClient(),
            images_client=compute_v1.ImagesClient(),
            tpu_client=tpu_client,
            project=project,
            thread_pool=thread_pool,
        )

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        if spec.accelerator_name and is_tpu_accelerator(spec.accelerator_name):
            async for offer in self._tpu_offers(spec):
                yield offer
            return

        resolved = await self._resolve_machine(spec)
        log.info(
            "Resolved machine: {mt} (gpu={gpu}x{model})",
            mt=resolved.machine_type,
            gpu=resolved.gpu_count,
            model=resolved.gpu_model,
        )

        accelerator = (
            Accelerator(
                name=resolved.gpu_model,
                memory=f"{resolved.gpu_vram_gb}GB" if resolved.gpu_vram_gb else "",
                count=resolved.gpu_count,
            )
            if resolved.gpu_count > 0
            else None
        )

        it = InstanceType(
            name=resolved.machine_type,
            accelerator=accelerator,
            vcpus=resolved.vcpus,
            memory_gb=resolved.memory_gb,
            architecture="x86_64",
            specific=None,
        )

        zone = self._config.zone
        yield Offer(
            id=f"gcp-{zone}-{resolved.machine_type}",
            instance_type=it,
            spot_price=None,
            on_demand_price=None,
            billing_unit="second",
            specific=resolved,
        )

    async def _tpu_offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        if not self._tpu:
            raise RuntimeError(
                "TPU support requires google-cloud-tpu. "
                "Install with: uv add 'skyward[gcp]'"
            )

        tpu_type = resolve_tpu_type(spec.accelerator_name or "")
        log.info("Resolved TPU type: {tpu_type}", tpu_type=tpu_type)

        resolved = ResolvedMachine(
            machine_type=f"tpu-{tpu_type}",
            uses_guest_accelerators=False,
            accelerator_type=tpu_type,
            gpu_count=0,
            gpu_model=tpu_type,
            gpu_vram_gb=0,
            vcpus=0,
            memory_gb=0.0,
        )

        it = InstanceType(
            name=f"tpu-{tpu_type}",
            accelerator=Accelerator(name=tpu_type, count=1),
            vcpus=0,
            memory_gb=0.0,
            architecture="x86_64",
            specific=None,
        )

        zone = self._config.zone
        yield Offer(
            id=f"gcp-{zone}-tpu-{tpu_type}",
            instance_type=it,
            spot_price=None,
            on_demand_price=None,
            billing_unit="second",
            specific=resolved,
        )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[GCPSpecific]:
        log.info("Preparing GCP cluster infrastructure")
        cluster_id = f"gcp-{uuid.uuid4().hex[:8]}"

        _, public_key = get_local_ssh_key()
        ssh_key_path = get_ssh_key_path()

        resolved: ResolvedMachine = offer.specific

        if resolved.machine_type.startswith("tpu-"):
            return await self._prepare_tpu(spec, offer, cluster_id, ssh_key_path)

        return await self._prepare_gpu(spec, offer, cluster_id, ssh_key_path, public_key)

    async def _prepare_tpu(
        self,
        spec: PoolSpec,
        offer: Offer,
        cluster_id: str,
        ssh_key_path: str,
    ) -> Cluster[GCPSpecific]:
        resolved: ResolvedMachine = offer.specific

        return Cluster(
            id=cluster_id,
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="root",
            use_sudo=False,
            shutdown_command="shutdown -h now",
            specific=GCPSpecific(
                project=self._project,
                zone=self._config.zone,
                template_name="",
                firewall_rule=None,
                machine_type=resolved.machine_type,
                image="tpu-ubuntu2204-base",
                uses_guest_accelerators=False,
                accelerator_type=resolved.accelerator_type,
            ),
        )

    async def _prepare_gpu(
        self,
        spec: PoolSpec,
        offer: Offer,
        cluster_id: str,
        ssh_key_path: str,
        public_key: str,
    ) -> Cluster[GCPSpecific]:
        resolved: ResolvedMachine = offer.specific

        has_gpu = resolved.gpu_count > 0
        image = select_image_family(has_gpu=has_gpu)
        use_spot = spec.allocation in ("spot", "spot-if-available")

        firewall_rule = f"skyward-ssh-{cluster_id}"
        await self._ensure_firewall(firewall_rule)

        ttl = spec.ttl or self._config.instance_timeout
        startup_script = _self_destruction_script(ttl, "sudo shutdown -h now")

        template_name = f"skyward-{cluster_id}"
        await self._create_template(
            template_name=template_name,
            machine_type=resolved.machine_type,
            image=image,
            public_key=public_key,
            startup_script=startup_script,
            spot=use_spot,
            uses_guest_accelerators=resolved.uses_guest_accelerators,
            accelerator_type=resolved.accelerator_type,
            gpu_count=resolved.gpu_count,
        )
        log.info("Created instance template: {name}", name=template_name)

        return Cluster(
            id=cluster_id,
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="skyward",
            use_sudo=True,
            shutdown_command="sudo shutdown -h now",
            specific=GCPSpecific(
                project=self._project,
                zone=self._config.zone,
                template_name=template_name,
                firewall_rule=firewall_rule,
                machine_type=resolved.machine_type,
                image=image,
                uses_guest_accelerators=resolved.uses_guest_accelerators,
                accelerator_type=resolved.accelerator_type,
                gpu_count=resolved.gpu_count,
                gpu_model=resolved.gpu_model,
                gpu_vram_gb=resolved.gpu_vram_gb,
                vcpus=resolved.vcpus,
                memory_gb=resolved.memory_gb,
            ),
        )

    async def provision(
        self, cluster: Cluster[GCPSpecific], count: int,
    ) -> tuple[Cluster[GCPSpecific], Sequence[Instance]]:
        if _is_tpu_cluster(cluster):
            return await self._provision_tpu(cluster, count)
        return await self._provision_gpu(cluster, count)

    async def _provision_tpu(
        self, cluster: Cluster[GCPSpecific], count: int,
    ) -> tuple[Cluster[GCPSpecific], Sequence[Instance]]:
        from google.cloud import tpu_v2  # type: ignore[reportMissingImports]

        specific = cluster.specific
        use_spot = cluster.spec.allocation in ("spot", "spot-if-available")
        _, public_key = get_local_ssh_key()

        parent = f"projects/{specific.project}/locations/{specific.zone}"

        async def _create_one(idx: int) -> Instance:
            node_id = f"skyward-{cluster.id}-{idx:04d}"
            node = tpu_v2.Node(
                accelerator_type=specific.accelerator_type,
                runtime_version=specific.image,
                network_config=tpu_v2.NetworkConfig(
                    network=f"global/networks/{self._config.network}",
                    enable_external_ips=True,
                    subnetwork=(
                        f"projects/{specific.project}/regions/"
                        f"{_zone_to_region(specific.zone)}/subnetworks/"
                        f"{self._config.subnet}"
                        if self._config.subnet
                        else ""
                    ),
                ),
                scheduling_config=tpu_v2.SchedulingConfig(
                    spot=use_spot,
                ),
                metadata={
                    "ssh-keys": f"root:{public_key}",
                },
            )

            log.info(
                "Creating TPU node {nid} (type={tpu_type})",
                nid=node_id, tpu_type=specific.accelerator_type,
            )

            operation = await self._run(
                self._tpu.create_node,  # type: ignore[union-attr]
                parent=parent, node=node, node_id=node_id,
            )
            await self._wait_for_operation(operation)

            created = await self._run(
                self._tpu.get_node,  # type: ignore[union-attr]
                name=f"{parent}/nodes/{node_id}",
            )

            ip = _extract_tpu_ip(created)
            log.info(
                "TPU node {nid} created (ip={ip})",
                nid=node_id, ip=ip,
            )

            return Instance(
                id=node_id,
                status="provisioning" if not ip else "provisioned",
                offer=cluster.offer,
                ip=ip,
                spot=use_spot,
                region=_zone_to_region(specific.zone),
            )

        instances = list(await asyncio.gather(
            *(_create_one(i) for i in range(count)),
        ))

        if not instances:
            raise RuntimeError(
                f"Failed to create any TPU nodes (requested {count})",
            )

        log.info("Provisioned {n} TPU nodes", n=len(instances))
        return cluster, instances

    async def _provision_gpu(
        self, cluster: Cluster[GCPSpecific], count: int,
    ) -> tuple[Cluster[GCPSpecific], Sequence[Instance]]:
        from google.cloud import compute_v1  # type: ignore[reportMissingImports]

        specific = cluster.specific
        use_spot = cluster.spec.allocation in ("spot", "spot-if-available")

        template_url = (
            f"projects/{specific.project}"
            f"/global/instanceTemplates/{specific.template_name}"
        )

        name_pattern = f"skyward-{cluster.id}-####"

        log.info(
            "Bulk inserting {count} instances (pattern={pattern})",
            count=count, pattern=name_pattern,
        )

        bulk_resource = compute_v1.BulkInsertInstanceResource(
            source_instance_template=template_url,
            count=count,
            name_pattern=name_pattern,
            min_count=count,
        )

        request = compute_v1.BulkInsertInstanceRequest(
            project=specific.project,
            zone=specific.zone,
            bulk_insert_instance_resource_resource=bulk_resource,
        )

        operation = await self._run(
            self._instances.bulk_insert, request=request,  # type: ignore[union-attr]
        )
        await self._wait_for_operation(operation)
        log.info("Bulk insert operation completed")

        gce_instances = await self._list_cluster_instances(
            specific.project, specific.zone, cluster.id,
        )

        instances = [
            Instance(
                id=str(gce_inst.id),  # type: ignore[union-attr]
                status="provisioning",
                offer=cluster.offer,
                ip=_extract_external_ip(gce_inst),
                spot=use_spot,
                region=_zone_to_region(specific.zone),
            )
            for gce_inst in gce_instances
        ]

        if not instances:
            raise RuntimeError(
                f"Failed to find any instances after bulk insert of {count}",
            )

        log.info("Provisioned {n} instances", n=len(instances))
        return cluster, instances

    async def get_instance(
        self, cluster: Cluster[GCPSpecific], instance_id: str,
    ) -> tuple[Cluster[GCPSpecific], Instance | None]:
        if _is_tpu_cluster(cluster):
            return await self._get_tpu_instance(cluster, instance_id)
        return await self._get_gpu_instance(cluster, instance_id)

    async def _get_tpu_instance(
        self, cluster: Cluster[GCPSpecific], instance_id: str,
    ) -> tuple[Cluster[GCPSpecific], Instance | None]:
        specific = cluster.specific
        name = (
            f"projects/{specific.project}/locations/{specific.zone}"
            f"/nodes/{instance_id}"
        )

        try:
            node = await self._run(
                self._tpu.get_node,  # type: ignore[union-attr]
                name=name,
            )
        except Exception as e:
            log.warning("Failed to get TPU node {nid}: {err}", nid=instance_id, err=e)
            return cluster, None

        state = getattr(node, "state", 0)
        use_spot = cluster.spec.allocation in ("spot", "spot-if-available")

        match state:
            case 5 | 8 | 12:
                return cluster, None
            case 2 if (ip := _extract_tpu_ip(node)):
                return cluster, Instance(
                    id=instance_id,
                    status="provisioned",
                    offer=cluster.offer,
                    ip=ip,
                    spot=use_spot,
                    region=_zone_to_region(specific.zone),
                )
            case _:
                return cluster, Instance(
                    id=instance_id,
                    status="provisioning",
                    offer=cluster.offer,
                    ip=_extract_tpu_ip(node),
                    spot=use_spot,
                    region=_zone_to_region(specific.zone),
                )

    async def _get_gpu_instance(
        self, cluster: Cluster[GCPSpecific], instance_id: str,
    ) -> tuple[Cluster[GCPSpecific], Instance | None]:
        from google.cloud import compute_v1  # type: ignore[reportMissingImports]

        specific = cluster.specific

        try:
            gce_inst = await self._run(
                self._instances.get,  # type: ignore[union-attr]
                request=compute_v1.GetInstanceRequest(
                    project=specific.project,
                    zone=specific.zone,
                    instance=instance_id,
                ),
            )
        except Exception as e:
            log.warning("Failed to get instance {iid}: {err}", iid=instance_id, err=e)
            return cluster, None

        status = getattr(gce_inst, "status", "")

        match status:
            case "TERMINATED" | "STOPPING" | "SUSPENDED":
                return cluster, None
            case "RUNNING" if _extract_external_ip(gce_inst):
                return cluster, _build_gcp_instance(
                    gce_inst, "provisioned", cluster.offer,
                )
            case _:
                return cluster, _build_gcp_instance(
                    gce_inst, "provisioning", cluster.offer,
                )

    async def terminate(
        self, cluster: Cluster[GCPSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[GCPSpecific]:
        if not instance_ids:
            return cluster

        if _is_tpu_cluster(cluster):
            return await self._terminate_tpu(cluster, instance_ids)
        return await self._terminate_gpu(cluster, instance_ids)

    async def _terminate_tpu(
        self, cluster: Cluster[GCPSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[GCPSpecific]:
        specific = cluster.specific

        async def _delete_one(node_id: str) -> None:
            name = (
                f"projects/{specific.project}/locations/{specific.zone}"
                f"/nodes/{node_id}"
            )
            try:
                await self._run(
                    self._tpu.delete_node,  # type: ignore[union-attr]
                    name=name,
                )
                log.info("Requested deletion of TPU node {nid}", nid=node_id)
            except Exception as e:
                log.error(
                    "Failed to delete TPU node {nid}: {err}",
                    nid=node_id, err=e,
                )

        await asyncio.gather(*(_delete_one(nid) for nid in instance_ids))
        return cluster

    async def _terminate_gpu(
        self, cluster: Cluster[GCPSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[GCPSpecific]:
        specific = cluster.specific
        names = await self._resolve_instance_names(
            specific.project, specific.zone, instance_ids, cluster.id,
        )

        async def _delete_one(name: str) -> None:
            from google.cloud import compute_v1  # type: ignore[reportMissingImports]

            try:
                await self._run(
                    self._instances.delete,  # type: ignore[union-attr]
                    request=compute_v1.DeleteInstanceRequest(
                        project=specific.project,
                        zone=specific.zone,
                        instance=name,
                    ),
                )
                log.info("Requested deletion of instance {name}", name=name)
            except Exception as e:
                log.error(
                    "Failed to delete instance {name}: {err}",
                    name=name, err=e,
                )

        await asyncio.gather(*(_delete_one(n) for n in names))
        return cluster

    async def teardown(self, cluster: Cluster[GCPSpecific]) -> Cluster[GCPSpecific]:
        specific = cluster.specific
        log.info("GCP teardown starting for cluster {cid}", cid=cluster.id)

        if not _is_tpu_cluster(cluster):
            await self._teardown_gpu(specific)

        log.info("GCP teardown completed for cluster {cid}", cid=cluster.id)
        self._pool.shutdown(wait=False)
        return cluster

    async def _teardown_gpu(self, specific: GCPSpecific) -> None:
        from google.cloud import compute_v1  # type: ignore[reportMissingImports]

        async def _delete_template() -> None:
            if not specific.template_name:
                return
            try:
                await self._run(
                    self._templates.delete,  # type: ignore[union-attr]
                    request=compute_v1.DeleteInstanceTemplateRequest(
                        project=specific.project,
                        instance_template=specific.template_name,
                    ),
                )
                log.info(
                    "Requested deletion of template: {name}",
                    name=specific.template_name,
                )
            except Exception as e:
                log.error("Failed to delete instance template: {err}", err=e)

        async def _delete_firewall() -> None:
            if not specific.firewall_rule:
                return
            try:
                await self._run(
                    self._firewalls.delete,  # type: ignore[union-attr]
                    request=compute_v1.DeleteFirewallRequest(
                        project=specific.project,
                        firewall=specific.firewall_rule,
                    ),
                )
                log.info(
                    "Requested deletion of firewall rule: {name}",
                    name=specific.firewall_rule,
                )
            except Exception as e:
                log.error("Failed to delete firewall rule: {err}", err=e)

        await asyncio.gather(_delete_template(), _delete_firewall())

    async def _wait_for_operation(self, operation: object) -> None:
        result = getattr(operation, "result", None)
        if callable(result):
            try:
                await self._run(result)
            except Exception as e:
                log.warning("Operation wait returned error: {err}", err=e)

    async def _resolve_machine(self, spec: PoolSpec) -> ResolvedMachine:
        from google.cloud import compute_v1  # type: ignore[reportMissingImports]

        project = self._project
        zone = self._config.zone

        if not spec.accelerator_name:
            min_vcpus = int(spec.vcpus or 2)
            min_memory = spec.memory_gb or 4.0
            machine_type = f"n1-standard-{max(min_vcpus, 2)}"

            pager = self._machines.list(  # type: ignore[union-attr]
                request=compute_v1.ListMachineTypesRequest(
                    project=project, zone=zone,
                ),
            )
            all_machines = await self._run(_collect_pager, pager)

            candidates = [
                mt for mt in all_machines
                if mt.guest_cpus >= min_vcpus and mt.memory_mb / 1024 >= min_memory
            ]

            if best := min(candidates, key=lambda m: m.guest_cpus, default=None):
                machine_type = best.name

            return ResolvedMachine(
                machine_type=machine_type,
                uses_guest_accelerators=False,
                accelerator_type="",
                gpu_count=0,
                gpu_model="",
                gpu_vram_gb=0,
                vcpus=best.guest_cpus if best else min_vcpus,
                memory_gb=round(best.memory_mb / 1024, 1) if best else min_memory,
            )

        pager = self._accelerators.list(  # type: ignore[union-attr]
            request=compute_v1.ListAcceleratorTypesRequest(
                project=project, zone=zone,
            ),
        )
        all_accels = await self._run(_collect_pager, pager)

        gcp_accel_types: list[str] = [at.name for at in all_accels]

        accel_type = match_accelerator_name(spec.accelerator_name, gcp_accel_types)
        gpu_count = spec.accelerator_count or 1
        guest_attachable = is_guest_attachable(accel_type)

        if guest_attachable:
            machine_type = default_n1_for_gpus(gpu_count)
        else:
            pager = self._machines.list(  # type: ignore[union-attr]
                request=compute_v1.ListMachineTypesRequest(
                    project=project, zone=zone,
                ),
            )
            all_machines = await self._run(_collect_pager, pager)

            builtin_candidates = [
                mt for mt in all_machines
                if mt.name.split("-")[0] in ("a2", "a3", "a4", "g2")
                and parse_builtin_gpu_count(mt.name) == gpu_count
            ]

            if builtin_candidates:
                best_builtin = min(builtin_candidates, key=lambda m: m.guest_cpus)  # type: ignore[union-attr]
                machine_type = best_builtin.name  # type: ignore[union-attr]
            else:
                machine_type = default_n1_for_gpus(gpu_count)
                guest_attachable = True

        pager = self._machines.list(  # type: ignore[union-attr]
            request=compute_v1.ListMachineTypesRequest(
                project=project, zone=zone,
                filter=f"name = {machine_type}",
            ),
        )
        machine_details = await self._run(_collect_pager, pager)

        matched = next((mt for mt in machine_details if mt.name == machine_type), None)
        vcpus = matched.guest_cpus if matched else 0
        memory_gb = round(matched.memory_mb / 1024, 1) if matched else 0.0

        gpu_model = re.sub(r"^nvidia-tesla-|^nvidia-", "", accel_type).upper()
        gpu_vram = estimate_vram(accel_type)

        log.debug(
            "Resolved: {mt} accel={accel} guest={guest} gpus={n}",
            mt=machine_type, accel=accel_type, guest=guest_attachable, n=gpu_count,
        )

        return ResolvedMachine(
            machine_type=machine_type,
            uses_guest_accelerators=guest_attachable,
            accelerator_type=accel_type,
            gpu_count=gpu_count,
            gpu_model=gpu_model,
            gpu_vram_gb=gpu_vram,
            vcpus=vcpus,
            memory_gb=memory_gb,
        )

    async def _ensure_firewall(self, rule_name: str) -> None:
        from google.cloud import compute_v1  # type: ignore[reportMissingImports]

        try:
            await self._run(
                self._firewalls.get,  # type: ignore[union-attr]
                request=compute_v1.GetFirewallRequest(
                    project=self._project,
                    firewall=rule_name,
                ),
            )
            log.debug("Firewall rule {name} already exists", name=rule_name)
            return
        except Exception:
            pass

        log.info("Creating firewall rule: {name}", name=rule_name)
        firewall = compute_v1.Firewall(
            name=rule_name,
            direction="INGRESS",
            allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["22"])],
            source_ranges=["0.0.0.0/0"],
            network=f"global/networks/{self._config.network}",
            target_tags=["skyward-node"],
        )

        operation = await self._run(
            self._firewalls.insert,  # type: ignore[union-attr]
            request=compute_v1.InsertFirewallRequest(
                project=self._project,
                firewall_resource=firewall,
            ),
        )
        await self._wait_for_operation(operation)

    async def _create_template(
        self,
        *,
        template_name: str,
        machine_type: str,
        image: str,
        public_key: str,
        startup_script: str,
        spot: bool,
        uses_guest_accelerators: bool,
        accelerator_type: str,
        gpu_count: int,
    ) -> None:
        from google.cloud import compute_v1  # type: ignore[reportMissingImports]

        disk = compute_v1.AttachedDisk(
            auto_delete=True,
            boot=True,
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                source_image=image,
                disk_size_gb=self._config.disk_size_gb,
                disk_type=self._config.disk_type,
            ),
        )

        network_interface = compute_v1.NetworkInterface(
            network=f"global/networks/{self._config.network}",
            access_configs=[
                compute_v1.AccessConfig(
                    name="External NAT",
                    type_="ONE_TO_ONE_NAT",
                ),
            ],
        )
        if self._config.subnet:
            region = _zone_to_region(self._config.zone)
            network_interface.subnetwork = (
                f"regions/{region}/subnetworks/{self._config.subnet}"
            )

        metadata = compute_v1.Metadata(
            items=[
                compute_v1.Items(key="ssh-keys", value=f"skyward:{public_key}"),
                compute_v1.Items(key="startup-script", value=startup_script),
            ],
        )

        if spot:
            scheduling = compute_v1.Scheduling(
                provisioning_model="SPOT",
                instance_termination_action="DELETE",
                on_host_maintenance="TERMINATE",
            )
        else:
            scheduling = compute_v1.Scheduling(
                on_host_maintenance="TERMINATE",
                automatic_restart=True,
            )

        properties = compute_v1.InstanceProperties(
            machine_type=machine_type,
            disks=[disk],
            network_interfaces=[network_interface],
            metadata=metadata,
            scheduling=scheduling,
            tags=compute_v1.Tags(items=["skyward-node"]),
        )

        if uses_guest_accelerators and gpu_count > 0:
            properties.guest_accelerators = [
                compute_v1.AcceleratorConfig(
                    accelerator_type=accelerator_type,
                    accelerator_count=gpu_count,
                ),
            ]

        if self._config.service_account:
            properties.service_accounts = [
                compute_v1.ServiceAccount(
                    email=self._config.service_account,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                ),
            ]

        template = compute_v1.InstanceTemplate(
            name=template_name,
            properties=properties,
        )

        operation = await self._run(
            self._templates.insert,  # type: ignore[union-attr]
            request=compute_v1.InsertInstanceTemplateRequest(
                project=self._project,
                instance_template_resource=template,
            ),
        )
        await self._wait_for_operation(operation)

    async def _list_cluster_instances(
        self, project: str, zone: str, cluster_id: str,
    ) -> list[object]:
        from google.cloud import compute_v1  # type: ignore[reportMissingImports]

        prefix = f"skyward-{cluster_id}-"
        pager = self._instances.list(  # type: ignore[union-attr]
            request=compute_v1.ListInstancesRequest(
                project=project,
                zone=zone,
                filter=f'name:"{prefix}*"',
            ),
        )
        all_instances = await self._run(_collect_pager, pager)
        return [inst for inst in all_instances if inst.name.startswith(prefix)]

    async def _resolve_instance_names(
        self,
        project: str,
        zone: str,
        instance_ids: tuple[str, ...],
        cluster_id: str,
    ) -> list[str]:
        gce_instances = await self._list_cluster_instances(
            project, zone, cluster_id,
        )

        id_to_name = {
            str(inst.id): inst.name  # type: ignore[union-attr]
            for inst in gce_instances
        }
        return [id_to_name.get(iid, iid) for iid in instance_ids]


# =============================================================================
# Pure helper functions (no GCP API calls)
# =============================================================================


def _is_tpu_cluster(cluster: Cluster[GCPSpecific]) -> bool:
    """Check if a cluster is a TPU cluster based on machine type."""
    return cluster.specific.machine_type.startswith("tpu-")


def _extract_tpu_ip(node: object) -> str | None:
    """Extract external IP from a TPU node."""
    endpoints = getattr(node, "network_endpoints", None)
    if not endpoints:
        return None
    for ep in endpoints:
        access_config = getattr(ep, "access_config", None)
        if external_ip := getattr(access_config, "external_ip", None):
            return str(external_ip)
    for ep in endpoints:
        if ip := getattr(ep, "ip_address", None):
            return str(ip)
    return None


def _resolve_project(explicit: str | None) -> str:
    """Resolve GCP project: explicit > env > ADC."""
    if explicit:
        return explicit

    if env_project := os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return env_project

    if env_project := os.environ.get("GCLOUD_PROJECT"):
        return env_project

    try:
        import google.auth  # type: ignore[reportMissingImports]

        _, project = google.auth.default()
        if project:
            return project
    except Exception:
        pass

    raise RuntimeError(
        "No GCP project found. Set GOOGLE_CLOUD_PROJECT env var, "
        "pass project= to GCP(), or configure Application Default Credentials."
    )


def _collect_pager(pager: object) -> list[Any]:
    """Collect all items from a sync GCP pager into a list."""
    return list(pager)  # type: ignore[arg-type]


def _extract_external_ip(instance: object) -> str | None:
    """Extract external IP from a GCE instance object."""
    interfaces = getattr(instance, "network_interfaces", None)
    if not interfaces:
        return None
    for iface in interfaces:
        for config in getattr(iface, "access_configs", []):
            ip = getattr(config, "nat_i_p", None)
            if ip:
                return str(ip)
    return None


def _build_gcp_instance(
    gce_inst: object, status: InstanceStatus, offer: Offer,
) -> Instance:
    """Build a Skyward Instance from a GCE instance."""
    return Instance(
        id=str(gce_inst.id),  # type: ignore[union-attr]
        status=status,
        offer=offer,
        ip=_extract_external_ip(gce_inst),
        spot=_is_spot_instance(gce_inst),
    )


def _is_spot_instance(gce_inst: object) -> bool:
    """Check if a GCE instance is a spot/preemptible instance."""
    scheduling = getattr(gce_inst, "scheduling", None)
    if not scheduling:
        return False
    return getattr(scheduling, "provisioning_model", "") == "SPOT"


def _zone_to_region(zone: str) -> str:
    """Extract region from zone (e.g., 'us-central1-a' -> 'us-central1')."""
    return zone.rsplit("-", 1)[0]


def _self_destruction_script(ttl: int, shutdown_command: str) -> str:
    from skyward.providers.bootstrap.compose import resolve
    from skyward.providers.bootstrap.ops import instance_timeout

    lines = ["#!/bin/bash", "set -e"]
    if ttl:
        lines.append(resolve(instance_timeout(ttl, shutdown_command=shutdown_command)))
    return "\n".join(lines) + "\n"
