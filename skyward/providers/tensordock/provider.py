"""TensorDock provider implementation.

Provisions GPU VMs on TensorDock's marketplace via their v0 REST API.
Hostnodes are queried for dynamic GPU availability, and VMs are deployed
with cloud-init for SSH key injection.
"""

from __future__ import annotations

import asyncio
import os
import random
import string
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
from skyward.providers.ssh_keys import get_local_ssh_key, get_ssh_key_path

from .client import TensorDockClient, TensorDockError
from .config import TensorDock
from .types import (  # noqa: F401
    HostnodeResponse,
    VmDetails,
    get_gpu_memory_gb,
    get_ssh_port,
    gpu_matches,
    normalize_gpu_name,
)

log = logger.bind(provider="tensordock")


def _get_credentials(config: TensorDock) -> tuple[str, str]:
    """Resolve API key and token from config or environment.

    Returns
    -------
    tuple[str, str]
        (api_key, api_token)

    Raises
    ------
    RuntimeError
        If credentials are not found.
    """
    api_key = config.api_key or os.environ.get("TENSORDOCK_API_KEY")
    api_token = config.api_token or os.environ.get("TENSORDOCK_API_TOKEN")

    if not api_key or not api_token:
        raise RuntimeError(
            "TensorDock credentials required. Set TENSORDOCK_API_KEY and "
            "TENSORDOCK_API_TOKEN environment variables or pass them to "
            "TensorDock(api_key=..., api_token=...)."
        )
    return api_key, api_token


def _build_cloudinit(ssh_public_key: str) -> str:
    """Generate cloud-init YAML to inject SSH key for user 'user'."""
    return (
        "#cloud-config\n"
        "users:\n"
        "  - name: user\n"
        "    sudo: ALL=(ALL) NOPASSWD:ALL\n"
        "    shell: /bin/bash\n"
        "    ssh_authorized_keys:\n"
        f"      - {ssh_public_key}\n"
    )


@dataclass(frozen=True, slots=True)
class TensorDockOfferData:
    """Carried in Offer.specific -- hostnode details for provisioning."""

    hostnode_id: str
    gpu_model: str
    gpu_count: int
    vcpus: int
    ram_gb: int
    hourly_rate: float


@dataclass(frozen=True, slots=True)
class TensorDockSpecific:
    """TensorDock-specific cluster data flowing through Cluster[TensorDockSpecific]."""

    ssh_public_key: str
    password: str
    location: str | None = None
    operating_system: str = "Ubuntu 22.04 LTS"


def _filter_hostnodes(
    hostnodes: dict[str, HostnodeResponse],
    spec: PoolSpec,
    config: TensorDock,
) -> list[tuple[str, str, int, int, int, float]]:
    """Filter and rank hostnodes by GPU match, availability, and price.

    Returns a list of (hostnode_id, gpu_model, gpu_count, vcpus, ram_gb, hourly_rate)
    sorted by price ascending.
    """
    gpu_count = spec.accelerator_count or 1
    candidates: list[tuple[str, str, int, int, int, float]] = []

    for hid, node in hostnodes.items():
        if config.location:
            loc = node.get("location", {})
            country = loc.get("country", "").lower()
            if country != config.location.lower():
                continue

        specs = node.get("specs", {})
        pricing = node.get("pricing")
        if not pricing:
            continue

        gpu_specs = specs.get("gpu", {})
        gpu_pricing = pricing.get("gpu", {})

        for gpu_model, gpu_info in gpu_specs.items():
            available = gpu_info.get("amount", 0)
            if available < gpu_count:
                continue

            if spec.accelerator_name and not gpu_matches(gpu_model, spec.accelerator_name):
                continue

            gpu_price = gpu_pricing.get(gpu_model)
            if gpu_price is None:
                continue

            ram_specs = specs.get("ram", {})
            cpu_specs = specs.get("cpu", {})
            ram_available = sum(ram_specs.values()) if ram_specs else 0
            cpu_available = sum(cpu_specs.values()) if cpu_specs else 0

            min_ram = config.min_ram_gb or 16
            min_vcpus = config.min_vcpus or 4
            ram_gb = int(max(min_ram, spec.memory_gb or 0))
            vcpus = max(min_vcpus, int(spec.vcpus or 0))

            if ram_available < ram_gb or cpu_available < vcpus:
                continue

            ram_price = pricing.get("ram", 0.0)
            cpu_price = pricing.get("cpu", 0.0)
            storage_price = pricing.get("storage", 0.0)

            hourly_rate = (
                gpu_price * gpu_count
                + ram_price * ram_gb
                + cpu_price * vcpus
                + storage_price * config.storage_gb
            )

            candidates.append((hid, gpu_model, gpu_count, vcpus, ram_gb, hourly_rate))

    if spec.max_hourly_cost:
        max_per_instance = spec.max_hourly_cost / spec.nodes
        candidates = [c for c in candidates if c[5] <= max_per_instance]

    candidates.sort(key=lambda c: c[5])
    return candidates


class TensorDockProvider(Provider[TensorDock, TensorDockSpecific]):
    """Stateless TensorDock provider. Holds only immutable config."""

    def __init__(self, config: TensorDock) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: TensorDock) -> TensorDockProvider:
        return cls(config)

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        api_key, api_token = _get_credentials(self._config)

        async with TensorDockClient(
            api_key, api_token, timeout=self._config.request_timeout,
        ) as client:
            hostnodes = await client.list_hostnodes()

        candidates = _filter_hostnodes(hostnodes, spec, self._config)

        if not candidates:
            log.debug("No matching hostnodes found")
            return

        for hid, gpu_model, gpu_count, vcpus, ram_gb, hourly_rate in candidates:
            display_name = normalize_gpu_name(gpu_model)
            memory_gb = get_gpu_memory_gb(gpu_model)
            accel = Accelerator(
                name=display_name,
                memory=f"{memory_gb}GB" if memory_gb else "",
                count=gpu_count,
            )

            it = InstanceType(
                name=display_name,
                accelerator=accel,
                vcpus=float(vcpus),
                memory_gb=float(ram_gb),
                architecture="x86_64",
                specific=None,
            )

            yield Offer(
                id=f"td-{hid}-{gpu_model}",
                instance_type=it,
                spot_price=None,
                on_demand_price=hourly_rate,
                billing_unit="second",
                specific=TensorDockOfferData(
                    hostnode_id=hid,
                    gpu_model=gpu_model,
                    gpu_count=gpu_count,
                    vcpus=vcpus,
                    ram_gb=ram_gb,
                    hourly_rate=hourly_rate,
                ),
            )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[TensorDockSpecific]:
        api_key, api_token = _get_credentials(self._config)
        ssh_key_path = get_ssh_key_path()
        _, ssh_public_key = get_local_ssh_key()

        async with TensorDockClient(
            api_key, api_token, timeout=self._config.request_timeout,
        ) as client:
            if not await client.test_auth():
                raise RuntimeError("TensorDock authentication failed — check credentials")

        password = "".join(random.choices(string.ascii_letters + string.digits, k=24))

        return Cluster(
            id=f"tensordock-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="user",
            use_sudo=True,
            shutdown_command="sudo shutdown -h now",
            specific=TensorDockSpecific(
                ssh_public_key=ssh_public_key,
                password=password,
                location=self._config.location,
                operating_system=self._config.operating_system,
            ),
        )

    async def provision(
        self, cluster: Cluster[TensorDockSpecific], count: int,
    ) -> tuple[Cluster[TensorDockSpecific], Sequence[Instance]]:
        specific = cluster.specific
        api_key, api_token = _get_credentials(self._config)

        cloudinit = _build_cloudinit(specific.ssh_public_key)
        instances: list[Instance] = []

        async with TensorDockClient(
            api_key, api_token, timeout=self._config.request_timeout,
        ) as client:
            hostnodes = await client.list_hostnodes()
            candidates = _filter_hostnodes(hostnodes, cluster.spec, self._config)

            if not candidates:
                log.error("No hostnodes available for provisioning")
                return cluster, instances

            for i in range(count):
                instance = await _try_deploy_from_candidates(
                    client, self._config, cluster, candidates, cloudinit, i,
                )
                if instance is None:
                    log.error("All hostnodes failed for instance {i}/{count}", i=i + 1, count=count)
                    continue
                instances.append(instance)

        return cluster, instances

    async def get_instance(
        self, cluster: Cluster[TensorDockSpecific], instance_id: str,
    ) -> tuple[Cluster[TensorDockSpecific], Instance | None]:
        api_key, api_token = _get_credentials(self._config)

        async with TensorDockClient(
            api_key, api_token, timeout=self._config.request_timeout,
        ) as client:
            vm = await client.get_vm(instance_id)

        if not vm:
            return cluster, None

        status_str = (vm.get("status") or "").lower()
        ip = vm.get("ip") or ""

        log.debug(
            "VM {vid} status={status}, ip={ip}",
            vid=instance_id, status=status_str, ip=ip,
        )

        match status_str:
            case "stopped" | "error" | "terminated" | "deleted":
                return cluster, None
            case "running" if ip:
                return cluster, _build_instance(
                    instance_id, vm, "provisioned", cluster.offer,
                )
            case _:
                return cluster, _build_instance(
                    instance_id, vm, "provisioning", cluster.offer,
                )

    async def terminate(
        self, cluster: Cluster[TensorDockSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[TensorDockSpecific]:
        if not instance_ids:
            return cluster

        api_key, api_token = _get_credentials(self._config)
        async with TensorDockClient(
            api_key, api_token, timeout=self._config.request_timeout,
        ) as client:
            async def _delete(iid: str) -> None:
                try:
                    await client.delete_vm(iid)
                except Exception as e:
                    log.error("Failed to delete VM {iid}: {err}", iid=iid, err=e)

            await asyncio.gather(*(_delete(iid) for iid in instance_ids))
        return cluster

    async def teardown(
        self, cluster: Cluster[TensorDockSpecific],
    ) -> Cluster[TensorDockSpecific]:
        return cluster


def _build_instance(
    instance_id: str,
    vm: VmDetails,
    status: InstanceStatus,
    offer: Offer,
) -> Instance:
    """Build an Instance from TensorDock VM details."""
    ip = vm.get("ip") or ""
    ssh_port = get_ssh_port(vm)

    return Instance(
        id=instance_id,
        status=status,
        offer=offer,
        ip=ip,
        private_ip=ip or None,
        ssh_port=ssh_port,
        spot=False,
    )


async def _try_deploy_from_candidates(
    client: TensorDockClient,
    config: TensorDock,
    cluster: Cluster[TensorDockSpecific],
    candidates: list[tuple[str, str, int, int, int, float]],
    cloudinit: str,
    instance_index: int,
) -> Instance | None:
    """Try deploying a VM on each candidate hostnode until one succeeds."""
    specific = cluster.specific
    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    vm_name = f"skyward-{cluster.id}-{instance_index}-{suffix}"

    for idx, (hid, gpu_model, gpu_count, vcpus, ram_gb, hourly_rate) in enumerate(candidates):
        log.debug(
            "Trying hostnode {i}/{total}: id={hid}, gpu={gpu}",
            i=idx + 1, total=len(candidates), hid=hid, gpu=gpu_model,
        )

        try:
            result = await client.deploy_vm(
                hostnode=hid,
                gpu_model=gpu_model,
                gpu_count=gpu_count,
                vcpus=vcpus,
                ram=ram_gb,
                storage=config.storage_gb,
                password=specific.password,
                name=vm_name,
                operating_system=specific.operating_system,
                internal_ports="22,25520",
                external_ports="22,25520",
                cloudinit_script=cloudinit,
            )

            server_id = result.get("server", "")
            if not server_id:
                log.warning("Deploy returned no server ID: {r}", r=result)
                continue

            ip = result.get("ip", "")
            port_forwards = result.get("port_forwards", {})
            ssh_port = port_forwards.get("22", 22)

            display_name = normalize_gpu_name(gpu_model)
            memory_gb = get_gpu_memory_gb(gpu_model)
            accel = Accelerator(
                name=display_name,
                memory=f"{memory_gb}GB" if memory_gb else "",
                count=gpu_count,
            )
            it = InstanceType(
                name=display_name,
                accelerator=accel,
                vcpus=float(vcpus),
                memory_gb=float(ram_gb),
                architecture="x86_64",
                specific=None,
            )
            offer = Offer(
                id=f"td-{hid}-{gpu_model}",
                instance_type=it,
                spot_price=None,
                on_demand_price=hourly_rate,
                billing_unit="second",
                specific=TensorDockOfferData(
                    hostnode_id=hid,
                    gpu_model=gpu_model,
                    gpu_count=gpu_count,
                    vcpus=vcpus,
                    ram_gb=ram_gb,
                    hourly_rate=hourly_rate,
                ),
            )

            log.info(
                "Deployed VM {name} on hostnode {hid}: id={sid}, ip={ip}:{port}",
                name=vm_name, hid=hid, sid=server_id, ip=ip, port=ssh_port,
            )

            return Instance(
                id=server_id,
                status="provisioning",
                offer=offer,
                ip=ip,
                private_ip=ip or None,
                ssh_port=ssh_port,
                spot=False,
            )
        except TensorDockError as e:
            log.warning(
                "Hostnode {i}/{total} failed: {err}",
                i=idx + 1, total=len(candidates), err=e,
            )

    return None
