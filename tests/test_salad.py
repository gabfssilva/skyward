import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from salad_cloud_sdk.models import (
    ContainerGroupInstance,
    GpuClass,
    GpuClassesList,
    GpuClassPrice,
)
from salad_cloud_sdk.net.transport.api_error import ApiError

from skyward.core.provider import Salad
from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.schemas import ComputeSpec, Image, NodeBounds, Offer, ProviderRef, Spec


def test_salad_factory_carries_account_and_priority() -> None:
    provider = Salad(
        api_key="key",
        organization="acme",
        project="training",
        priority="low",
        country_codes=("us", "ca"),
        image="image:latest",
        storage_gb=100,
        request_timeout=11,
        allocation_timeout=12,
        poll_interval=13,
    )

    assert provider.kind == "salad"
    assert provider.credentials == {"api_key": "key"}
    assert provider.config == {
        "organization": "acme",
        "project": "training",
        "priority": "low",
        "country_codes": ("us", "ca"),
        "image": "image:latest",
        "storage_gb": 100,
        "request_timeout": 11,
        "allocation_timeout": 12,
        "poll_interval": 13,
    }


@pytest.mark.asyncio
async def test_salad_catalog_maps_gpu_class_resources_and_price() -> None:
    from skyward.providers.salad import SaladProvider

    class OrganizationData:
        async def list_gpu_classes(self, organization_name: str) -> GpuClassesList:
            assert organization_name == "acme"
            return GpuClassesList(
                items=[
                    GpuClass(
                        id_="gpu-class-1",
                        name="RTX 4090",
                        gpu_count=1,
                        max_ram=8192,
                        max_storage=107374182400,
                        max_vcpu=8,
                        min_ram=4096,
                        min_storage=53687091200,
                        min_vcpu=2,
                        prices=[GpuClassPrice(price="0.10", priority="low")],
                    )
                ]
            )

    class SDK:
        def __init__(self) -> None:
            self.api_key: tuple[str, str] | None = None
            self.organization_data = OrganizationData()

        def set_api_key(self, api_key: str, api_key_header: str = "Salad-Api-Key") -> None:
            self.api_key = (api_key, api_key_header)

    sdk = SDK()
    with patch("skyward.providers.salad.SaladCloudSdkAsync", return_value=sdk):
        adapter = SaladProvider.create("prv", "salad", {"api_key": "key"}, {"organization": "acme", "project": "training", "priority": "low"})
        offers = [offer async for offer in adapter.offers()]

    assert sdk.api_key == ("key", "Salad-Api-Key")
    assert len(offers) == 1
    offer = offers[0]
    assert offer.id == "gpu-class-1:low"
    assert offer.accelerator == "rtx-4090"
    assert offer.accelerator_count == 1
    assert offer.cpus == 8
    assert offer.memory_gb == 8
    assert offer.disk_gb == 100
    assert offer.on_demand_price == 0.1
    assert offer.spot_price is None
    assert offer.vram == 24
    assert offer.specific["gpu_class_id"] == "gpu-class-1"


@pytest.mark.asyncio
async def test_salad_mints_one_group_per_node_and_dials_it_through_the_gateway() -> None:
    from skyward.providers.salad import SaladProvider

    now = datetime.now(UTC)
    offer = Offer(
        id="gpu-class-1:low",
        provider_id="prv",
        provider_name="salad",
        kind="salad",
        instance_type="RTX 4090",
        accelerator_count=1,
        cpus=8,
        memory_gb=8,
        fetched_at=now,
        expires_at=now,
        accelerator="RTX 4090",
        on_demand_price=0.1,
        specific={"gpu_class_id": "gpu-class-1"},
    )
    spec = ComputeSpec(
        specs=(Spec(provider=ProviderRef(kind="salad"), accelerator="rtx-4090"),),
        nodes=NodeBounds(desired=1),
        image=Image(python="3.12"),
    )

    class ContainerGroups:
        listed = 0
        transient_instance_errors = 0
        created: list[str] = []
        deleted: list[str] = []

        async def create_container_group(self, body: object, organization: str, project: str) -> object:
            assert project == "default"
            self.body = body
            self.created.append(body.name)
            return SimpleNamespace(networking=SimpleNamespace(dns=f"{body.name}.salad.cloud"))

        async def list_container_group_instances(self, organization: str, project: str, group: str) -> object:
            if self.transient_instance_errors:
                self.transient_instance_errors -= 1
                raise ApiError(status=500)
            self.listed += 1
            return SimpleNamespace(
                instances=[
                    ContainerGroupInstance(
                        id_="instance-1",
                        machine_id="machine-1",
                        state="running",
                        update_time="2026-01-01T00:00:00Z",
                        version=1,
                        ready=True,
                    )
                ]
            )

        async def delete_container_group(self, organization: str, project: str, group: str) -> None:
            self.deleted.append(group)

    groups = ContainerGroups()

    class SystemLogs:
        async def get_system_logs(self, organization: str, project: str, group: str) -> object:
            return SimpleNamespace(items=[])

    class SDK:
        def __init__(self) -> None:
            self.api_key: tuple[str, str] | None = None
            self.container_groups = groups
            self.organization_data = SimpleNamespace()
            self.system_logs = SystemLogs()

        def set_api_key(self, api_key: str, api_key_header: str = "Salad-Api-Key") -> None:
            self.api_key = (api_key, api_key_header)

    sdk = SDK()
    config = {
        "organization": "acme",
        "project": "Default",
        "priority": "low",
        "allocation_timeout": 1,
        "poll_interval": 0,
    }

    with patch("skyward.providers.salad.SaladCloudSdkAsync", return_value=sdk):
        adapter = SaladProvider.create("prv", "salad", {"api_key": "key"}, config)
        binding = await adapter.initialize("cmp_1", spec, offer, "on_demand", "ssh-rsa public")
        launched, launched_machines = await adapter.launch(binding, "on_demand", count=1, min_count=1)
        reachable = await adapter.machines(launched)
        await adapter.release(launched)

    group_name = groups.created[0]
    assert group_name.startswith("skyward-cmp-1-")
    assert groups.body.container.resources.gpu_classes == ["gpu-class-1"]
    assert sdk.api_key == ("key", "Salad-Api-Key")
    assert groups.body.container.image == "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04"
    assert groups.body.container.environment_variables == {"PUBLIC_KEY": "ssh-rsa public"}
    assert groups.body.replicas == 1
    assert groups.body.networking.port == 8888
    assert groups.body.networking.auth is False

    command = groups.body.container.command[2]
    assert "apt-get" in command
    assert "printf" in command
    assert "ssh-keygen -A" in command
    assert "ws-l:[::]:8888" in command
    assert "/etc/ssh/sshd_config" not in command
    assert "[ -x /usr/sbin/sshd ]" in command
    assert "[ -x /usr/local/bin/websocat ]" in command

    assert [machine.id for machine in launched_machines] == [group_name]
    assert launched[
        "groups"
    ] == {group_name: f"{group_name}.salad.cloud"}

    machine = reachable[group_name]
    assert machine.state == "running"
    assert machine.host == "127.0.0.1"
    assert machine.port != 0
    assert groups.deleted == [group_name]


@pytest.mark.asyncio
async def test_salad_bridge_carries_bytes_both_ways_over_a_websocket() -> None:
    from websockets.asyncio.server import serve

    from skyward.providers.salad import _Bridge

    async def gateway(socket: object) -> None:
        async for message in socket:
            await socket.send(bytes(reversed(message)))

    async with serve(gateway, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        bridge = _Bridge(f"ws://127.0.0.1:{port}/")
        await bridge.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", bridge.port)
            writer.write(b"skyward")
            await writer.drain()
            assert await reader.readexactly(7) == b"drawyks"
            writer.close()
        finally:
            await bridge.close()


@pytest.mark.asyncio
async def test_salad_treats_missing_group_instances_as_empty() -> None:
    from skyward.providers.salad import SaladProvider

    class ContainerGroups:
        async def list_container_group_instances(self, organization: str, project: str, group: str) -> object:
            raise ApiError(status=404)

    sdk = SimpleNamespace(
        container_groups=ContainerGroups(),
        organization_data=SimpleNamespace(),
        set_api_key=lambda *_args: None,
    )

    with patch("skyward.providers.salad.SaladCloudSdkAsync", return_value=sdk):
        provider = SaladProvider.create(
            "prv",
            "salad",
            {"api_key": "key"},
            {"organization": "acme", "project": "default"},
        )

    assert await provider.machines({"groups": {"missing": "missing.salad.cloud"}}) == {}


@pytest.mark.asyncio
async def test_salad_discards_the_groups_it_created_when_none_gets_allocated() -> None:
    from skyward.providers.salad import SaladProvider

    class ContainerGroups:
        created: list[str] = []
        deleted: list[str] = []

        async def create_container_group(self, body: object, organization: str, project: str) -> object:
            self.created.append(body.name)
            return SimpleNamespace(networking=SimpleNamespace(dns=f"{body.name}.salad.cloud"))

        async def list_container_group_instances(self, organization: str, project: str, group: str) -> object:
            return SimpleNamespace(instances=[])

        async def delete_container_group(self, organization: str, project: str, group: str) -> None:
            self.deleted.append(group)

    groups = ContainerGroups()
    sdk = SimpleNamespace(
        container_groups=groups,
        organization_data=SimpleNamespace(),
        set_api_key=lambda *_args: None,
    )

    with patch("skyward.providers.salad.SaladCloudSdkAsync", return_value=sdk):
        provider = SaladProvider.create(
            "prv",
            "salad",
            {"api_key": "key"},
            {
                "organization": "acme",
                "project": "default",
                "allocation_timeout": 0,
                "poll_interval": 0,
            },
        )

    binding = {
        "compute_id": "cmp_1",
        "gpu_class_id": "gpu-class-1",
        "image": "ubuntu:24.04",
        "public_key": "ssh-rsa public",
        "groups": {},
    }

    with pytest.raises(CapabilityMismatchError):
        await provider.launch(binding, "on_demand", count=1, min_count=1)

    assert groups.created
    assert groups.deleted == groups.created


@pytest.mark.asyncio
async def test_salad_recovers_instances_from_system_logs_when_listing_fails() -> None:
    from skyward.providers.salad import SaladProvider

    class ContainerGroups:
        async def list_container_group_instances(self, organization: str, project: str, group: str) -> object:
            raise ApiError(status=500)

        async def get_container_group_instance(
            self,
            organization: str,
            project: str,
            group: str,
            instance_id: str,
        ) -> object:
            assert instance_id == "instance-1"
            return SimpleNamespace(
                id_="instance-1",
                state="running",
                ready=True,
                ssh_ip="203.0.113.10",
                ssh_port=2222,
            )

    class SystemLogs:
        async def get_system_logs(self, organization: str, project: str, group: str) -> object:
            return SimpleNamespace(items=[SimpleNamespace(instance_id="instance-1")])

    sdk = SimpleNamespace(
        container_groups=ContainerGroups(),
        organization_data=SimpleNamespace(),
        system_logs=SystemLogs(),
        set_api_key=lambda *_args: None,
    )

    with patch("skyward.providers.salad.SaladCloudSdkAsync", return_value=sdk):
        provider = SaladProvider.create(
            "prv",
            "salad",
            {"api_key": "key"},
            {"organization": "acme", "project": "default"},
        )

    machines = await provider.machines({"groups": {"recovered": "recovered.salad.cloud"}})

    assert machines["recovered"].state == "running"
    assert machines["recovered"].host == "127.0.0.1"


@pytest.mark.asyncio
async def test_salad_recovers_from_the_sdk_system_log_parser_bug() -> None:
    from skyward.providers.salad import SaladProvider

    class ContainerGroups:
        async def list_container_group_instances(self, organization: str, project: str, group: str) -> object:
            raise ApiError(status=500)

        async def get_container_group_instance(
            self,
            organization: str,
            project: str,
            group: str,
            instance_id: str,
        ) -> object:
            return SimpleNamespace(
                id_=instance_id,
                state="running",
                ready=True,
                ssh_ip="203.0.113.10",
                ssh_port=2222,
            )

    class SystemLogs:
        _last_response = SimpleNamespace(body={"items": [{"instance_id": "instance-1"}]})

        async def get_system_logs(self, organization: str, project: str, group: str) -> object:
            raise TypeError("SystemLog.__init__() missing required arguments")

    sdk = SimpleNamespace(
        container_groups=ContainerGroups(),
        organization_data=SimpleNamespace(),
        system_logs=SystemLogs(),
        set_api_key=lambda *_args: None,
    )

    with patch("skyward.providers.salad.SaladCloudSdkAsync", return_value=sdk):
        provider = SaladProvider.create(
            "prv",
            "salad",
            {"api_key": "key"},
            {"organization": "acme", "project": "default"},
        )

    machines = await provider.machines({"groups": {"parsed": "parsed.salad.cloud"}})

    assert machines["parsed"].host == "127.0.0.1"


@pytest.mark.asyncio
async def test_salad_terminate_tolerates_a_missing_group() -> None:
    from skyward.providers.salad import SaladProvider

    class ContainerGroups:
        async def delete_container_group(self, organization: str, project: str, group: str) -> None:
            raise ApiError(status=404)

    sdk = SimpleNamespace(
        container_groups=ContainerGroups(),
        organization_data=SimpleNamespace(),
        set_api_key=lambda *_args: None,
    )

    with patch("skyward.providers.salad.SaladCloudSdkAsync", return_value=sdk):
        provider = SaladProvider.create(
            "prv",
            "salad",
            {"api_key": "key"},
            {"organization": "acme", "project": "default"},
        )

    await provider.terminate({"groups": {"missing": "missing.salad.cloud"}}, ("missing",))
