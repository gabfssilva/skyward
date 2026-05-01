"""Verda provisioning — partial-failure recovery."""

from __future__ import annotations

from typing import Any

import pytest

from skyward.accelerators import Accelerator
from skyward.core.model import Cluster, InstanceType, Offer
from skyward.core.spec import Nodes, PoolSpec
from skyward.providers.verda.config import Verda
from skyward.providers.verda.provider import VerdaProvider, VerdaSpecific

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class _FakeClient:
    """Stub VerdaClient — get_availability scriptable, create_instance always succeeds."""

    def __init__(self, *, availability_responses: list[dict[str, frozenset[str]]]) -> None:
        self._responses = list(availability_responses)
        self.created: list[str] = []

    async def get_availability(self, is_spot: bool = False) -> dict[str, frozenset[str]]:
        del is_spot
        return self._responses.pop(0) if self._responses else {}

    async def create_instance(self, **kwargs: Any) -> dict[str, Any]:
        iid = f"id-{len(self.created)}"
        self.created.append(iid)
        return {"id": iid, "hostname": kwargs.get("hostname", ""), "status": "provisioning",
                "ip": "", "is_spot": kwargs.get("is_spot", False)}


def _cluster(instance_type: str = "1RTXPRO6000.30V") -> Cluster[VerdaSpecific]:
    accel = Accelerator.from_name("RTX PRO 6000")
    return Cluster(
        id="test",
        status="ready",
        spec=PoolSpec(
            nodes=Nodes(desired=3),
            accelerator=accel,
            region="FIN-03",
            allocation="spot",
        ),
        offer=Offer(
            id="verda-test",
            instance_type=InstanceType(
                name=instance_type,
                accelerator=accel,
                vcpus=30.0,
                memory_gb=120.0,
                architecture="x86_64",
                specific=None,
            ),
            spot_price=0.5,
            on_demand_price=1.0,
            billing_unit="hour",
            specific=None,
        ),
        ssh_key_path="/tmp/key",
        ssh_user="root",
        use_sudo=True,
        shutdown_command="shutdown -h now",
        specific=VerdaSpecific(
            ssh_key_id="ssh-id",
            startup_script_id="script-id",
            instance_type=instance_type,
            os_image="ubuntu-24.04-cuda-12.8-open",
            region="FIN-03",
            hourly_rate=0.5,
            on_demand_rate=1.0,
            gpu_count=1,
            gpu_model="RTX PRO 6000",
            gpu_vram_gb=96,
            vcpus=30,
            memory_gb=120.0,
        ),
    )


@pytest.mark.asyncio
async def test_provision_returns_partial_when_region_runs_out_mid_loop() -> None:
    """Region exhausted on 3rd call must NOT abandon the 2 already-created instances."""
    fake = _FakeClient(availability_responses=[
        {"FIN-03": frozenset({"1RTXPRO6000.30V"})},
        {"FIN-03": frozenset({"1RTXPRO6000.30V"})},
        {},
    ])
    provider = VerdaProvider(Verda(), fake)  # type: ignore[arg-type]

    _, instances = await provider.provision(_cluster(), count=3)

    assert len(instances) == 2
    assert [i.id for i in instances] == ["id-0", "id-1"]
    assert fake.created == ["id-0", "id-1"]


@pytest.mark.asyncio
async def test_provision_raises_when_no_instances_created() -> None:
    fake = _FakeClient(availability_responses=[{}, {}, {}])
    provider = VerdaProvider(Verda(), fake)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="Failed to provision any"):
        await provider.provision(_cluster(), count=3)
