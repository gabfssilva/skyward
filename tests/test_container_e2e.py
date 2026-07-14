"""The container provider, against a real Docker daemon.

The only provider whose machines are free, and therefore the only one where the
provisioning half of the contract can be proven rather than mocked.
"""

import asyncio
import uuid

import pytest

from skyward.protocol.schemas import ComputeSpec, Image, NodeBounds, ProviderRef, Spec
from skyward.providers.container import ContainerProvider

pytestmark = pytest.mark.e2e

PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBSkywardTestKeyNotUsedForAnythingReal test"

SPEC = ComputeSpec(
    specs=(Spec(provider=ProviderRef(kind="container"), cpus=1, memory_gb=1),),
    nodes=NodeBounds(desired=2),
    image=Image(python="3.13"),
)


@pytest.fixture
def provider() -> ContainerProvider:
    return ContainerProvider.create("prv_container", "local", {}, {})


@pytest.fixture
async def binding(provider: ContainerProvider):
    offer = [offer async for offer in provider.offers() if offer.cpus == 1 and offer.memory_gb == 1.0][0]
    binding = await provider.initialize(f"cmp_{uuid.uuid4().hex[:8]}", SPEC, offer, "on_demand", PUBLIC_KEY)
    yield binding
    await provider.terminate(binding, tuple(await provider.machines(binding)))
    await provider.release(binding)


async def running(provider: ContainerProvider, binding, expected: int, timeout: float = 30.0):
    async with asyncio.timeout(timeout):
        while len(machines := await provider.machines(binding)) != expected:
            await asyncio.sleep(0.5)
        return machines


async def test_the_catalog_is_free_cpu_only_hardware(provider: ContainerProvider):
    offers = [offer async for offer in provider.offers()]

    assert offers
    assert all(offer.accelerator is None and offer.accelerator_count == 0 for offer in offers)
    assert all(offer.price == 0.0 for offer in offers)


async def test_a_launched_machine_becomes_reachable(provider: ContainerProvider, binding):
    _, launched = await provider.launch(binding, count=2, min_count=2)
    assert len(launched) == 2

    machines = await running(provider, binding, 2)

    assert set(machines) == {machine.id for machine in launched}, "launch must return the ids machines() reports"
    assert all(machine.host == "127.0.0.1" for machine in machines.values())
    assert all(machine.port != 22 for machine in machines.values()), "each machine gets its own published port"
    assert all(machine.private_host for machine in machines.values()), "peers reach each other on the bridge"


async def test_machines_are_found_by_tag_not_by_memory(provider: ContainerProvider, binding):
    """A machine launched by a process that died is still this compute's machine.

    Nothing here remembers what launch returned — the query goes to Docker, and
    the label is what makes the container findable. That is what keeps a crash
    between the launch and the commit from leaking a paid instance.
    """
    await provider.launch(binding, count=1, min_count=1)
    machines = await running(provider, binding, 1)

    forgetful = ContainerProvider.create("prv_container", "local", {}, {})

    assert set(await forgetful.machines(binding)) == set(machines)


async def test_terminating_one_machine_leaves_the_rest(provider: ContainerProvider, binding):
    _, launched = await provider.launch(binding, count=2, min_count=2)
    await running(provider, binding, 2)

    await provider.terminate(binding, (launched[0].id,))
    survivors = await running(provider, binding, 1)

    assert set(survivors) == {launched[1].id}


async def test_terminating_a_machine_that_is_already_gone_is_not_an_error(provider: ContainerProvider, binding):
    _, launched = await provider.launch(binding, count=1, min_count=1)
    await running(provider, binding, 1)

    await provider.terminate(binding, (launched[0].id,))
    await provider.terminate(binding, (launched[0].id,))

    assert await provider.machines(binding) == {}


async def test_initialize_tolerates_finding_its_own_work_already_done(provider: ContainerProvider, binding):
    """The daemon can die between creating the network and committing the binding.

    On the way back up it calls initialize again, with no binding, and has to get
    the same infrastructure back rather than an "already exists" error.
    """
    offer = [offer async for offer in provider.offers() if offer.cpus == 1 and offer.memory_gb == 1.0][0]

    again = await provider.initialize(binding["compute_id"], SPEC, offer, "on_demand", PUBLIC_KEY)

    assert again == binding
