"""A container, provisioned and then made into a worker.

The first test where the two halves meet: the provider hands over a machine, the
node turns it into something that runs Python, and the last assertion is a real
casty call through the tunnel. Everything here is real — the container, the SSH,
the bootstrap, the wheel built from this very checkout — which is the only reason
it is worth running.
"""

import asyncio
import uuid

import asyncssh
import casty
import msgspec
import pytest

from skyward.protocol import codec, frames
from skyward.protocol.schemas import ComputeSpec, Image, NodeBounds, NodeState, ProviderRef, Spec
from skyward.providers.container import ContainerProvider
from skyward.runtime import worker
from skyward.runtime.node import Node
from skyward.runtime.source import Source, resolve

pytestmark = pytest.mark.e2e

IMAGE = Image(python="3.13", packages=("msgspec",), env={"SKYWARD_TEST": "1"})

SPEC = ComputeSpec(
    specs=(Spec(provider=ProviderRef(kind="container"), cpus=2, memory_gb=2),),
    nodes=NodeBounds(desired=1),
    image=IMAGE,
)


@pytest.fixture
def key() -> asyncssh.SSHKey:
    return asyncssh.generate_private_key("ssh-ed25519")


@pytest.fixture(scope="session")
def source() -> Source:
    """The wheel, built once — every node in every test installs the same one."""
    return asyncio.run(resolve("local"))


@pytest.fixture
async def machine(key: asyncssh.SSHKey):
    provider = ContainerProvider.create("prv_container", "local", {}, {})
    offer = [offer async for offer in provider.offers() if offer.cpus == 2 and offer.memory_gb == 2.0][0]
    binding = await provider.initialize(
        f"cmp_{uuid.uuid4().hex[:8]}", SPEC, offer, "on_demand", key.export_public_key().decode(),
    )

    await provider.launch(binding, count=1, min_count=1)
    async with asyncio.timeout(30):
        while not (machines := await provider.machines(binding)):
            await asyncio.sleep(0.5)

    yield next(iter(machines.values()))

    await provider.terminate(binding, tuple(await provider.machines(binding)))
    await provider.release(binding)


async def settled(states: list[NodeState]) -> None:
    async with asyncio.timeout(600):
        while "ready" not in states and "failed" not in states:
            await asyncio.sleep(0.5)


async def test_a_machine_becomes_a_worker(machine, key: asyncssh.SSHKey, source: Source):
    states: list[NodeState] = []
    console: list[str] = []

    node = Node(
        machine,
        compute=f"cmp_{uuid.uuid4().hex[:8]}",
        private_key=key.export_private_key().decode(),
        image=IMAGE,
        source=source,
        listener=lambda state, _: states.append(state),
        output=lambda content, _: console.append(content),
    )
    await node.start()
    await settled(states)

    assert states == ["connecting", "bootstrapping", "ready"], "\n".join(console)
    assert node.tunnel, "a ready node is one the daemon has a way in to"

    installed = await node._ssh.run("/opt/skyward/.venv/bin/python -c 'import msgspec, skyward; print(skyward.__name__)'")
    assert installed.stdout.strip() == "skyward", "the image's packages and skyward itself are in the venv the worker runs in"

    await node.close()


async def test_the_worker_answers_and_runs_what_it_is_sent(machine, key: asyncssh.SSHKey, source: Source):
    """The proof that `ready` means something: a task, over casty, through the tunnel.

    A tunnel that accepts TCP only proves sshd is alive. This is the whole path —
    the daemon's casty client dialling a local port, casty routing to the member,
    the worker unpickling, running, and pickling back.
    """
    states: list[NodeState] = []
    console: list[str] = []

    node = Node(
        machine,
        compute=f"cmp_{uuid.uuid4().hex[:8]}",
        private_key=key.export_private_key().decode(),
        image=IMAGE,
        source=source,
        listener=lambda state, _: states.append(state),
        output=lambda content, task: console.append(f"{task}: {content}"),
    )
    await node.start()
    await settled(states)
    assert states[-1] == "ready", "\n".join(console)

    system = await casty.connect(
        [node.seed],
        address_map=lambda _: f"127.0.0.1:{node.tunnel}",
        cluster_name=node._compute,
    )
    try:
        assert await system.service(worker.Control).ping() == machine.id

        def double(x: int) -> int:
            print("doubling")
            return x * 2

        code = await codec.payload.encode(double)
        args = await codec.payload.encode(((21,), {}))
        reply = await system.service(worker.Worker).run("tsk_1", code, args)

        outcome = msgspec.msgpack.decode(reply, type=frames.Outcome)
        assert isinstance(outcome, frames.Done), outcome
        assert await codec.payload.decode(outcome.value) == 42
    finally:
        await system.close()

    async with asyncio.timeout(30):
        while "tsk_1: doubling" not in console:
            await asyncio.sleep(0.5)

    await node.close()


async def test_a_bootstrap_that_cannot_work_fails_the_node_instead_of_hanging(
    machine, key: asyncssh.SSHKey, source: Source,
):
    """A package that does not exist is the ordinary way a bootstrap dies.

    The failure has to arrive as a state, not as a timeout: a node stuck in
    `bootstrapping` for fifteen minutes is indistinguishable from a slow one, and
    the reconciler cannot replace what it does not know is dead.
    """
    states: list[NodeState] = []
    errors: list[str | None] = []

    node = Node(
        machine,
        compute=f"cmp_{uuid.uuid4().hex[:8]}",
        private_key=key.export_private_key().decode(),
        image=Image(python="3.13", packages=("skyward-does-not-exist-9e3f",)),
        source=source,
        listener=lambda state, error: (states.append(state), errors.append(error)),
        output=lambda _, __: None,
    )
    await node.start()
    await settled(states)

    assert states[-1] == "failed"
    assert "deps" in str(errors[-1]), errors

    await node.close()
