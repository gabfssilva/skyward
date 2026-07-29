"""Three machines, one cluster, one client that can reach any of them.

The node tests prove a machine can be made to run Python. This proves the thing
a compute actually is: nodes that found each other over the private network, and
a daemon that reaches every one of them through its own tunnel without any of
them being special.

The client is given **one** seed on purpose. If it sees three members afterwards,
the other two joined through the network and not through us — which is the only
evidence that separates a cluster from three machines that happen to be running
the same program.
"""

import asyncio
import uuid

import asyncssh
import casty
import msgspec
import pytest

from skyward.shared.provider import Machine
from skyward.shared import codec, frames
from skyward.shared.schemas import ComputeSpec, Image, NodeBounds, NodeState, ProviderRef, Spec
from skyward.providers.container import ContainerProvider
from skyward.worker import worker
from skyward.server.application.node import Node
from skyward.server.application.source import Source, resolve

pytestmark = pytest.mark.e2e

NODES = 3

IMAGE = Image(python="3.13")

SPEC = ComputeSpec(
    specs=(Spec(provider=ProviderRef(kind="container"), cpus=1, memory_gb=1),),
    nodes=NodeBounds(desired=NODES),
    image=IMAGE,
)


@pytest.fixture
def key() -> asyncssh.SSHKey:
    return asyncssh.generate_private_key("ssh-ed25519")


@pytest.fixture(scope="session")
def source() -> Source:
    return asyncio.run(resolve("local"))


@pytest.fixture
async def compute(key: asyncssh.SSHKey):
    compute = f"cmp_{uuid.uuid4().hex[:8]}"
    provider = ContainerProvider.create("prv_container", "local", {}, {})
    offer = [offer async for offer in provider.offers() if offer.cpus == 1 and offer.memory_gb == 1.0][0]
    binding = await provider.initialize(compute, SPEC, offer, "on_demand", key.export_public_key().decode())

    await provider.launch(binding, "on_demand", count=NODES, min_count=NODES)
    async with asyncio.timeout(60):
        while len(machines := await provider.machines(binding)) < NODES:
            await asyncio.sleep(0.5)

    yield compute, tuple(machines.values())

    await provider.terminate(binding, tuple(await provider.machines(binding)))
    await provider.release(binding)


async def cluster(
    compute: str, machines: tuple[Machine, ...], key: asyncssh.SSHKey, source: Source,
) -> tuple[Node, ...]:
    """Every node, brought up at once, all but the first joining the first.

    They are started in parallel deliberately: the seed is not up when the others
    go looking for it, which is the ordinary case on a real provider and the one
    the worker's `reachable` exists for.
    """
    states: dict[str, list[NodeState]] = {machine.id: [] for machine in machines}
    seed = f"{machines[0].private_host}:{worker.PORT}"

    nodes = tuple(
        Node(
            machine,
            compute=compute,
            private_key=key.export_private_key().decode(),
            image=IMAGE,
            source=source,
            seeds=() if rank == 0 else (seed,),
            listener=lambda state, _, node=machine.id: states[node].append(state),
            output=lambda _, __: None,
            sample=lambda *_: None,
            phase=lambda *_: None,
        )
        for rank, machine in enumerate(machines)
    )

    for node in nodes:
        await node.start()

    async with asyncio.timeout(600):
        while not all(seen and seen[-1] in ("ready", "failed") for seen in states.values()):
            await asyncio.sleep(0.5)

    assert all(seen[-1] == "ready" for seen in states.values()), states
    return nodes


async def test_the_nodes_find_each_other_and_the_client_finds_all_of_them(
    compute, key: asyncssh.SSHKey, source: Source,
):
    name, machines = compute
    nodes = await cluster(name, machines, key, source)
    tunnels = {node.seed: f"127.0.0.1:{node.tunnel}" for node in nodes}

    system = await casty.connect(
        [nodes[0].seed],
        address_map=lambda addr: tunnels.get(addr, addr),
        cluster_name=name,
    )
    try:
        async with asyncio.timeout(60):
            while len(members := system.members()) < NODES:
                await asyncio.sleep(0.5)

        assert {member.addr for member in members} == set(tunnels), (
            "the client was given one seed; the rest of the cluster reached it on its own"
        )

        pinged = {
            await system.service(worker.Control, at=member).ping()
            for member in members
        }
        assert pinged == {machine.id for machine in machines}, "every node is addressable, and none of them twice"

        def where() -> str:
            """Which node am I? Answered by the worker's environment, not by the caller."""
            import os

            return os.environ["SKYWARD_NODE"]

        for node, machine in zip(nodes, machines, strict=True):
            member = next(m for m in members if m.addr == node.seed)
            code = await codec.payload.encode(where)
            args = await codec.payload.encode(((), {}))
            reply = await system.service(worker.Worker, at=member).run(f"tsk_{machine.id}", code, args)

            outcome = msgspec.msgpack.decode(reply, type=frames.Outcome)
            assert isinstance(outcome, frames.Done), outcome
            assert await codec.payload.decode(outcome.value) == machine.id, (
                "a task pinned to a node runs on that node"
            )
    finally:
        await system.close()

    for node in nodes:
        await node.close()
