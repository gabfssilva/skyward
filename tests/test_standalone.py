from __future__ import annotations

from unittest.mock import patch

import msgspec
import pytest

from skyward.server.application import connector
from skyward.server.application.mock import NODE
from skyward.shared.provider import Machine
from skyward.server.application.runtimes import Runtime
from skyward.shared.schemas import Image, Options
from skyward.server.application.node import Node
from skyward.server.application.source import Source

pytestmark = pytest.mark.unit


class Client:
    def __init__(self, seed: str) -> None:
        self.seed = seed
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _node(name: str, rank: int) -> Node:
    node = Node(
        Machine(id=name, state="running", host=f"10.0.0.{rank + 1}"),
        compute="compute",
        private_key="key",
        image=Image(),
        source=Source(arguments=("skyward",)),
        listener=lambda *_: None,
        output=lambda *_: None,
        sample=lambda *_: None,
        phase=lambda *_: None,
        rank=rank,
        peers=("10.0.0.1", "10.0.0.2"),
        options=Options(cluster=False),
    )
    node.tunnel = 3000 + rank
    return node


async def test_standalone_runtime_connects_to_each_worker_independently() -> None:
    runtime = Runtime("compute", Source(arguments=("skyward",)), "key", cluster=False)
    runtime.track("n0", _node("n0", 0))
    runtime.track("n1", _node("n1", 1))
    seeds: list[tuple[str, ...]] = []
    clients: list[Client] = []

    async def connect(nodes: list[str], **_: object) -> Client:
        seeds.append(tuple(nodes))
        client = Client(nodes[0])
        clients.append(client)
        return client

    with patch("skyward.server.application.runtimes.casty.connect", side_effect=connect):
        first = await runtime.system("n0")
        second = await runtime.system("n1")
        again = await runtime.system("n0")

    assert first is again
    assert first is not second
    assert seeds == [(_node("n0", 0).seed,), (_node("n1", 1).seed,)]

    await runtime.close()
    assert all(client.closed for client in clients)


def test_standalone_nodes_never_receive_a_cluster_seed() -> None:
    second = msgspec.structs.replace(NODE, id="n1", rank=1, address="10.0.0.2")

    assert connector._seeds((NODE, second), second, cluster=False) == ()
    assert connector._seeds((NODE, second), second, cluster=True) == (f"{NODE.address}:25520",)
