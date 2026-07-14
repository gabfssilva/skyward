"""The half of a compute that cannot be written down.

A row survives the daemon; an SSH connection does not. Everything here is the
live counterpart of what the store remembers — the open channels, the forwarded
ports, the casty client that dials them — and all of it is rebuilt from the store
when a daemon comes up, never recovered from it.

There is exactly one of these per compute per process, which is what the compute's
lease is for: two daemons holding SSH connections to the same machines would both
believe they were the one bootstrapping it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import asyncssh
import casty

from skyward2.application.provider import Machine
from skyward2.protocol.schemas import Image, NodeState
from skyward2.runtime.node import Node
from skyward2.runtime.source import Source

logger = logging.getLogger(__name__)

type Listener = Callable[[str, str, NodeState, str | None], None]
"""(compute, node, state, error)"""

type Output = Callable[[str, str, str, str | None], None]
"""(compute, node, content, task)"""


def keypair() -> tuple[str, str]:
    """A key for one compute, and only for it.

    Per compute rather than per installation: a machine rented from a marketplace
    is a machine somebody else administers, and a key that opens every compute the
    user has ever run is a key that should not be on it.
    """
    key = asyncssh.generate_private_key("ssh-ed25519")
    return key.export_private_key().decode(), key.export_public_key().decode()


class Runtime:
    """One compute's live machinery.

    Nodes are held by node id — the store's name for them — because that is what
    the reconciler and the task manager both speak. The casty client is built
    lazily, on the first thing that needs to talk to a worker, because there is
    nothing to connect to until a node says it is ready.
    """

    def __init__(self, compute: str, source: Source, private_key: str) -> None:
        self.compute = compute
        self.source = source
        self.private_key = private_key
        self.nodes: dict[str, Node] = {}
        self.dispatched: set[str] = set()
        """Executions already in flight. Two coalesced reconciles must not both send one."""

        self._system: casty.Client | None = None
        self._tunnels: dict[str, str] = {}
        self._connecting = asyncio.Lock()

    def track(self, node_id: str, node: Node) -> None:
        self.nodes[node_id] = node

    def forget(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

    @property
    def ready(self) -> tuple[str, ...]:
        return tuple(node_id for node_id, node in self.nodes.items() if node.tunnel)

    async def system(self) -> casty.Client:
        """The client, dialling every worker through its own tunnel.

        The address map is the whole trick. Workers advertise themselves on the
        private network, where they can reach each other and the daemon cannot;
        every address they hand out is rewritten to the local port that tunnels to
        it. The dict is live — a node that becomes ready after the client
        connected is reachable through the same map.
        """
        async with self._connecting:
            if self._system is None:
                seeds = [self.nodes[node_id].seed for node_id in self.ready]
                if not seeds:
                    raise RuntimeError(f"compute {self.compute} has no ready node to connect to")

                self._refresh()
                self._system = await casty.connect(
                    seeds,
                    address_map=lambda addr: self._tunnels.get(addr, addr),
                    cluster_name=self.compute,
                )

        self._refresh()
        return self._system

    async def member(self, node_id: str) -> casty.Member:
        system = await self.system()
        seed = self.nodes[node_id].seed

        async with asyncio.timeout(30):
            while True:
                if found := next((m for m in system.members() if m.addr == seed), None):
                    return found
                await asyncio.sleep(0.2)

    async def close(self) -> None:
        if self._system:
            await self._system.close()
            self._system = None

        for node in self.nodes.values():
            await node.close()
        self.nodes.clear()

    def _refresh(self) -> None:
        self._tunnels = {
            node.seed: f"127.0.0.1:{node.tunnel}"
            for node in self.nodes.values()
            if node.tunnel
        }


class Runtimes:
    """Every live compute this daemon is holding."""

    def __init__(self, listener: Listener, output: Output) -> None:
        self._listener = listener
        self._output = output
        self._runtimes: dict[str, Runtime] = {}

    def of(self, compute: str) -> Runtime | None:
        return self._runtimes.get(compute)

    def open(self, compute: str, source: Source, private_key: str) -> Runtime:
        return self._runtimes.setdefault(compute, Runtime(compute, source, private_key))

    async def start(
        self,
        runtime: Runtime,
        node_id: str,
        machine: Machine,
        image: Image,
        rank: int,
        peers: tuple[str, ...],
        seeds: tuple[str, ...],
        concurrency: int,
    ) -> None:
        """Bring a machine up, and wire what it learns back to the store.

        The callbacks are the only way anything the node knows reaches anybody
        else. The node itself has never heard of a compute, a task or a database,
        and is not going to.
        """
        node = Node(
            machine,
            compute=runtime.compute,
            private_key=runtime.private_key,
            image=image,
            source=runtime.source,
            rank=rank,
            peers=peers,
            seeds=seeds,
            concurrency=concurrency,
            listener=lambda state, error: self._listener(runtime.compute, node_id, state, error),
            output=lambda content, task: self._output(runtime.compute, node_id, content, task),
        )
        runtime.track(node_id, node)
        await node.start()

    async def close(self, compute: str) -> None:
        if runtime := self._runtimes.pop(compute, None):
            await runtime.close()

    async def shutdown(self) -> None:
        for compute in tuple(self._runtimes):
            await self.close(compute)
