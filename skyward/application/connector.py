"""Turning a machine into a node.

The machine exists and answers on an address; that is all anybody upstream knows
about it. What happens here is the only part of the system that is genuinely a
sequence — log in, install, start the worker, forward the port — and it is the
only part that holds something a database cannot: a live connection.

It decides nothing. A node that fails to come up says so and stops. Whether a
lost node is worth replacing is a question about how many the compute wanted, and
the reconciler is the one holding that.
"""

from __future__ import annotations

import msgspec

from skyward import plugins
from skyward.application.provider import Machine
from skyward.application.runtimes import Runtimes
from skyward.persistence.computes import ComputeStore
from skyward.persistence.nodes import LIVE, NodeStore
from skyward.protocol.schemas import Node, NodeState
from skyward.runtime import worker
from skyward.runtime.source import resolve

HELD: tuple[NodeState, ...] = ("connecting", "bootstrapping", "ready")
"""States that mean somebody should be holding a live connection to this machine."""


class Connector:
    def __init__(self, computes: ComputeStore, nodes: NodeStore, runtimes: Runtimes) -> None:
        self._computes = computes
        self._nodes = nodes
        self._runtimes = runtimes

    async def connect(self, compute_id: str, node_id: str) -> None:
        """Take hold of one machine, once.

        Asked of every machine the store says is up, not only of the ones that have
        just appeared — because ``ready`` is a fact about the machine and not about
        this process. A daemon that restarted, or one that attached to a compute
        another process started, is looking at rows in ``ready`` with nothing on this
        end of them: no connection, no tunnel, no way to reach the worker that is
        still perfectly alive over there.

        Idempotent by the only means available: a node already being held is a node
        already in hand, and the same event arriving twice must not bootstrap it
        twice. What the node does when it arrives at a machine that is already
        working is adopt it — see :meth:`skyward.runtime.node.Node._serving`.
        """
        compute = await self._computes.get(compute_id)
        infrastructure = await self._computes.infrastructure(compute_id)
        if not infrastructure.private_key:
            return

        nodes = await self._nodes.of(compute_id)
        node = next((candidate for candidate in nodes if candidate.id == node_id), None)
        if node is None or node.state not in HELD or not node.provider_binding:
            return

        if not _photographable(nodes):
            return

        source = await resolve(compute.spec.image.skyward)
        runtime = self._runtimes.open(compute_id, source, infrastructure.private_key)
        if node_id in runtime.nodes:
            return

        await self._runtimes.start(
            runtime,
            node_id,
            msgspec.convert(node.provider_binding, Machine),
            image=plugins.image(compute.spec.image, plugins.resolve(compute.spec.plugins)),
            rank=node.rank,
            peers=_peers(nodes),
            seeds=_seeds(nodes, node),
            concurrency=compute.spec.worker.concurrency or 1,
            plugins=compute.spec.plugins,
        )


def _photographable(nodes: tuple[Node, ...]) -> bool:
    """Whether the world can be described yet.

    Nobody starts until every machine of the cohort has an address, because what a
    worker is handed is the whole peer list and its own index into it — and a worker
    started while a peer was still booting would be told the world has two machines
    in it when it has three. It would not fail: it would run, and shard a third of
    the data into nowhere.

    So the first node waits for the last one. The cost is the slowest machine's boot
    time, paid once; the alternative is a silently wrong answer.
    """
    return all(node.address for node in nodes if node.state in LIVE)


def _peers(nodes: tuple[Node, ...]) -> tuple[str, ...]:
    """Every address the workers can use to reach each other, in rank order.

    This is what a distributed job is handed as its world, and it is a photograph
    taken when the compute started. A node added later is not in the list the earlier
    ones were given, so anything that divides work by the size of the world — a
    broadcast, a shard, a collective — is talking about the compute it started with.
    """
    return tuple(node.address or "" for node in sorted(nodes, key=lambda node: node.rank) if node.address)


def _seeds(nodes: tuple[Node, ...], node: Node) -> tuple[str, ...]:
    """Whom this worker knocks on to find the cluster.

    The lowest-ranked node with an address, and the lowest-ranked node itself has
    none: somebody has to be the door, and a cluster where everybody waits to be let
    in is a cluster of one, N times over.

    It is a bootstrap contact and not a head. After the knock every member is equal,
    and the door may leave.
    """
    ordered = [candidate for candidate in sorted(nodes, key=lambda node: node.rank) if candidate.address]
    if not ordered or ordered[0].id == node.id:
        return ()

    return (f"{ordered[0].address}:{worker.PORT}",)
