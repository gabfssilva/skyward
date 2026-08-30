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

from skyward.server.application.runtimes import Runtimes
from skyward.server.application.source import resolve
from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import LIVE, NodeStore
from skyward.shared.provider import Machine
from skyward.shared.schemas import Node, NodeState
from skyward.worker import plugins, worker

HELD: tuple[NodeState, ...] = ("connecting", "bootstrapping", "ready")
"""States that mean somebody should be holding a live connection to this machine."""


class Connector:
    def __init__(self, computes: ComputeStore, nodes: NodeStore, runtimes: Runtimes, blobs: BlobStore) -> None:
        self._computes = computes
        self._nodes = nodes
        self._runtimes = runtimes
        self._blobs = blobs

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
        twice. Membership alone cannot say so — the node is only held several awaits
        from now — so the machine is claimed first, synchronously, and the event
        that finds it claimed leaves. What the node does when it arrives at a
        machine that is already working is adopt it — see
        :meth:`skyward.server.application.node.Node._serving`.
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
        cluster = bool(infrastructure.binding.get("skyward_cluster", True))
        runtime = self._runtimes.open(compute_id, source, infrastructure.private_key, cluster, infrastructure.authority)
        if node_id in runtime.nodes:
            await runtime.retopology(node_id, _peers(nodes))
            return
        if not runtime.claim(node_id):
            return

        try:
            includes = compute.spec.image.includes_sha256
            user_code = await self._blobs.get(includes) if includes else None
            match infrastructure.binding.get("instance_timeout"):
                case int() as provider_timeout:
                    instance_timeout = compute.spec.ttl or provider_timeout
                case _:
                    instance_timeout = None

            await self._runtimes.start(
                runtime,
                node_id,
                msgspec.convert(node.provider_binding, Machine),
                image=plugins.image(compute.spec.image, plugins.resolve(compute.spec.plugins)),
                rank=node.rank,
                peers=_peers(nodes),
                seeds=_seeds(nodes, node, cluster),
                concurrency=compute.spec.worker.concurrency or 1,
                buffer=compute.spec.worker.buffer,
                executor=compute.spec.worker.executor,
                reuse=compute.spec.worker.reuse,
                options=msgspec.structs.replace(compute.spec.options, cluster=cluster),
                plugins=compute.spec.plugins,
                user_code=user_code,
                volumes=infrastructure.volumes,
                instance_timeout=instance_timeout,
            )
        finally:
            runtime.release(node_id)

    async def disconnect(self, compute_id: str, node_id: str) -> None:
        """Let go of one machine before it is terminated.

        The mirror of :meth:`connect`. A machine on its way out is dropped from this
        end first, so the SSH channel is closed by us and not surprised by the remote
        going away — see :meth:`skyward.server.application.runtimes.Runtime.detach`.
        """
        await self._runtimes.detach(compute_id, node_id)


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

    Only live nodes count. A node that failed or was preempted keeps its row, and its
    address, until somebody sends the terminate it is still owed — but it is not part
    of the world, and a peer list that included it would hand the workers a rank that
    answers to nobody and a world one larger than the one that exists.
    """
    return tuple(node.address or "" for node in sorted(nodes, key=lambda node: node.rank) if node.address and node.state in LIVE)


def _seeds(nodes: tuple[Node, ...], node: Node, cluster: bool = True) -> tuple[str, ...]:
    """Whom this worker knocks on to find the cluster.

    The lowest-ranked live node with an address, and the lowest-ranked node itself has
    none: somebody has to be the door, and a cluster where everybody waits to be let
    in is a cluster of one, N times over.

    Live for the same reason as the peer list: a dead node keeps its address until it
    is terminated, and pointing every worker at a machine that is not answering — the
    lowest-ranked one, so *everybody* is pointed at it, including the node that should
    have been the door itself — is a cluster that never forms.

    It is a bootstrap contact and not a head. After the knock every member is equal,
    and the door may leave.
    """
    if not cluster:
        return ()
    ordered = [candidate for candidate in sorted(nodes, key=lambda node: node.rank) if candidate.address and candidate.state in LIVE]
    if not ordered or ordered[0].id == node.id:
        return ()

    return (f"{ordered[0].address}:{worker.PORT}",)
