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
from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import LIVE, NodeStore
from skyward.shared.observability import logger
from skyward.shared.provider import Machine
from skyward.shared.schemas import Node, NodeState
from skyward.worker import plugins, worker

logger = logger.bind(component="connector")

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
        log = logger.bind(compute_id=compute_id, node_id=node_id)
        compute = await self._computes.get(compute_id)
        infrastructure = await self._computes.infrastructure(compute_id)
        if not infrastructure.private_key:
            log.debug("nothing to log in with yet: the compute has no key")
            return

        nodes = await self._nodes.of(compute_id)
        node = next((candidate for candidate in nodes if candidate.id == node_id), None)
        if node is None:
            log.debug("no such row")
            return

        if node.state not in HELD or not node.provider_binding:
            log.debug("not connectable: {} and {} a machine", node.state, "with" if node.provider_binding else "without")
            return

        cluster = bool(infrastructure.binding.get("skyward_cluster", True))
        runtime = self._runtimes.open(compute_id, compute.spec.image.skyward, infrastructure.private_key, cluster, infrastructure.authority)
        if node_id in runtime.nodes:
            await runtime.retopology(node_id, _peers(nodes))
            return
        if not runtime.claim(node_id):
            log.debug("another connect already has it")
            return

        log.info("taking hold of rank {} at {}", node.rank, node.address)

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
                seeds=_seeds(nodes, node, runtime.opens(node_id, formed=_formed(nodes, node))),
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

    async def close(self, compute_id: str) -> None:
        """Let go of everything this daemon holds for a compute that is deleted.

        The mirror of :meth:`connect` for the compute as a whole. Its machines are gone,
        and the casty client, the channels and the material written for it would
        otherwise be held until the daemon itself shuts down.
        """
        await self._runtimes.close(compute_id)


def _peers(nodes: tuple[Node, ...]) -> tuple[str, ...]:
    """Every address the workers can use to reach each other, in rank order.

    A photograph of the world as it is now, not as it will be: a node is started as
    soon as it can be logged into, with whichever peers have an address at that
    moment, and told again through :meth:`Runtime.retopology` each time the world
    changes. The pool is dynamic and the workers are written for that; what decides
    when work may start is the compute reaching ``min``, not the last machine booting.

    Only live nodes count. A node that failed or was preempted keeps its row, and its
    address, until somebody sends the terminate it is still owed — but it is not part
    of the world, and a peer list that included it would hand the workers a rank that
    answers to nobody and a world one larger than the one that exists.
    """
    return tuple(node.address or "" for node in sorted(nodes, key=lambda node: node.rank) if node.address and node.state in LIVE)


def _formed(nodes: tuple[Node, ...], node: Node) -> bool:
    """Whether there is already a cluster to knock on, as the rows have it.

    A node that reached ``ready`` has a worker that answered, and a worker that
    answered is one that joined. The rows are what carries that across a daemon
    restart, where this process holds no node yet and every machine it takes hold of
    would otherwise look like the first one.
    """
    return any(candidate.state == "ready" and candidate.id != node.id for candidate in nodes)


def _seeds(nodes: tuple[Node, ...], node: Node, opens: bool) -> tuple[str, ...]:
    """Whom this worker knocks on to find the cluster.

    Every live machine with an address but its own, because casty joins through the
    first seed that answers and gives up only when none of them does. One contact
    would do if it were certain to be up, and none of them is: the list is every
    machine that *has* an address, which includes the ones still installing their
    dependencies, and a worker pointed at a single one of those waits out the slowest
    bootstrap in the compute before it joins anything.

    Live for the same reason as the peer list: a dead node keeps its address until it
    is terminated, and a seed list of machines that are not answering is a cluster
    that never forms.

    An empty list means open the cluster instead of joining it — said by
    :meth:`Runtime.opens` for the first machine, and true by arithmetic for one that
    has nobody left to knock on. It is a bootstrap contact and not a head: after the
    knock every member is equal, and the door may leave.
    """
    if opens:
        return ()

    return tuple(
        f"{candidate.address}:{worker.PORT}"
        for candidate in sorted(nodes, key=lambda candidate: candidate.rank)
        if candidate.address and candidate.state in LIVE and candidate.id != node.id
    )
