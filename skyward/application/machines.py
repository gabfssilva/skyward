"""The only thing in the control plane that talks to a cloud.

Four verbs, and each one is somebody else's decision already taken: bind a compute
to a provider, create the machine a row is asking for, find out what became of the
machines we created, take one away. Nothing here decides how many.

The order in ``create`` is the one a payment gateway uses and for the same reason:
the intent is written down before the money is spent. The row exists in
``requested`` before the provider is called, so a create that succeeds and a
process that dies before recording the id leaves a row with no machine — visible,
countable, and recoverable — rather than a machine with no row, which is a machine
nobody will ever find and everybody will keep paying for.
"""

from __future__ import annotations

import asyncio
import logging

import msgspec

from skyward.application import market
from skyward.application.errors import CapabilityMismatchError
from skyward.application.provider import Machine, Provider
from skyward.application.runtimes import keypair
from skyward.persistence.computes import ComputeStore, Infrastructure
from skyward.persistence.nodes import NodeStore
from skyward.persistence.offers import OfferCache
from skyward.persistence.providers import ProviderStore
from skyward.protocol.schemas import Compute, Error, Node

logger = logging.getLogger(__name__)


class Machines:
    def __init__(
        self,
        computes: ComputeStore,
        nodes: NodeStore,
        providers: ProviderStore,
        offers: OfferCache,
    ) -> None:
        self._computes = computes
        self._nodes = nodes
        self._providers = providers
        self._offers = offers
        self._binding = asyncio.Lock()

    async def create(self, compute_id: str, node_id: str) -> None:
        """Buy the machine one row is asking for, and write down what we got."""
        compute = await self._computes.get(compute_id)
        node = await self._nodes.get(compute_id, node_id)
        if node.state != "requested":
            return

        infrastructure = await self.bind(compute)
        adapter = await self.adapter(infrastructure.provider_id)

        binding, machines = await adapter.launch(infrastructure.binding, count=1, min_count=1)
        if not machines:
            raise CapabilityMismatchError(f"{adapter.kind} accepted the request and returned no machine")

        if binding != infrastructure.binding:
            await self._computes.bind(compute_id, msgspec.structs.replace(infrastructure, binding=binding))

        await self._nodes.launched(node_id, machines[0])

    async def bind(self, compute: Compute) -> Infrastructure:
        """Give the compute an address in the world, once, before anything is launched.

        The binding is committed before a single machine is: the reverse order is
        how a crash turns into a fleet that bills forever and that nothing can find.
        """
        async with self._binding:
            infrastructure = await self._computes.infrastructure(compute.id)
            if infrastructure.provider_id:
                return infrastructure

            offer, chosen = await market.pick(compute.spec, self._offers)
            adapter = await self.adapter(offer.provider_id)

            private, public = keypair()
            binding = await adapter.initialize(compute.id, compute.spec, offer, chosen, public)

            infrastructure = Infrastructure(
                provider_id=offer.provider_id,
                offer_id=offer.id,
                binding=binding,
                private_key=private,
            )
            await self._computes.bind(compute.id, infrastructure)
            return infrastructure

    async def resolve(self, compute: Compute, nodes: tuple[Node, ...]) -> None:
        """Ask the provider what became of the machines we asked it for.

        This is the one thing a provider is the authority on. It is not asked which
        machines exist — that is ours, and it is in the store — it is asked for the
        address it assigned and whether the machine is still there. A machine that
        has vanished from under a node is the same event as a preemption, a
        broken bootstrap or a partitioned network: the node is no longer usable, and
        it becomes a deficit like any other.

        A machine the provider reports and no row claims is a machine we created and
        failed to record. It is handed to the oldest row still waiting for one, which
        is the only interpretation available and the only one that does not leak it.
        """
        infrastructure = await self._computes.infrastructure(compute.id)
        if not infrastructure.provider_id:
            return

        pending = [node for node in nodes if node.state in ("requested", "provisioning")]
        watched = [node for node in nodes if node.state in ("connecting", "bootstrapping", "ready")]
        if not pending and not watched:
            return

        adapter = await self.adapter(infrastructure.provider_id)
        observed = await adapter.machines(infrastructure.binding)

        claimed = {node.machine for node in nodes if node.machine}
        orphans = [machine for machine_id, machine in observed.items() if machine_id not in claimed]

        for node in pending:
            if node.machine is None and orphans:
                await self._nodes.launched(node.id, orphans.pop(0))
                logger.warning("node %s adopted a machine nobody had written down", node.id)
                continue

            match observed.get(node.machine or ""):
                case Machine(state="running") as machine if machine.host or machine.private_host:
                    await self._nodes.reachable(node.id, machine)
                case None if node.machine:
                    await self._lost(node, "the provider no longer has it")
                case _:
                    pass

        for node in watched:
            if node.machine and node.machine not in observed:
                await self._lost(node, "the machine went away")

    async def terminate(self, compute_id: str, node_id: str) -> None:
        """Stop paying for it. Idempotent by nature: a machine already gone is a no-op."""
        node = await self._nodes.get(compute_id, node_id)
        if node.state != "deleting":
            return

        infrastructure = await self._computes.infrastructure(compute_id)
        if node.machine and infrastructure.provider_id:
            adapter = await self.adapter(infrastructure.provider_id)
            await adapter.terminate(infrastructure.binding, (node.machine,))

        await self._nodes.observe(node_id, "deleted")

    async def release(self, compute_id: str) -> None:
        """Give back everything that was the compute's and not a machine's."""
        infrastructure = await self._computes.infrastructure(compute_id)
        if not infrastructure.provider_id:
            return

        adapter = await self.adapter(infrastructure.provider_id)
        await adapter.release(infrastructure.binding)

    async def adapter(self, provider_id: str | None) -> Provider:
        adapter = await self._providers.adapter(provider_id or "")
        if not isinstance(adapter, Provider):
            raise CapabilityMismatchError(f"{adapter.kind} can quote hardware but cannot provision it")
        return adapter

    async def _lost(self, node: Node, why: str) -> None:
        await self._nodes.observe(node.id, "lost", Error(code="not_found", message=why, retryable=True))
