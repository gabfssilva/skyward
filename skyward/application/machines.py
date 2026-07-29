"""The only thing in the control plane that talks to a cloud.

Five verbs, and each one is somebody else's decision already taken: bind a compute
to a provider, create the machine a row is asking for, find out what became of the
machines we created, take one away, keep one as an image because the spec asked for
it. Nothing here decides how many.

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
from typing import Protocol

import msgspec

from skyward.application import market
from skyward.application.errors import CapabilityMismatchError
from skyward.application.provider import Bakeable, Binding, Machine, Mount, Mountable, Preemptible, Provider
from skyward.application.runtimes import keypair, public_key
from skyward.persistence.computes import ComputeStore, Infrastructure
from skyward.persistence.functions import BlobStore
from skyward.persistence.nodes import NodeStore
from skyward.persistence.offers import OfferCache
from skyward.persistence.providers import ProviderStore
from skyward.persistence.store import now
from skyward.protocol.schemas import Compute, ComputeSpec, Endpoint, Error, Image, Market, Node, Offer, Volume
from skyward.runtime import bootstrap
from skyward.runtime.source import detect, resolve

logger = logging.getLogger(__name__)


class _ClusterFormation(Protocol):
    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool: ...


def _clustered(adapter: _ClusterFormation, spec: ComputeSpec, offer: Offer) -> bool:
    allowed = adapter.allows_cluster_formation(spec, offer)
    requested = spec.options.cluster
    if requested and not allowed:
        raise CapabilityMismatchError(
            f"{offer.provider_name} does not allow cluster formation",
            provider=offer.provider_name,
        )
    return allowed if requested is None else requested


def _effective_spec(adapter: Provider, spec: ComputeSpec, offer: Offer) -> ComputeSpec:
    return msgspec.structs.replace(
        spec,
        options=msgspec.structs.replace(
            spec.options,
            cluster=_clustered(adapter, spec, offer),
        ),
    )

DOUBT_SECONDS = 120.0
"""How old a machine must be before its absence from the provider's listing means death.

The listing is filtered by tag, and tag indexes are eventually consistent: a machine
bought seconds ago is running, billing, and invisible. Believing that absence
immediately terminates a healthy machine on its first appearance-check — reliably,
because the tick that asks is the one that was queued behind the launch. Younger
than this, absence is the index lagging; older, the machine is gone.
"""


class Machines:
    def __init__(
        self,
        computes: ComputeStore,
        nodes: NodeStore,
        providers: ProviderStore,
        offers: OfferCache,
        blobs: BlobStore,
    ) -> None:
        self._computes = computes
        self._nodes = nodes
        self._providers = providers
        self._offers = offers
        self._blobs = blobs
        self._binding = asyncio.Lock()
        self._fleet: dict[str, asyncio.Lock] = {}
        self._baked: set[tuple[str, str]] = set()
        """Which compute has already had which environment offered to the provider, once."""

    async def create(self, compute_id: str, node_id: str) -> None:
        """Buy the machine one row is asking for, and write down what we got.

        The state is read again under the fleet lock. A launch takes tens of
        seconds and every tick re-offers the row while it runs, so a wake that
        read ``requested`` before the lock and bought on it would buy a second
        machine for a node about to have one — a machine no row points at, that
        no terminate will ever find, and that bills until someone notices.
        """
        compute = await self._computes.get(compute_id)
        node = await self._nodes.get(compute_id, node_id)
        if node.state != "requested":
            return

        infrastructure = await self.bind(compute)
        adapter = await self.adapter(infrastructure.provider_id)

        async with self._fleet_lock(compute_id):
            node = await self._nodes.get(compute_id, node_id)
            if node.state != "requested":
                return

            infrastructure = await self._computes.infrastructure(compute_id)
            placed, machine, market = await self._place(adapter, compute, infrastructure)

            if placed != infrastructure:
                await self._computes.bind(compute_id, placed)

            await self._nodes.launched(node_id, machine, offer=placed.offer, market=market)

    async def _place(self, adapter: Provider, compute: Compute, infrastructure: Infrastructure) -> tuple[Infrastructure, Machine, Market]:
        """Buy one machine, and if the bound region will not sell one, find one that will.

        Two nested fallbacks. Within a region, ``markets`` is tried in order — this is
        where ``spot_if_available`` becomes liquid, leading with spot and dropping to
        on-demand for the node whose spot launch the provider refuses. Across regions,
        when a whole region has no market left that will sell — an exhausted quota, no
        capacity — the next cheapest offer is bound into its own region and tried there,
        and the region that refused is released so it does not leak the network it
        briefly held.

        The region is a decision the first placed node makes and the rest inherit: the
        winning infrastructure is returned for ``create`` to commit, so the next node
        reads a binding already pointing at the region that sold, and only its markets
        are tried. Only when every region has refused does the compound failure escape,
        and the tick offers the row again.
        """
        failures: list[Exception] = []

        if bought := await self._buy(adapter, infrastructure, failures):
            binding, machine, sold = bought
            return msgspec.structs.replace(infrastructure, binding=binding), machine, sold

        tried = {infrastructure.offer_id}
        for offer in await market.rank(compute.spec, self._offers):
            if offer.id in tried:
                continue
            tried.add(offer.id)

            relocated = await self._relocate(adapter, compute, infrastructure, offer)
            if bought := await self._buy(adapter, relocated, failures):
                binding, machine, sold = bought
                await self._abandon(adapter, infrastructure)
                return msgspec.structs.replace(relocated, binding=binding), machine, sold
            await self._abandon(adapter, relocated)

        if not failures:
            raise CapabilityMismatchError(f"{adapter.kind} was bound with no market to buy on")
        raise ExceptionGroup(f"no market could place a {adapter.kind} machine", failures)

    async def _buy(
        self, adapter: Provider, infrastructure: Infrastructure, failures: list[Exception]
    ) -> tuple[Binding, Machine, Market] | None:
        """Try each market the binding allows, in order; the first that sells wins.

        A launch that raises is the signal to try the next market. When none sell, the
        reasons are appended for the caller to raise as one and ``None`` says so, which
        is the caller's cue to look at another region. The market that sold comes back
        with the machine: it is what the machine is billed under, and the node records it.
        """
        for option in infrastructure.markets:
            try:
                binding, machines = await adapter.launch(infrastructure.binding, option, count=1, min_count=1)
            except Exception as failure:
                failures.append(failure)
                continue
            if machines:
                return binding, machines[0], option
            failures.append(CapabilityMismatchError(f"{adapter.kind} accepted the {option} request and returned no machine"))
        return None

    async def _relocate(self, adapter: Provider, compute: Compute, infrastructure: Infrastructure, offer: Offer) -> Infrastructure:
        """Bind the compute's identity into another offer's region, carrying its key.

        The private key the store already holds is the one the running machines trust,
        so its public half is recovered and re-imported rather than a fresh pair minted
        the fleet would reject.
        """
        if infrastructure.private_key is None:
            raise CapabilityMismatchError(f"{adapter.kind} cannot relocate a compute with no key")

        markets = market.order(offer, compute.spec.allocation)
        spec = _effective_spec(adapter, compute.spec, offer)
        binding = await adapter.initialize(compute.id, spec, offer, markets[0], public_key(infrastructure.private_key))
        return msgspec.structs.replace(
            infrastructure,
            offer_id=offer.id,
            offer=offer,
            binding={**binding, "skyward_cluster": spec.options.cluster},
            markets=markets,
        )

    async def _abandon(self, adapter: Provider, infrastructure: Infrastructure) -> None:
        """Give back the network a region held for a launch it then refused.

        Best effort: a region that will not release is a leak to log, not a reason to
        fail a placement that has already found a region that sold.
        """
        try:
            await adapter.release(infrastructure.binding)
        except Exception:
            logger.warning("could not release the abandoned binding %s", infrastructure.binding, exc_info=True)

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
            spec = _effective_spec(adapter, compute.spec, offer)

            private, public = await asyncio.to_thread(keypair)
            binding = await adapter.initialize(compute.id, spec, offer, chosen, public)
            binding = await self._reuse(adapter, spec.image, binding)

            mount = await self._mount(adapter, spec, binding)

            infrastructure = Infrastructure(
                provider_id=offer.provider_id,
                offer_id=offer.id,
                offer=offer,
                binding={
                    **binding,
                    **mount.binding_patch,
                    "skyward_cluster": spec.options.cluster,
                },
                private_key=private,
                markets=market.order(offer, compute.spec.allocation),
                volumes=mount.phases,
            )
            await self._computes.bind(compute.id, infrastructure)
            return infrastructure

    async def _mount(self, adapter: Provider, spec: ComputeSpec, binding: Binding) -> Mount:
        """Turn the compute's volumes into a launch hint and a bootstrap phase, once.

        Here rather than at connect time because the two halves have different
        deadlines and only this seam meets both: a provider that attaches its own
        storage must name it in the launch request, which is over before a node
        exists, and the phases must be written down before the daemon that renders
        them can die.

        Two ways to reach a bucket, and never both in one compute. A volume with a
        digest was resolved by the client, which had credentials the daemon has none
        of — an R2 or Backblaze bucket that no provider record describes. A volume
        without one is the provider's to resolve, from the account that is already
        paying for the machines. Mixing them would mean two endpoints to sign for
        under one set of mounts, and a compute that is half one account and half
        another is a compute nobody can reason about the cost of.
        """
        if not spec.volumes:
            return Mount()

        managed: list[Volume] = []
        brought: list[tuple[Volume, str]] = []
        for volume in spec.volumes:
            match volume.storage_sha256:
                case None:
                    managed.append(volume)
                case digest:
                    brought.append((volume, digest))

        if managed and brought:
            raise CapabilityMismatchError(
                "a compute mounts volumes the client brought credentials for or volumes the provider resolves, not both",
                provider=adapter.kind,
            )

        if brought:
            resolved = [(volume, await self._endpoint(digest)) for volume, digest in brought]
            return Mount(phases=(bootstrap.mounts(tuple(resolved)),))

        if not isinstance(adapter, Mountable):
            buckets = ", ".join(volume.bucket for volume in managed)
            raise CapabilityMismatchError(
                f"{adapter.kind} cannot mount {buckets}: pass storage= on the volume to bring your own credentials",
                provider=adapter.kind,
            )

        return await adapter.mount(binding, tuple(managed))

    async def _endpoint(self, digest: str) -> Endpoint:
        return msgspec.json.decode(await self._blobs.get(digest), type=Endpoint)

    async def _reuse(self, adapter: Provider, image: Image, binding: Binding) -> Binding:
        """Point the binding at an image this environment was already baked into.

        Every adapter resolves its boot image inside ``initialize`` and writes it into
        the binding under the same key, so a warm image is not another code path — it
        is another value, and everything downstream launches from it without being
        told. The machine still bootstraps: every phase finds its work already done
        and returns, which is what makes an image that has drifted slow rather than
        wrong. That is the whole reason there is nothing here to skip.
        """
        tag = await self._tag(image)
        if tag is None or not isinstance(adapter, Bakeable):
            return binding

        warm = await adapter.baked(binding, tag)
        return {**binding, "image": warm} if warm else binding

    async def bake(self, compute_id: str, node_id: str) -> None:
        """Keep what a machine's bootstrap built, so the next compute does not build it again.

        Asked of rank zero the first time it serves, because one snapshot describes the
        whole compute: every node of it booted from the same image and bootstrapped to
        the same place. Once per compute and environment, and not once per node that
        comes up — the second snapshot would be of the same machine, and would cost
        what the first one costs.

        Best effort by construction. Nothing here is on the path to running work, so a
        provider that will not commit is a line in the log and not a compute that
        fails.
        """
        compute = await self._computes.get(compute_id)
        tag = await self._tag(compute.spec.image)
        if tag is None or (compute_id, tag) in self._baked:
            return

        node = await self._nodes.get(compute_id, node_id)
        if node.rank != 0 or not node.machine:
            return

        infrastructure = await self._computes.infrastructure(compute_id)
        adapter = await self.adapter(infrastructure.provider_id)
        if not isinstance(adapter, Bakeable):
            return

        self._baked.add((compute_id, tag))
        try:
            if await adapter.baked(infrastructure.binding, tag) is None:
                await adapter.bake(infrastructure.binding, node.machine, tag)
        except Exception:
            logger.warning("could not bake %s into an image for %s", node.machine, compute_id, exc_info=True)

    async def _tag(self, image: Image) -> str | None:
        """What to call the image this compute bakes to, or nothing if it must not bake one.

        A local skyward is installed from a wheel built out of the daemon's own
        checkout, and those bytes change with every edit that has been committed
        nowhere. An image named after one would be handed, tomorrow, to a compute
        expecting different code — so a compute running a local skyward bakes nothing,
        whether it asked to or not.
        """
        if not image.warm:
            return None

        match image.skyward if image.skyward != "auto" else detect():
            case "local":
                return None
            case mode:
                return image.content_hash((await resolve(mode)).argument)

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
        is the only interpretation available and the only one that does not leak it —
        but only under the fleet lock, so a machine a concurrent ``create`` has
        launched and not yet written down is not mistaken for one a dead process left
        behind. Without it, adoption races the launch that owns the machine and hands
        it to the wrong row, which is one machine on two ranks and another on none.
        """
        infrastructure = await self._computes.infrastructure(compute.id)
        if not infrastructure.provider_id:
            return

        if not any(node.state in ("requested", "provisioning", "connecting", "bootstrapping", "ready") for node in nodes):
            return

        adapter = await self.adapter(infrastructure.provider_id)

        async with self._fleet_lock(compute.id):
            nodes = await self._nodes.of(compute.id)
            pending = [node for node in nodes if node.state in ("requested", "provisioning")]
            watched = [node for node in nodes if node.state in ("connecting", "bootstrapping", "ready")]

            observed = await adapter.machines(infrastructure.binding)

            if isinstance(adapter, Preemptible):
                claimed_ids = tuple(node.machine for node in nodes if node.machine)
                if claimed_ids:
                    watched_by_machine = {node.machine: node for node in watched}
                    warned = await adapter.interruptions(infrastructure.binding, claimed_ids)
                    for machine_id, reason in warned.items():
                        if node := watched_by_machine.get(machine_id):
                            await self._lost(node, reason)

            claimed = {node.machine for node in nodes if node.machine}
            orphans = [machine for machine_id, machine in observed.items() if machine_id not in claimed]

            for node in pending:
                if node.machine is None and orphans:
                    market_guess = infrastructure.markets[0] if infrastructure.markets else None
                    await self._nodes.launched(node.id, orphans.pop(0), offer=infrastructure.offer, market=market_guess)
                    logger.warning("node %s adopted a machine nobody had written down", node.id)
                    continue

                match observed.get(node.machine or ""):
                    case Machine(state="running") as machine if machine.host or machine.private_host:
                        await self._nodes.reachable(node.id, machine)
                    case None if node.machine and _settled(node):
                        await self._lost(node, "the provider no longer has it")
                    case _:
                        pass

            for node in watched:
                if node.machine and node.machine not in observed and _settled(node):
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

    def _fleet_lock(self, compute_id: str) -> asyncio.Lock:
        """One machine bought or adopted at a time, per compute.

        Launching and orphan-adoption both write a node's machine, and the emitter
        runs them as separate tasks. Serialized, an adoption sweep only ever sees
        machines that are either recorded or genuinely lost — never one a launch is
        halfway through claiming.
        """
        return self._fleet.setdefault(compute_id, asyncio.Lock())

    async def _lost(self, node: Node, why: str) -> None:
        await self._nodes.observe(node.id, "lost", Error(code="not_found", message=why, retryable=True))


def _settled(node: Node) -> bool:
    return (now() - node.created_at).total_seconds() > DOUBT_SECONDS
