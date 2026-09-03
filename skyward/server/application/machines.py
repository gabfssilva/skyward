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
from datetime import datetime
from typing import Protocol

import msgspec

from skyward.server.application import market
from skyward.server.application.runtimes import keypair, public_key
from skyward.server.application.source import detect, resolve
from skyward.server.persistence.computes import ComputeStore, Infrastructure
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.store import now
from skyward.shared import codec
from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.observability import logger
from skyward.shared.provider import Bakeable, Binding, Machine, Mount, Mountable, Preemptible, Provider
from skyward.shared.schemas import (
    Compute,
    ComputeSpec,
    Endpoint,
    Error,
    Event,
    Image,
    Market,
    Node,
    NodeEvent,
    Offer,
    ProgressEvent,
    Volume,
    progressed,
)
from skyward.shared.tls import authority
from skyward.worker import bootstrap

logger = logger.bind(component="machines")


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
        events: EventStore,
    ) -> None:
        self._computes = computes
        self._nodes = nodes
        self._providers = providers
        self._offers = offers
        self._blobs = blobs
        self._events = events
        self._binding = asyncio.Lock()
        self._baked: set[tuple[str, str]] = set()
        """Which compute has already had which environment offered to the provider, once."""
        self._progress: dict[str, tuple[str | None, datetime]] = {}
        """What each machine still short of an address was last seen doing, and when that changed."""

    async def create(self, compute_id: str, node_id: str) -> None:
        """Buy the machine one row is asking for, and write down what we got.

        Every row is bought on its own task, and the tasks run together: what had
        to be settled before the first machine was settled once in :meth:`bind`,
        and a launch only reads it. The machine is claimed for the row before the
        provider is called, so a reply that never arrives is not a machine lost —
        :meth:`resolve` finds it by the claim and writes the id the reply would
        have carried.

        The state is read once more after the binding, because a tick that fired
        while the binding was being made has offered the row to somebody who may
        already have bought on it.
        """
        log = logger.bind(compute_id=compute_id, node_id=node_id)
        compute = await self._computes.get(compute_id)
        node = await self._nodes.get(compute_id, node_id)
        if node.state != "requested":
            log.debug("not buying: the row is {}", node.state)
            return

        infrastructure = await self.bind(compute)
        adapter = await self.adapter(infrastructure.provider_id)

        node = await self._nodes.get(compute_id, node_id)
        if node.state != "requested":
            log.debug("not buying: the row became {} while the compute was being bound", node.state)
            return

        infrastructure = await self._computes.infrastructure(compute_id)
        log.debug("buying one machine on {}", adapter.kind)
        placed, machine, sold = await self._place(adapter, compute, infrastructure, claim(node_id))

        await self._nodes.launched(node_id, machine, offer=placed.offer, market=sold)
        log.bind(instance_id=machine.id).info(
            "bought a {} on the {} market in {}",
            placed.offer.instance_type if placed.offer else adapter.kind,
            sold,
            placed.offer.region if placed.offer and placed.offer.region else "the bound region",
        )

    async def _place(
        self, adapter: Provider, compute: Compute, infrastructure: Infrastructure, node: str
    ) -> tuple[Infrastructure, Machine, Market]:
        """Buy one machine, and if the bound region will not sell one, find one that will.

        Two nested fallbacks. Within a region, ``markets`` is tried in order — this is
        where ``spot_if_available`` becomes liquid, leading with spot and dropping to
        on-demand for the node whose spot launch the provider refuses. Across regions,
        when a whole region has no market left that will sell — an exhausted quota, no
        capacity — the next cheapest offer is bound into its own region and tried there,
        and the region that refused is released so it does not leak the network it
        briefly held.

        The region is a decision one refused node makes and the rest inherit. Twenty
        nodes refused together are twenty candidates to move the compute, and one
        move is all it needs: the walk is taken under the binding lock, after reading
        the binding back, and a node that finds it already moved buys where the
        winner did rather than move it again. Only when every region has refused does
        the compound failure escape, and the tick offers the row again.
        """
        failures: list[Exception] = []

        if bought := await self._buy(adapter, infrastructure, node, failures):
            machine, sold = bought
            return infrastructure, machine, sold

        async with self._binding:
            current = await self._computes.infrastructure(compute.id)
            if current.offer_id != infrastructure.offer_id:
                logger.bind(compute_id=compute.id).debug("the compute moved while this node was being refused; following it")
                return await self._place(adapter, compute, current, node)

            tried = {infrastructure.offer_id}
            for offer in await market.rank(compute.spec, self._offers):
                if offer.id in tried:
                    continue
                tried.add(offer.id)

                logger.bind(compute_id=compute.id, provider=offer.provider_name).info(
                    "no market here would sell; trying {} in {}",
                    offer.instance_type,
                    offer.region or "another region",
                )
                relocated = await self._relocate(adapter, compute, infrastructure, offer)
                if bought := await self._buy(adapter, relocated, node, failures):
                    machine, sold = bought
                    await self._computes.bind(compute.id, relocated)
                    await self._abandon(adapter, infrastructure)
                    return relocated, machine, sold
                await self._abandon(adapter, relocated)

        if not failures:
            raise CapabilityMismatchError(f"{adapter.kind} was bound with no market to buy on")
        raise ExceptionGroup(f"no market could place a {adapter.kind} machine", failures)

    async def _buy(
        self, adapter: Provider, infrastructure: Infrastructure, node: str, failures: list[Exception]
    ) -> tuple[Machine, Market] | None:
        """Try each market the binding allows, in order; the first that sells wins.

        A launch that raises is the signal to try the next market. When none sell, the
        reasons are appended for the caller to raise as one and ``None`` says so, which
        is the caller's cue to look at another region. The market that sold comes back
        with the machine: it is what the machine is billed under, and the node records it.
        """
        for option in infrastructure.markets:
            try:
                machine = await adapter.launch(infrastructure.binding, option, node)
            except Exception as failure:
                logger.debug("{} refused a {} machine: {}", adapter.kind, option, failure)
                failures.append(failure)
                continue
            return machine, option
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

        Asked of the provider first, because ``release`` destroys what ``initialize``
        created and is owed the condition it is written under: no machine left. A
        binding that still has machines under it is not abandoned — the compute's
        earlier nodes are living in it, and on a provider whose machine *is* the thing
        ``initialize`` describes, releasing it would take them down.

        Best effort otherwise: a region that will not release is a leak to log, not a
        reason to fail a placement that has already found a region that sold.
        """
        try:
            if await adapter.machines(infrastructure.binding):
                return
            await adapter.release(infrastructure.binding)
        except Exception:
            logger.warning("could not release the abandoned binding {}", infrastructure.binding, exc_info=True)

    async def bind(self, compute: Compute) -> Infrastructure:
        """Give the compute an address in the world, once, before anything is launched.

        The binding is committed before a single machine is: the reverse order is
        how a crash turns into a fleet that bills forever and that nothing can find.

        Once for the whole file, not once per process. The lock only covers this
        daemon; a second daemon on the same database passes its own lock and minted
        pairs would race — machines launched trusting a public key whose private
        half the row no longer holds, and a fleet nobody can log into. So the store's
        write is conditional on the key, and the loser reads the row back and adopts
        the binding that won, releasing what its own ``initialize`` briefly held.
        """
        async with self._binding:
            infrastructure = await self._computes.infrastructure(compute.id)
            if infrastructure.provider_id:
                return infrastructure

            offer, chosen = await market.pick(compute.spec, self._offers)
            adapter = await self.adapter(offer.provider_id)
            spec = _effective_spec(adapter, compute.spec, offer)

            private, public = await asyncio.to_thread(keypair)
            signing = await asyncio.to_thread(authority)
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
                authority=signing,
                markets=market.order(offer, compute.spec.allocation),
                volumes=mount.phases,
            )
            await self._computes.bind(compute.id, infrastructure)
            logger.bind(compute_id=compute.id, provider=offer.provider_name).info(
                "bound to {} in {}, buying on {}",
                offer.instance_type,
                offer.region or "the account's default region",
                " then ".join(infrastructure.markets) or "no market",
            )

            stored = await self._computes.infrastructure(compute.id)
            if stored.private_key != private:
                logger.warning("compute {} was bound by another daemon; adopting its binding", compute.id)
                await self._abandon(adapter, infrastructure)
                return stored
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
                logger.bind(compute_id=compute_id, instance_id=node.machine).info("baking this environment into {}", tag)
                await adapter.bake(infrastructure.binding, node.machine, tag)
        except Exception:
            logger.bind(compute_id=compute_id).warning("could not bake {} into an image", node.machine, exc_info=True)

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

        Every machine the provider reports carries the claim it was launched under,
        and the claim names a row. A row that has no id yet and whose claim is on a
        machine is a launch whose reply this process never saw — a crash, a restart,
        a timeout — and the id is written here, exactly as ``create`` would have. A
        machine whose claim names no row still alive is nobody's: the row it was
        bought for is gone, and nothing but this will ever stop it billing. Nothing
        here is decided from memory; the store and the listing are the whole truth,
        which is what lets this run while twenty launches are in flight.
        """
        infrastructure = await self._computes.infrastructure(compute.id)
        if not infrastructure.provider_id:
            return

        if not any(node.state in ("requested", "provisioning", "connecting", "bootstrapping", "ready") for node in nodes):
            return

        adapter = await self.adapter(infrastructure.provider_id)

        nodes = await self._nodes.of(compute.id)
        pending = [node for node in nodes if node.state in ("requested", "provisioning")]
        watched = [node for node in nodes if node.state in ("connecting", "bootstrapping", "ready")]

        observed = await adapter.machines(infrastructure.binding)
        logger.bind(compute_id=compute.id).debug(
            "the provider has {} machines under this binding; {} still without an address, {} being held",
            len(observed),
            len(pending),
            len(watched),
        )

        if isinstance(adapter, Preemptible):
            claimed_ids = tuple(node.machine for node in nodes if node.machine)
            if claimed_ids:
                watched_by_machine = {node.machine: node for node in watched}
                warned = await adapter.interruptions(infrastructure.binding, claimed_ids)
                for machine_id, reason in warned.items():
                    if node := watched_by_machine.get(machine_id):
                        await self._lost(node, reason)

        by_claim = {machine.node: machine for machine in observed.values() if machine.node}
        deadline = compute.spec.options.provision_timeout

        for node in pending:
            if node.machine is None and (machine := by_claim.get(claim(node.id))):
                market_guess = infrastructure.markets[0] if infrastructure.markets else None
                await self._nodes.launched(node.id, machine, offer=infrastructure.offer, market=market_guess)
                logger.bind(compute_id=compute.id, node_id=node.id, instance_id=machine.id).info(
                    "found the machine bought for rank {} before its reply arrived", node.rank
                )
                node = msgspec.structs.replace(node, machine=machine.id)

            match observed.get(node.machine or ""):
                case Machine(state="running") as machine if machine.host or machine.private_host:
                    self._progress.pop(node.id, None)
                    await self._nodes.reachable(node.id, machine)
                    logger.bind(compute_id=compute.id, node_id=node.id, instance_id=machine.id).info(
                        "reachable at {}", machine.host or machine.private_host
                    )
                case None if node.machine and _settled(node):
                    await self._lost(node, "the provider no longer has it")
                case Machine() as machine if await self._stalled(node, machine, deadline):
                    await self._lost(node, _unaddressed(machine, deadline))
                case _:
                    pass

        for node in watched:
            match observed.get(node.machine or ""):
                case None if node.machine and _settled(node):
                    await self._lost(node, "the machine went away")
                case Machine(state="running") as machine if _moved(node, machine):
                    logger.bind(compute_id=compute.id, node_id=node.id, instance_id=machine.id).info(
                        "now reached at {}:{}; logging in again", machine.host or machine.private_host, machine.port
                    )
                    await self._nodes.reachable(node.id, machine)
                case _:
                    pass

        owned = {node.machine for node in nodes if node.machine} | {claim(node.id) for node in nodes if node.state != "deleted"}
        strays = tuple(machine.id for machine in observed.values() if machine.id not in owned and machine.node not in owned)
        if strays:
            logger.bind(compute_id=compute.id).warning("terminating {} machine(s) no row of this compute owns: {}", len(strays), ", ".join(strays))
            await adapter.terminate(infrastructure.binding, strays)

    async def terminate(self, compute_id: str, node_id: str) -> None:
        """Stop paying for it. Idempotent by nature: a machine already gone is a no-op."""
        node = await self._nodes.get(compute_id, node_id)
        if node.state != "deleting":
            return

        infrastructure = await self._computes.infrastructure(compute_id)
        if node.machine and infrastructure.provider_id:
            adapter = await self.adapter(infrastructure.provider_id)
            logger.bind(compute_id=compute_id, node_id=node_id, instance_id=node.machine).info("terminating")
            await adapter.terminate(infrastructure.binding, (node.machine,))

        await self._nodes.observe(node_id, "deleted")

    async def release(self, compute_id: str) -> None:
        """Give back everything that was the compute's and not a machine's."""
        infrastructure = await self._computes.infrastructure(compute_id)
        if not infrastructure.provider_id:
            return

        adapter = await self.adapter(infrastructure.provider_id)
        logger.bind(compute_id=compute_id).info("releasing what {} held for it", adapter.kind)
        await adapter.release(infrastructure.binding)

    async def adapter(self, provider_id: str | None) -> Provider:
        adapter = await self._providers.adapter(provider_id or "")
        if not isinstance(adapter, Provider):
            raise CapabilityMismatchError(f"{adapter.kind} can quote hardware but cannot provision it")
        return adapter

    async def _stalled(self, node: Node, machine: Machine, deadline: float) -> bool:
        """A machine that was bought, is there, and has stopped getting closer.

        Every other timeout in the system starts once there is an address to dial.
        This is the window before that one, and it is the window in which a machine
        bills for nothing: the provider has it running, the compute is paying for it,
        and no node can be made out of it.

        The window is measured from the last sign of movement rather than from the
        launch, because elapsed time cannot tell a machine that is never coming up
        from one that is pulling a multi-gigabyte image. Measured from the launch, the
        pull is killed at the deadline and the replacement starts the same pull from
        nothing, for as long as the compute is asked for. A machine whose reported
        progress keeps changing is one to wait for; one that has been saying the same
        thing for the whole window has stopped.

        A provider that reports nothing has one unchanging answer, so its machines are
        held to the deadline from the moment they were bought — which is what every
        provider was held to before there was anything to report.
        """
        if not deadline or node.launched_at is None:
            return False

        seen, since = self._progress.get(node.id, (None, node.launched_at))
        moved = _token(machine)
        if moved != seen:
            self._progress[node.id] = (moved, now())
            await self._progressed(node, machine)
            return False

        return (now() - since).total_seconds() > deadline

    async def _progressed(self, node: Node, machine: Machine) -> None:
        """Say what the machine is doing, once, to whoever is watching.

        The moment it moves is the only moment worth saying so, which is why this is
        here and not on a timer. Published rather than recorded, for the reason a
        gauge is: a compute waiting ten minutes on an image would otherwise leave
        hundreds of rows behind that say nothing the node's state does not.
        """
        if machine.progress is None:
            return

        logger.bind(node_id=node.id, instance_id=machine.id).debug("{}", progressed(machine.progress, machine.completion))
        payload = ProgressEvent(compute=node.compute_id, node=node.id, progress=machine.progress, completion=machine.completion)
        await self._events.publish("node.progress", await codec.json(Event).encode(payload), compute=node.compute_id)

    async def _lost(self, node: Node, why: str) -> None:
        """Give up on a machine, and say so where anybody can read it.

        A node given up on is replaced, and the replacement is all the event log
        would otherwise show. The loss is announced here because here is where it
        is discovered — the reconciler learns of it as a row already in ``lost``,
        with nothing left to say about which machine went missing or why.
        """
        logger.bind(compute_id=node.compute_id, node_id=node.id).warning("giving up on rank {}: {}", node.rank, why)
        self._progress.pop(node.id, None)
        await self._nodes.observe(node.id, "lost", Error(code="not_found", message=why, retryable=True))
        payload = NodeEvent(compute=node.compute_id, node=node.id, state="lost", error=why)
        await self._events.record("node.lost", await codec.json(Event).encode(payload), compute=node.compute_id)


def claim(node_id: str) -> str:
    """The row's id as a provider will carry it: a hostname-safe token, written on the machine at launch.

    What :meth:`Machines.resolve` matches :attr:`Machine.node` against. Row ids are
    ``nod_`` and hex; the underscore is the one character a hostname will not take.
    """
    return node_id.lower().replace("_", "-")


def _settled(node: Node) -> bool:
    return (now() - node.created_at).total_seconds() > DOUBT_SECONDS


def _moved(node: Node, machine: Machine) -> bool:
    """Whether the machine is reached somewhere other than where the row says.

    Most providers never move a machine. One that reaches its machines through a
    relay it stands up itself hands out a new door every time the daemon starts,
    and a row that remembers the old one is a machine nobody can log into any more.
    """
    held = node.provider_binding
    return (machine.host, machine.private_host, machine.port) != (held.get("host"), held.get("private_host"), held.get("port"))


def _token(machine: Machine) -> str | None:
    """The one string that changes exactly when the machine has got closer."""
    return None if machine.progress is None else progressed(machine.progress, machine.completion)


def _unaddressed(machine: Machine, deadline: float) -> str:
    """Why a machine short of an address was given up on, in terms of what it was doing."""
    token = _token(machine)
    if token is None:
        return f"the machine never published an address in {deadline:.0f}s"
    return f"the machine has been {token} for {deadline:.0f}s without publishing an address"
