from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import timedelta
from typing import Any, ClassVar, Literal, Protocol, Self, runtime_checkable

from msgspec import Struct, field

from skyward.shared.schemas import ComputeSpec, Market, Offer, Volume

type Binding = Mapping[str, Any]
"""Whatever the adapter needs to remember about one compute's infrastructure.

The network it created, the keypair it imported, the security group, the
availability zone it got pinned to. Opaque to everyone but the adapter that
wrote it, persisted on the compute, and handed back on every later call — a
compute outlives the process that started it, so this cannot live in memory.
"""

type MachineState = Literal["pending", "running"]


class Machine(Struct, frozen=True):
    """A machine as the provider currently sees it.

    Only what is needed to reach the machine and to place a peer next to it.
    Everything else about a node — its rank, its price, what it is running — is
    the control plane's business and is not asked of the provider.

    A machine that is gone is not a state: it is simply absent from
    :meth:`Provider.machines`. Terminated, shutting down, never existed — the
    control plane treats them the same, so the adapter does not report the
    difference.

    Attributes
    ----------
    id : str
        The provider's own identifier. The node's identity, not a name the
        control plane invented.
    state : MachineState
        ``"pending"`` until the provider says the machine is up, then
        ``"running"``. Whether SSH is actually answering yet is not the
        provider's business — the node retries until it is.
    host : str | None
        Where the control plane dials. Unknown while the machine is pending: a
        fleet request comes back with ids long before it comes back with
        addresses.
    port : int
        SSH port.
    user : str
        SSH user.
    private_host : str | None
        The address peers use to reach this machine. Absent when the provider has
        no private network, in which case peers use ``host``.
    password : str | None
        Set only by providers that hand back a root password instead of taking a
        public key.
    progress : str | None
        What a pending machine is doing right now, in the provider's own words.
        It is read as a token and not as prose: a value that keeps changing is a
        machine still getting closer to an address, and the provision deadline is
        measured from the last time it changed rather than from the launch. A
        provider with nothing to say leaves it out, and its machines are held to
        the deadline from the moment they were bought.
    """

    id: str
    state: MachineState
    host: str | None = None
    port: int = 22
    user: str = "root"
    private_host: str | None = None
    password: str | None = None
    progress: str | None = None


@runtime_checkable
class Catalog(Protocol):
    """What a cloud provider must implement to be registered.

    Deliberately minimal: it only has to price its hardware. Provisioning is a
    separate protocol on purpose — a provider can be listed and compared long
    before it can be trusted to run a compute.

    Credentials arrive through :meth:`create`. An adapter never reads an
    environment variable, a config file or a keychain on its own: the controller
    owns the provider record, so it owns credential resolution. That is what
    allows two accounts of the same kind to coexist in one process.

    Attributes
    ----------
    kind : str
        Selects the adapter. Several registered providers may share one.
    credential_fields : tuple[str, ...]
        The secrets :meth:`create` expects, named.
    offers_ttl : timedelta
        How long this provider's catalog stays true. A marketplace bundle is gone
        in minutes; an instance type is not.
    """

    kind: ClassVar[str]
    credential_fields: ClassVar[tuple[str, ...]]
    offers_ttl: ClassVar[timedelta]

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        """Build an adapter for one registered provider account.

        Parameters
        ----------
        provider_id : str
            Identifier of the provider record. Stamped on every offer.
        name : str
            The account's alias, as the user named it.
        credentials : Mapping[str, str]
            Keyed by :attr:`credential_fields`.
        config : Mapping[str, Any]
            Non-secret provider settings, such as which regions to read.

        Returns
        -------
        Self
            An adapter bound to that account.
        """
        ...

    def offers(self) -> AsyncIterator[Offer]:
        """Yield the whole catalog, unfiltered.

        Filtering and sorting happen against the cache, not against the API — a
        provider that filters server-side would have to be re-queried for every
        distinct question a user asks.

        Yields
        ------
        Offer
            Prices are per hour. A provider that bills per second normalizes
            here, so nothing downstream has to carry a billing unit around.
        """
        ...


@runtime_checkable
class Provider(Catalog, Protocol):
    """A provider that can also be made to run a compute.

    Every method is idempotent, and not as a courtesy: the control plane is a
    daemon that can die between an API call and the commit that records it. On
    the way back up it calls the same method again, so "already exists" is a
    success, and the adapter is the one that has to know it.

    Infrastructure is destroyed by :meth:`terminate` and by :meth:`release`, and
    only ever because the intent changed. A compute outlives by days the process
    that started it, so nothing an adapter holds is a machine, and letting go of
    an adapter takes nothing down with it.
    """

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        """Whether this placement can give workers a shared network."""
        ...

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """Create what the compute's machines will share, before any of them exists.

        A network, a security group, an imported keypair, a resolved image. Called
        once, when the compute has no binding yet — and called again after a crash
        that lost the binding, which is why it must tolerate finding its own work
        already done.

        Parameters
        ----------
        compute_id : str
            Scopes the infrastructure, and tags it, so that
            :meth:`machines` can find machines nobody wrote down.
        spec : ComputeSpec
            What the user asked for.
        offer : Offer
            The hardware the control plane picked to satisfy it.
        market : Market
            The preferred market — the first one :meth:`launch` will be asked for.
            It is a hint for shared setup that legitimately varies by market (a
            region that only lists spot capacity), not a billing decision: the
            market a machine is bought on is chosen per launch, so the binding this
            returns must stay usable for either market. An adapter whose provider
            has one market only is handed that one.
        public_key : str
            Generated per compute by the control plane. The adapter installs it
            however its provider expects: registered through an API, baked into a
            launch config, or written into ``authorized_keys``.

        Returns
        -------
        Binding
            Persisted on the compute and handed back to every later call.
        """
        ...

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        """Ask for ``count`` machines, accepting as few as ``min_count``.

        Parameters
        ----------
        binding : Binding
            As returned by :meth:`initialize`, or by the previous launch.
        market : Market
            Which market to buy these machines on, decided per launch rather than
            per compute: ``spot_if_available`` tries spot first and, on a node whose
            spot launch is refused, calls again for on-demand. The binding is shared
            and market-neutral; the market that a launch is billed at is this, not
            anything baked into the binding. A provider with one market only is
            handed that one and ignores it.
        count : int
            How many machines are wanted.
        min_count : int
            How few are still worth having. Partial readiness — a fleet request
            that would rather come back with four machines than with nothing.

        Returns
        -------
        tuple[Binding, Sequence[Machine]]
            The machines come back already identified: one that was created but
            whose id was never learned is a machine that is billed and cannot be
            found. The binding comes back because provisioning teaches the adapter
            things — the availability zone the fleet actually landed in, and which
            every later machine now has to be pinned to.
        """
        ...

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """Every machine this compute currently has.

        One call for the whole compute, answering two questions with one query:
        whether the machines asked for have come up, and whether machines exist
        that the control plane never got to write down. A crash between
        :meth:`launch` and the commit is recovered here — the machine is found
        because the adapter tagged it as belonging to this compute, not because
        anyone remembered its id.

        Parameters
        ----------
        binding : Binding
            Identifies the compute whose machines to look for.

        Returns
        -------
        Mapping[str, Machine]
            Keyed by machine id. Machines that are gone are absent.
        """
        ...

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        """Destroy these machines.

        Called on scale-down as well as on delete, so it must not assume it is
        taking the last machine with it.

        Parameters
        ----------
        binding : Binding
            Identifies the compute the machines belong to.
        machine_ids : tuple[str, ...]
            Machines to destroy. Ones that are already gone are not an error.
        """
        ...

    async def release(self, binding: Binding) -> None:
        """Destroy what :meth:`initialize` created, once no machine is left.

        Parameters
        ----------
        binding : Binding
            The infrastructure to take down.
        """
        ...


@runtime_checkable
class Preemptible(Catalog, Protocol):
    """A provider that can be asked, ahead of the fact, which machines it is reclaiming.

    Spot and preemptible capacity is taken back with warning: a rebalance
    recommendation, an interruption notice, a marked-for-termination flag raised
    while the machine is still up. A provider that surfaces such a warning
    implements this so the control plane can turn it into a deficit before the
    machine drops, rather than waiting for it to vanish from :meth:`machines`.

    Detection stays optional and structural: an adapter that carries this method is
    polled, one that does not is not. Nothing has to register.
    """

    async def interruptions(self, binding: Binding, machine_ids: tuple[str, ...]) -> Mapping[str, str]:
        """Which of these machines the provider warns are being reclaimed, and why.

        Parameters
        ----------
        binding : Binding
            Identifies the compute the machines belong to.
        machine_ids : tuple[str, ...]
            The machines currently claimed by a node, the only ones worth asking about.

        Returns
        -------
        Mapping[str, str]
            Machine id to reason, and only for machines the provider is warning it
            will take. A machine absent from the answer is one it has raised no
            warning about — the same convention as :meth:`Provider.machines`, where
            gone is simply absent.
        """
        ...


class Mount(Struct, frozen=True):
    """How one provider satisfies a compute's volumes: before the machine, and on it.

    ``binding_patch`` is spent before anything is launched — it is how a provider
    that attaches storage itself gets the volume named in the launch request, which
    is a decision that cannot be taken once the machine exists. ``phases`` run on
    the machine after SSH opens, and are rendered here rather than on the node
    because the credentials they carry never leave the daemon.

    A provider needs neither: one that mounts nothing and hints nothing returns an
    empty ``Mount``, and the compute boots with no volume phase at all.
    """

    binding_patch: Binding = field(default_factory=dict)
    phases: tuple[str, ...] = ()


@runtime_checkable
class Mountable(Protocol):
    """A provider that can turn a compute's volumes into something a node can read.

    What that means is the provider's own: an S3-compatible endpoint the machine
    FUSE-mounts, credentials minted for the length of the compute, or a volume the
    host attaches so the machine finds it already there. The control plane asks and
    stores the answer; it does not know which of those happened.

    Optional and structural, like :class:`Preemptible`: an adapter carrying this
    method is asked, one that does not makes a compute with volumes a
    ``capability_mismatch`` rather than a compute that boots without them.
    """

    async def mount(self, binding: Binding, volumes: tuple[Volume, ...]) -> Mount:
        """Say what these volumes cost the launch and what they cost the boot.

        Called once per compute, after :meth:`Provider.initialize` and before a
        single machine is launched, so anything minted here — an access key, an
        attachment — is created once and lives as long as the compute does.

        Parameters
        ----------
        binding : Binding
            The compute's infrastructure as :meth:`Provider.initialize` left it.
            The region a bucket must be reached in and the account it belongs to
            are read from here, not from the adapter's config.
        volumes : tuple[Volume, ...]
            The volumes whose credentials the daemon did not already have. A
            volume the client brought its own credentials for never arrives here.

        Returns
        -------
        Mount
            Hints merged into the binding before launch, and phases appended to
            every node's bootstrap.
        """
        ...


@runtime_checkable
class Bakeable(Protocol):
    """A provider that can turn a machine which has finished bootstrapping into a boot image.

    Bootstrap does the same work every time — the same apt packages, the same wheels,
    the same driver — and on a fat image it is most of what a node does before it is
    worth anything. A provider able to snapshot a running machine can be asked to keep
    that work, so the next compute with the same environment boots into it already
    done. Most providers cannot: one that boots from a docker image name has nowhere
    to put a snapshot, and does not carry these methods.

    Optional and structural, like :class:`Preemptible`: an adapter that has them is
    asked, one that does not is not, and nothing registers.

    Nothing here deletes. That is why baking is opt-in on
    :attr:`skyward.shared.schemas.Image.warm` — a snapshot bills for its storage
    until somebody removes it, and the somebody is whoever turned it on.
    """

    async def bake(self, binding: Binding, machine_id: str, tag: str) -> str:
        """Snapshot a machine that is up and serving into an image others can boot from.

        Asked of one machine per compute, once it is working, and never of one on its
        way out: the machine goes on running and goes on holding whatever it was given.

        Parameters
        ----------
        binding : Binding
            Identifies the compute the machine belongs to, and the region the image is
            created in — a snapshot is not global on any provider that has regions.
        machine_id : str
            The machine to snapshot.
        tag : str
            Names the environment rather than the image: it is the hash of what the
            bootstrap installed, so every compute that would bootstrap to the same
            place asks for the same one. The adapter mints its own name from it and
            marks what it creates, which is how :meth:`baked` finds it later.

        Returns
        -------
        str
            The image, in whatever form this adapter's binding carries one.
        """
        ...

    async def baked(self, binding: Binding, tag: str) -> str | None:
        """The image this environment was already baked into, if the provider still has it.

        The provider is the authority and is asked every time. An image that was
        deregistered, or that lives in an account or a region this binding is not
        pointing at, is one that does not exist — and a table on this side would keep
        claiming it did.

        Parameters
        ----------
        binding : Binding
            Identifies the compute, and the region to look in.
        tag : str
            The environment, as :meth:`bake` was given it.

        Returns
        -------
        str | None
            Something that can be booted right now, or ``None`` — which is also the
            answer for an image that exists but is still being registered, since a
            launch pointed at one of those fails.
        """
        ...
