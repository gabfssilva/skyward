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
import shlex
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import asyncssh
import casty

from skyward.server.application.node import DEFAULT_OPTIONS, Node
from skyward.server.application.ports import Route, Target
from skyward.server.application.source import Source
from skyward.server.application.ssh import Channel, Result
from skyward.shared.errors import ComputeNotConnectedError
from skyward.shared.observability import logger
from skyward.shared.provider import Machine
from skyward.shared.schemas import Executor, Image, NodeState, Options, PhaseMark, PluginRef
from skyward.shared.tls import Authority, identity
from skyward.worker import worker

logger = logger.bind(component="runtimes")

type Listener = Callable[[str, str, NodeState, str | None], None]
"""(compute, node, state, error)"""

type Output = Callable[[str, str, str, str | None], None]
"""(compute, node, content, task)"""

type Sample = Callable[[str, str, str, float], None]
"""(compute, node, name, value)"""

type Phased = Callable[[str, str, PhaseMark, str, str | None], None]
"""(compute, node, event, phase, error)"""

CALL_TIMEOUT = 86_400.0
"""A day. Long enough not to be a limit, short enough to eventually give up.

A call to a worker carries the user's function, and the user's function is what the
machine was rented for. The thing that ends a task early is its own deadline, which
the user set and the store knows about; this is only the point past which a reply is
never coming.
"""


def keypair() -> tuple[str, str]:
    """A key for one compute, and only for it.

    Per compute rather than per installation: a machine rented from a marketplace
    is a machine somebody else administers, and a key that opens every compute the
    user has ever run is a key that should not be on it.
    """
    key = asyncssh.generate_private_key("ssh-ed25519")
    return key.export_private_key().decode(), key.export_public_key().decode()


def public_key(private: str) -> str:
    """The public half of a compute's private key, to re-import it into another region.

    A region-fallback launch binds the same compute into a second region, which has
    to carry the same key: the private half is what the store kept, so the public half
    is recovered from it rather than minting a new pair the running machines reject.
    """
    return asyncssh.import_private_key(private).export_public_key().decode()


class Runtime:
    """One compute's live machinery.

    Nodes are held by node id — the store's name for them — because that is what
    the reconciler and the task manager both speak. The casty client is built
    lazily, on the first thing that needs to talk to a worker, because there is
    nothing to connect to until a node says it is ready.
    """

    def __init__(self, compute: str, source: Source, private_key: str, cluster: bool = True, authority: Authority | None = None) -> None:
        self.compute = compute
        self.source = source
        self.private_key = private_key
        self.cluster = cluster
        self.authority = authority
        """What signed every member of this compute, and refuses everybody else.

        Null on a compute bound before there was one. The workers on it were started
        without material and would not understand a client that presented any.
        """
        self.nodes: dict[str, Node] = {}
        self.dispatched: set[str] = set()
        """Executions already in flight. Two coalesced reconciles must not both send one."""

        self._claims: set[str] = set()
        """Machines a connect is mid-flight for, before there is a node to hold."""

        self._systems: dict[str | None, casty.Client] = {}
        self._tunnels: dict[str, str] = {}
        self._tls: casty.TLS | None = None
        self._directory: Path | None = None
        self._connecting = asyncio.Lock()
        self._cursor = 0
        """Where the next forwarded connection lands in the round-robin."""

    def track(self, node_id: str, node: Node) -> None:
        self.nodes[node_id] = node

    def forget(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

    def claim(self, node_id: str) -> bool:
        """Reserve one machine for the connect that is about to build its node.

        The reconciler re-offers ``node.connect`` on every tick, and membership in
        :attr:`nodes` only exists once the node is built — several awaits after the
        connector checked for it. Two offers a tick apart would both pass that
        check and hold two SSH channels, and two connect deadlines, to the same
        machine. The claim is synchronous, so the second offer stops here.
        """
        if node_id in self._claims or node_id in self.nodes:
            return False
        self._claims.add(node_id)
        return True

    def release(self, node_id: str) -> None:
        """Let a claim go, whether the node was built or the connect died trying.

        Harmless after :meth:`track` — a tracked node refuses the next claim by
        membership — and necessary after a failure, or the machine could never be
        picked up again.
        """
        self._claims.discard(node_id)

    async def detach(self, node_id: str) -> None:
        """Let go of one machine on purpose, before it is terminated.

        A machine torn down under us drops its SSH link, and a channel that learns
        of the drop before it is told the teardown was deliberate warns and burns its
        reconnect budget chasing a machine that is not coming back. Closing the node
        here makes the drop expected: the channel is already down when the link goes.

        The node stays in the map, closed. It is what tells :meth:`Connector.connect`
        this machine is already in hand — dropping it would let a ``node.connect`` in
        flight from when the node was alive re-adopt a machine that is on its way out.
        Its tunnel is cleared so the live cluster stops routing to it at once.
        """
        if node := self.nodes.get(node_id):
            await node.close()
            node.tunnel = None
            self._refresh()
        if not self.cluster and (system := self._systems.pop(node_id, None)):
            await system.close()

    @property
    def ready(self) -> tuple[str, ...]:
        return tuple(node_id for node_id, node in self.nodes.items() if node.tunnel)

    async def system(self, node_id: str | None = None) -> casty.Client:
        """The client, dialling every worker through its own tunnel.

        The address map is the whole trick. Workers advertise themselves on the
        private network, where they can reach each other and the daemon cannot;
        every address they hand out is rewritten to the local port that tunnels to
        it. The dict is live — a node that becomes ready after the client
        connected is reachable through the same map.

        The call timeout is the user's function, not a protocol round trip. Casty
        defaults it to ten seconds, which is right for asking an actor a question and
        wrong for the thing this cluster exists to do: a call here *is* the training
        run, and it is allowed to take as long as one takes. Whether it has taken too
        long is the task's own deadline to answer, and a node that has actually died
        is reported by the membership protocol, which has its own much shorter clock.
        """
        if not self.cluster and node_id is None:
            raise ValueError("a standalone runtime needs the node whose worker it should reach")
        if self.cluster:
            key = None
            seeds = [self.nodes[ready].seed for ready in self.ready]
            if len(set(seeds)) != len(seeds):
                raise RuntimeError(f"compute {self.compute} has nodes sharing an address — a cluster cannot tell their workers apart")
        else:
            if node_id is None:
                raise ValueError("a standalone runtime needs the node whose worker it should reach")
            key = node_id
            seeds = [self.nodes[node_id].seed]

        async with self._connecting:
            if key not in self._systems:
                if not seeds:
                    raise RuntimeError(f"compute {self.compute} has no ready node to connect to")

                self._refresh()
                self._systems[key] = await casty.connect(
                    seeds,
                    tls=self._material(),
                    config=casty.Config(call_timeout=CALL_TIMEOUT),
                    address_map=self.address_map(key),
                    cluster_name=self.compute,
                )

        self._refresh()
        return self._systems[key]

    def address_map(self, node_id: str | None) -> Callable[[str], str]:
        """How one client turns an advertised address into something it can dial.

        The cluster client resolves through the shared map, keyed by what each
        worker advertises — unique there, because a cluster only exists on a
        private network. A standalone client belongs to one node, and that node's
        advertised address is not its identity: two marketplace machines behind
        the same NAT advertise the same ``host:port``, and a map keyed by it would
        route both clients to whichever tunnel registered last. So the standalone
        client ignores the address entirely and dials its own node's tunnel, read
        live because an SSH reconnect moves it.
        """
        if node_id is None:
            return lambda addr: self._tunnels.get(addr, addr)

        def via(addr: str) -> str:
            node = self.nodes.get(node_id)
            return f"127.0.0.1:{node.tunnel}" if node and node.tunnel else addr

        return via

    async def retopology(self, node_id: str, peers: tuple[str, ...]) -> None:
        """Tell one running worker that the compute is a different size than it was.

        The worker's environment was written when it started, and a resize does not
        restart anybody — that is the point of a resize. So the new world is pushed
        to the nodes that were already here, and only when it actually changed: a
        node that has just come up was started with it.
        """
        node = self.nodes.get(node_id)
        if node is None or node.peers == peers or node_id not in self.ready:
            return

        system = await self.system(node_id)
        await system.service(worker.Control, at=await self.member(node_id)).topology(peers)
        node.peers = peers

    async def member(self, node_id: str) -> casty.Member:
        system = await self.system(node_id)
        seed = self.nodes[node_id].seed

        async with asyncio.timeout(30):
            while True:
                if found := next((m for m in system.members() if m.addr == seed), None):
                    return found
                await asyncio.sleep(0.2)

    async def open_channel(self, remote_port: int, route: Route = "round_robin") -> Channel:
        """One TCP channel to a ready node's port, chosen by ``route``.

        Round-robin over the nodes ready right now — the same set the live cluster
        routes to — so a port served on every node is spread across them one
        connection at a time. The channel rides that node's existing SSH link,
        opened per connection rather than held open the way the worker tunnel is.
        """
        ready = self.ready
        if not ready:
            raise RuntimeError(f"compute {self.compute} has no ready node to reach")

        match route:
            case "round_robin":
                node = self.nodes[ready[self._cursor % len(ready)]]
                self._cursor += 1

        return await node._ssh.open_connection("127.0.0.1", remote_port)

    async def open_shell(
        self,
        node_id: str | None = None,
        command: str | None = None,
        term: str = "xterm-256color",
        size: tuple[int, int] = (80, 24),
    ) -> Channel:
        """A pseudo-terminal on one named node, or on the first ready one.

        Deliberately not round-robin, which is what a forward does: a forward is one
        connection among many to a service replicated across the compute, and a shell
        is a person. Somebody who opens a terminal, reads a file and opens another
        expects the same machine both times, so the choice is either the node they
        named or the lowest-numbered one ready — never the next in a rotation.
        """
        ready = self.ready
        if not ready:
            raise RuntimeError(f"compute {self.compute} has no ready node to reach")

        match node_id:
            case None:
                chosen = ready[0]
            case named if named in ready:
                chosen = named
            case named:
                raise RuntimeError(f"node {named} is not ready on compute {self.compute}")

        return await self.nodes[chosen]._ssh.open_shell(command, term, size)

    def _select(self, target: Target) -> tuple[str, ...]:
        """The ready nodes an operation lands on.

        ``all`` is every node ready right now, so an operation reaches the compute
        the caller can see and not the one it was asked for. A rank is the single
        node holding it, and a rank that is not ready is refused rather than
        skipped: a write nobody performed is not a write that succeeded everywhere.
        """
        ready = self.ready
        if not ready:
            raise RuntimeError(f"compute {self.compute} has no ready node to reach")

        match target:
            case "all":
                return ready
            case rank:
                chosen = tuple(node_id for node_id in ready if self.nodes[node_id]._rank == rank)
                if not chosen:
                    raise RuntimeError(f"compute {self.compute} has no ready node at rank {rank}")
                return chosen

    async def run(self, target: Target, command: str) -> tuple[tuple[str, Result], ...]:
        """One command on every targeted node at once, tagged with whose answer it is."""
        selected = self._select(target)
        results = await asyncio.gather(*(self.nodes[node_id]._ssh.run(command) for node_id in selected))
        return tuple(zip(selected, results, strict=True))

    async def put(self, target: Target, path: str, content: bytes) -> tuple[tuple[str, str | None], ...]:
        """Write the same bytes to every targeted node, and report per node.

        A machine that refused the write is one line of the answer rather than the
        end of it. The caller asked for four copies and is owed which of the four
        exist — raising on the first refusal would leave them knowing neither.
        """
        selected = self._select(target)
        outcomes = await asyncio.gather(
            *(self.nodes[node_id]._ssh.put(path, content) for node_id in selected),
            return_exceptions=True,
        )
        return tuple(
            (node_id, None if outcome is None else str(outcome))
            for node_id, outcome in zip(selected, outcomes, strict=True)
        )

    def get(self, rank: int, path: str) -> AsyncIterator[bytes]:
        """One node's copy of a file.

        Not itself a generator, so a rank nobody is holding is refused now rather
        than at the first chunk — by which time a response has already been opened
        and the status code spent.
        """
        (node_id,) = self._select(rank)
        return self.nodes[node_id]._ssh.get(path)

    async def close(self) -> None:
        for system in self._systems.values():
            await system.close()
        self._systems.clear()

        for node in self.nodes.values():
            await node.close()
        self.nodes.clear()

        if self._directory is not None:
            rmtree(self._directory, ignore_errors=True)
            self._directory, self._tls = None, None

    def _material(self) -> casty.TLS | None:
        """This daemon's own way into the compute, written where casty can read it.

        Once, and lazily: casty is given paths rather than bytes, so the material has
        to live in a directory for as long as this process holds the compute — and a
        compute nobody has dialled has no reason to have one on disk at all. Signed by
        the compute's authority like every node is, because to a worker the daemon is
        one more member: there is no second kind of credential and nothing that skips
        the check.
        """
        if self.authority is None:
            return None

        if self._tls is None:
            self._directory = Path(mkdtemp(prefix=f"skyward-{self.compute}-"))
            member = identity(self.authority, self.compute)
            certificate, key, authority = self._directory / "daemon.crt", self._directory / "daemon.key", self._directory / "ca.crt"
            certificate.write_text(member.certificate)
            key.write_text(member.key)
            key.chmod(0o600)
            authority.write_text(member.authority)
            self._tls = casty.TLS(cert=str(certificate), key=str(key), ca=str(authority))

        return self._tls

    def _refresh(self) -> None:
        self._tunnels = {
            node.seed: f"127.0.0.1:{node.tunnel}"
            for node in self.nodes.values()
            if node.tunnel
        }


class Runtimes:
    """Every live compute this daemon is holding."""

    def __init__(self, listener: Listener, output: Output, sample: Sample, phase: Phased) -> None:
        self._listener = listener
        self._output = output
        self._sample = sample
        self._phase = phase
        self._runtimes: dict[str, Runtime] = {}

    def of(self, compute: str) -> Runtime | None:
        return self._runtimes.get(compute)

    def open(
        self,
        compute: str,
        source: Source,
        private_key: str,
        cluster: bool = True,
        authority: Authority | None = None,
    ) -> Runtime:
        return self._runtimes.setdefault(compute, Runtime(compute, source, private_key, cluster, authority))

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
        plugins: tuple[PluginRef, ...],
        buffer: int = 0,
        executor: Executor = "thread",
        reuse: bool = True,
        options: Options = DEFAULT_OPTIONS,
        user_code: bytes | None = None,
        volumes: tuple[str, ...] = (),
        instance_timeout: int | None = None,
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
            buffer=buffer,
            executor=executor,
            reuse=reuse,
            options=options,
            plugins=plugins,
            user_code=user_code,
            volumes=volumes,
            instance_timeout=instance_timeout,
            tls=identity(runtime.authority, node_id) if runtime.authority else None,
            listener=lambda state, error: self._listener(runtime.compute, node_id, state, error),
            output=lambda content, task: self._output(runtime.compute, node_id, content, task),
            sample=lambda name, value: self._sample(runtime.compute, node_id, name, value),
            phase=lambda event, name, error: self._phase(runtime.compute, node_id, event, name, error),
        )
        runtime.track(node_id, node)
        await node.start()

    async def detach(self, compute: str, node_id: str) -> None:
        """Close this daemon's live connection to one node before it is terminated."""
        if runtime := self._runtimes.get(compute):
            await runtime.detach(node_id)

    async def close(self, compute: str) -> None:
        if runtime := self._runtimes.pop(compute, None):
            await runtime.close()

    async def shutdown(self) -> None:
        for compute in tuple(self._runtimes):
            await self.close(compute)


class Paired:
    """The two half-duplex streams of one channel to a node, tied by id.

    The transport is paired requests because HTTP/1.1 carries a body one way at a
    time: the caller's bytes ride up, the node's ride down, and the id the caller
    mints is what says the two are the same connection. The up half opens the
    channel and hands its far end to the down half waiting on that id — so either
    request may arrive first, and the one that does not open it simply waits.

    The channel is half-closed, not slammed shut: the caller reaching the end of
    what it has to send is not the node reaching the end of what it has to say. Up
    signals EOF and stops; the channel is not released until down sees the node
    close its side.

    What the channel is at the far end — a forwarded socket, a terminal — changes
    nothing here, which is why opening it is the subclass's business and pumping it
    is not.
    """

    def __init__(self, runtimes: Runtimes) -> None:
        self._runtimes = runtimes
        self._channels: dict[str, asyncio.Future[Channel]] = {}

    def _slot(self, cid: str) -> asyncio.Future[Channel]:
        return self._channels.setdefault(cid, asyncio.get_running_loop().create_future())

    def _runtime(self, compute_id: str) -> Runtime:
        runtime = self._runtimes.of(compute_id)
        if runtime is None:
            raise RuntimeError(f"compute {compute_id} is not live on this daemon")
        return runtime

    async def _pump(self, cid: str, opening: Callable[[], Awaitable[Channel]], chunks: AsyncIterator[bytes]) -> None:
        slot = self._slot(cid)
        try:
            channel = await opening()
        except Exception as exc:
            if not slot.done():
                slot.set_exception(exc)
            raise

        slot.set_result(channel)
        _, writer = channel
        with suppress(OSError, asyncssh.Error):
            async for chunk in chunks:
                writer.write(chunk)
                await writer.drain()
            writer.write_eof()

    async def down(self, cid: str) -> AsyncIterator[bytes]:
        try:
            reader, writer = await self._slot(cid)
        except Exception:
            self._channels.pop(cid, None)
            raise
        try:
            with suppress(OSError, asyncssh.Error):
                while data := await reader.read(65536):
                    yield data
        finally:
            self._channels.pop(cid, None)
            with suppress(OSError, asyncssh.Error):
                writer.close()


class Forward(Paired):
    """One local TCP connection carried to a node port."""

    async def up(self, compute_id: str, cid: str, remote_port: int, route: Route, chunks: AsyncIterator[bytes]) -> None:
        await self._pump(cid, lambda: self._runtime(compute_id).open_channel(remote_port, route), chunks)


class Files:
    """One compute's filesystem and shell, reached over the daemon's own SSH links.

    Not a :class:`Paired`. A file operation is one request with one answer, so
    there is no second half to rendezvous with and no per-connection state to hold
    between them. What it shares with a forward is only where it arrives: the SSH
    link the node already has, which the daemon holds and never hands out.

    ``ls`` and ``rm`` are shell commands with the path quoted, because a path is
    the user's and a shell will read ``;`` in one as an invitation.
    """

    def __init__(self, runtimes: Runtimes) -> None:
        self._runtimes = runtimes

    async def ls(self, compute_id: str, target: Target, path: str) -> tuple[tuple[str, Result], ...]:
        return await self._live(compute_id).run(target, f"ls -la {shlex.quote(path)}")

    async def rm(self, compute_id: str, target: Target, path: str) -> tuple[tuple[str, Result], ...]:
        return await self._live(compute_id).run(target, f"rm -rf {shlex.quote(path)}")

    async def put(self, compute_id: str, target: Target, path: str, content: bytes) -> tuple[tuple[str, str | None], ...]:
        return await self._live(compute_id).put(target, path, content)

    def get(self, compute_id: str, rank: int, path: str) -> AsyncIterator[bytes]:
        return self._live(compute_id).get(rank, path)

    async def run(self, compute_id: str, target: Target, command: str) -> tuple[tuple[str, Result], ...]:
        return await self._live(compute_id).run(target, command)

    def _live(self, compute_id: str) -> Runtime:
        """The connections this daemon is holding for a compute, if it is holding any.

        A compute is a row anybody can read and a set of SSH links exactly one
        process holds. Asking a daemon that is not the one holding them is not a
        malformed request and not a missing compute — it is the wrong daemon, and
        the answer is worth saying rather than raising through as a 500.
        """
        runtime = self._runtimes.of(compute_id)
        if runtime is None:
            raise ComputeNotConnectedError(f"compute {compute_id} is not connected to this daemon", compute=compute_id)
        return runtime


class Terminal(Paired):
    """One interactive session carried to a node's pseudo-terminal.

    The same pump as a forward, with a shell on the far end instead of a socket:
    keystrokes go up, everything the terminal paints comes down, and the daemon is
    the only side holding an SSH connection either way.
    """

    async def up(
        self,
        compute_id: str,
        cid: str,
        node_id: str | None,
        command: str | None,
        term: str,
        size: tuple[int, int],
        chunks: AsyncIterator[bytes],
    ) -> None:
        await self._pump(cid, lambda: self._runtime(compute_id).open_shell(node_id, command, term, size), chunks)
