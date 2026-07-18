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
from collections.abc import AsyncIterator, Callable
from contextlib import suppress

import asyncssh
import casty

from skyward.application.ports import Route
from skyward.application.provider import Machine
from skyward.protocol.schemas import Executor, Image, NodeState, Options, PluginRef
from skyward.runtime.node import DEFAULT_OPTIONS, Node
from skyward.runtime.source import Source
from skyward.runtime.ssh import Channel

logger = logging.getLogger(__name__)

type Listener = Callable[[str, str, NodeState, str | None], None]
"""(compute, node, state, error)"""

type Output = Callable[[str, str, str, str | None], None]
"""(compute, node, content, task)"""

type Sample = Callable[[str, str, str, float], None]
"""(compute, node, name, value)"""

type Phased = Callable[[str, str, str, str, str | None], None]
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
        self._cursor = 0
        """Where the next forwarded connection lands in the round-robin."""

    def track(self, node_id: str, node: Node) -> None:
        self.nodes[node_id] = node

    def forget(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

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

        The call timeout is the user's function, not a protocol round trip. Casty
        defaults it to ten seconds, which is right for asking an actor a question and
        wrong for the thing this cluster exists to do: a call here *is* the training
        run, and it is allowed to take as long as one takes. Whether it has taken too
        long is the task's own deadline to answer, and a node that has actually died
        is reported by the membership protocol, which has its own much shorter clock.
        """
        async with self._connecting:
            if self._system is None:
                seeds = [self.nodes[node_id].seed for node_id in self.ready]
                if not seeds:
                    raise RuntimeError(f"compute {self.compute} has no ready node to connect to")

                self._refresh()
                self._system = await casty.connect(
                    seeds,
                    config=casty.Config(call_timeout=CALL_TIMEOUT),
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

    def __init__(self, listener: Listener, output: Output, sample: Sample, phase: Phased) -> None:
        self._listener = listener
        self._output = output
        self._sample = sample
        self._phase = phase
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
        plugins: tuple[PluginRef, ...],
        buffer: int = 0,
        executor: Executor = "thread",
        reuse: bool = True,
        options: Options = DEFAULT_OPTIONS,
        user_code: bytes | None = None,
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


class Forward:
    """The two half-duplex streams of one forwarded connection, tied by id.

    The transport is paired requests because HTTP/1.1 carries a body one way at a
    time: the caller's bytes ride up, the node's ride down, and the id the caller
    mints is what says the two are the same connection. The up half opens the
    channel and hands its far end to the down half waiting on that id — so either
    request may arrive first, and the one that does not open it simply waits.

    The channel is half-closed, not slammed shut: the caller reaching the end of
    what it has to send is not the node reaching the end of what it has to say. Up
    signals EOF and stops; the channel is not released until down sees the node
    close its side.
    """

    def __init__(self, runtimes: Runtimes) -> None:
        self._runtimes = runtimes
        self._channels: dict[str, asyncio.Future[Channel]] = {}

    def _slot(self, cid: str) -> asyncio.Future[Channel]:
        return self._channels.setdefault(cid, asyncio.get_running_loop().create_future())

    async def up(self, compute_id: str, cid: str, remote_port: int, route: Route, chunks: AsyncIterator[bytes]) -> None:
        slot = self._slot(cid)
        try:
            runtime = self._runtimes.of(compute_id)
            if runtime is None:
                raise RuntimeError(f"compute {compute_id} is not live on this daemon")
            channel = await runtime.open_channel(remote_port, route)
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
