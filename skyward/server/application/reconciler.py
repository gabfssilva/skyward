"""How many machines there should be, and nothing else.

It counts what it wanted and counts what it has, and writes down the difference.
It does not buy machines, log into them, or place work on them — it writes rows,
and the rows are what somebody else reacts to. That is the entire job, and it is
why it fits on a page.

Two properties fall out of it being written this way, and both are the reason it
exists. It can be run at any moment, from any state, twice: it reads the world
rather than remembering it, so a pass that died halfway is just a pass that will be
run again. And a machine already on its way up counts as a machine — the row exists
before the provider is called — so the pass that runs while a node is booting does
not buy a second one.

The tick is also the safety net under the events. An event is a wakeup, not a unit
of work: if one is lost — a crash between the write and the emit, a listener that
died, a restart — the intent is still in the store, and the next pass re-offers it
to whoever was supposed to act on it. That is what buys the right to skip an outbox.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import timedelta
from math import ceil
from time import monotonic

from skyward.server.application.machines import Machines
from skyward.server.persistence.computes import ComputeStore, GenerationStore
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.nodes import LIVE, NodeStore
from skyward.server.persistence.store import now
from skyward.server.persistence.tasks import TaskStore
from skyward.shared import codec, lifecycle
from skyward.shared.errors import NotFoundError, SkywardError
from skyward.shared.events import (
    ComputeAbandoned,
    ComputeDegraded,
    ComputeDeleted,
    ComputeDeleting,
    ComputeProvisioning,
    ComputeReady,
    ComputeReleaseFailed,
    Event,
    GenerationApplied,
    NodeEvent,
)
from skyward.shared.observability import logger
from skyward.shared.schemas import Compute, ComputeSpec, Error, Node, NodeState
from skyward.worker import plugins

logger = logger.bind(component="reconciler")

type Wake = Callable[..., None]

ABANDON_SECONDS = 60.0
"""How old a compute must be before having no owner means nobody is coming.

A compute is claimed moments after it is created, but by a different request than
the one that created it, and the tick can land in between. Younger than this and
ownerless is a newborn; older, it is a script that was killed — and reconciling it
forward would buy machines for a process that no longer exists, once per tick,
forever. That is precisely the bug this exists to stop.
"""


class Wakeup:
    """The emitter, before there is an app to emit into.

    The services are built to construct the app, and the app owns the emitter, so
    one of the two has to be handed to the other late. This is the seam, and it is
    a no-op until it is bound — which is also what makes a reconciler testable
    without a web framework around it.
    """

    def __init__(self) -> None:
        self._emit: Wake = lambda *_, **__: None

    def bind(self, emit: Wake) -> None:
        self._emit = emit

    def __call__(self, event: str, **payload: str) -> None:
        self._emit(event, **payload)


class Reconciler:
    def __init__(
        self,
        computes: ComputeStore,
        generations: GenerationStore,
        nodes: NodeStore,
        tasks: TaskStore,
        machines: Machines,
        events: EventStore,
        wake: Wakeup,
    ) -> None:
        self._computes = computes
        self._generations = generations
        self._nodes = nodes
        self._tasks = tasks
        self._machines = machines
        self._events = events
        self._wake = wake
        self._locks: dict[str, asyncio.Lock] = {}
        self._idle: dict[str, float] = {}
        """When each node last had nothing to do. Absent means it is doing something."""

    async def compute(self, compute_id: str) -> None:
        async with self._lock(compute_id):
            try:
                await self._pass(await self._computes.get(compute_id))
            except NotFoundError:
                pass
            except Exception as exc:
                logger.bind(compute_id=compute_id).exception("reconcile failed")
                await self._computes.apply(compute_id, ComputeDegraded(compute=compute_id, error=str(exc)))

    async def observed(self, compute_id: str, node_id: str, state: NodeState, error: str) -> None:
        """What the node's own lifecycle reported, and what follows from it.

        The only thing the reconciler is told rather than reads. No query can answer
        whether the SSH connection this process is holding got as far as a running
        worker — only the process holding it knows, and this is how it says so.

        It is also the one moment a machine is known to be fully built, which is the
        only moment an image of it is worth anything. Whether one is taken is the
        spec's decision and the provider's capability, so it is asked of machines and
        settled there.
        """
        logger.bind(compute_id=compute_id, node_id=node_id).debug("node reports {}{}", state, f": {error}" if error else "")
        await self._nodes.observe(
            node_id,
            state,
            Error(code="not_found", message=error, retryable=True) if error else None,
        )
        await self._record(f"node.{state}", NodeEvent(compute=compute_id, node=node_id, state=state, error=error))
        await self.compute(compute_id)

        if state == "ready":
            self._wake("compute.dispatch", compute_id=compute_id)
            await self._machines.bake(compute_id, node_id)

    async def unsettled(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return await self._computes.live(), await self._tasks.unsettled()

    async def _pass(self, compute: Compute) -> None:
        if compute.status.state == "deleted":
            return

        log = logger.bind(compute_id=compute.id)

        if abandoned(compute) and compute.spec.delete_on_exit:
            log.info("nobody has held the lease for {:.0f}s: deleting it", ABANDON_SECONDS)
            await self._computes.apply(compute.id, ComputeAbandoned(compute=compute.id))
            compute = await self._computes.get(compute.id)
            await self._computes.delete(compute.id, compute.revision, f"abandoned:{compute.id}")
            compute = await self._computes.get(compute.id)

        await self._machines.resolve(compute, await self._nodes.of(compute.id))

        nodes = await self._nodes.of(compute.id)
        alive = [node for node in nodes if node.state in LIVE]
        load = await self._tasks.load(compute.id)
        buy, spare = demand(compute, nodes, load)
        lower, upper = bounds(compute.spec)
        surplus = await self._surplus(compute, alive, spare)

        log.debug(
            "pass: {} alive within [{}, {}], {} in the queue, nodes {} — buy {}, {} over the target, draining {}",
            len(alive),
            lower,
            upper,
            load,
            census(nodes),
            buy,
            max(spare, 0),
            len(surplus),
        )

        for _ in range(buy):
            node = await self._nodes.request(compute.id, compute.generation)
            log.bind(node_id=node.id).info("wants one more machine")
            await self._announce("requested", compute.id, node.id)

        for node in surplus:
            log.bind(node_id=node.id).info("draining rank {}: it is not needed any more", node.rank)
            await self._nodes.observe(node.id, "draining")
            await self._announce("draining", compute.id, node.id)

        await self._push(compute)
        await self._status(compute)

    async def _push(self, compute: Compute) -> None:
        """Re-offer every row that is waiting on somebody, and retire the ones that are done.

        A row in ``requested`` is a machine nobody has bought yet; in ``connecting``,
        one nobody has logged into. Either is a reactor that never ran, or one that
        died halfway, and neither can be told apart from one that is simply still
        working — so the event is sent again, and the reactors are written to ignore
        it if they are.

        A node that is up gets offered too, and that is not a mistake. ``ready`` is a
        fact about the machine, not about this process — the row survives the daemon
        and the SSH connection does not. A daemon that has just come up, or one that
        has attached to a compute somebody else started, finds machines that are
        working perfectly and that nothing here is holding a connection to. Somebody
        has to go and pick them up, and the connector is written to notice when they
        are already in hand.

        A draining node leaves as soon as the work it was holding is finished. This is
        the only place the two halves meet: the reconciler decided it should go, the
        queue decides when.
        """
        holding, _ = await self._tasks.busy(compute.id)
        doomed = compute.spec.desired == "deleted"

        for node in await self._nodes.of(compute.id):
            match node.state:
                case "requested":
                    await self._announce("requested", compute.id, node.id, record=False)
                case "connecting" | "bootstrapping" | "ready":
                    self._wake("node.connect", compute_id=compute.id, node_id=node.id)
                case "draining" if doomed or not holding[node.id]:
                    await self._nodes.observe(node.id, "deleting")
                    await self._announce("deleting", compute.id, node.id)
                case "lost" | "failed":
                    await self._nodes.observe(node.id, "deleting")
                    await self._announce("deleting", compute.id, node.id)
                case "deleting":
                    await self._announce("deleting", compute.id, node.id, record=False)
                case _:
                    pass

    async def _surplus(self, compute: Compute, alive: list[Node], surplus: int) -> list[Node]:
        """Which machines to give back, if any.

        Idle for long enough, newest first. Newest because the oldest are the ones
        most likely to be holding something a query cannot see — a broadcast froze its
        ranks when it was admitted, and the low ranks are the ones it froze.

        A compute being deleted skips the waiting. The patience exists to stop a pool
        from thrashing around a burst; a pool nobody wants any more is not going to
        need these machines back in a moment, and making the user watch us hesitate
        before we stop billing them is the wrong kind of caution.
        """
        if surplus <= 0:
            return []

        if compute.spec.desired == "deleted":
            return sorted(alive, key=lambda node: node.rank, reverse=True)[:surplus]

        idle = await self._idlers(compute.id, compute.spec.options.autoscale_idle_timeout)
        leavable = [node for node in alive if node.id in idle]

        return sorted(leavable, key=lambda node: node.rank, reverse=True)[:surplus]

    async def _idlers(self, compute_id: str, idle_timeout: float) -> set[str]:
        """Nodes with nothing on them, and nothing owed to them, for long enough.

        The clock starts at ``ready``, and only runs while the node stays leavable:
        anything else resets it, so a machine that spends five minutes booting has
        been idle for zero seconds when it arrives.
        """
        holding, owed = await self._tasks.busy(compute_id)
        now = monotonic()

        idle: set[str] = set()
        for node in await self._nodes.of(compute_id):
            if not leavable(node, holding, owed):
                self._idle.pop(node.id, None)
                continue

            since = self._idle.setdefault(node.id, now)
            if now - since >= idle_timeout:
                idle.add(node.id)

        return idle

    async def _status(self, compute: Compute) -> None:
        """Say what the count came to. The store decides whether that is news.

        The event carries the observation — how many answer, of how many — and
        the state it leads to is the table's to say. Most passes say what the last
        one said, and those write the counts and record nothing.
        """
        nodes = await self._nodes.of(compute.id)
        live = [node for node in nodes if node.state in LIVE]
        ready = [node for node in live if node.state == "ready"]
        gone = all(node.state == "deleted" for node in nodes)

        if compute.spec.desired == "deleted" and gone:
            try:
                await self._machines.release(compute.id)
            except SkywardError as error:
                if not error.retryable:
                    raise
                await self._computes.apply(compute.id, ComputeReleaseFailed(compute=compute.id, error=error.message))
                return
            await self._computes.apply(compute.id, ComputeDeleted(compute=compute.id))
            logger.bind(compute_id=compute.id).info("deleted: every machine is gone and the binding is released")
            return

        lower, _ = bounds(compute.spec)
        counted = {"compute": compute.id, "nodes_ready": len(ready), "nodes_total": len(live)}
        if compute.spec.desired == "deleted":
            event = ComputeDeleting(**counted)
        elif len(ready) >= lower:
            event = ComputeReady(**counted, generation=compute.generation)
        else:
            event = ComputeProvisioning(**counted, generation=compute.generation)

        if not await self._computes.apply(compute.id, event):
            return

        logger.bind(compute_id=compute.id).info("{} → {}, {} of {} nodes ready", compute.status.state, lifecycle.leads(event), len(ready), len(live))

        if isinstance(event, ComputeReady):
            await self._generations.apply(compute.id, compute.generation)
            await self._computes.apply(compute.id, GenerationApplied(compute=compute.id, number=compute.generation))
            self._wake("compute.dispatch", compute_id=compute.id)

    async def _announce(self, state: NodeState, compute_id: str, node_id: str, record: bool = True) -> None:
        self._wake(f"node.{state}", compute_id=compute_id, node_id=node_id)
        if record:
            await self._record(f"node.{state}", NodeEvent(compute=compute_id, node=node_id, state=state))

    async def _record(self, name: str, payload: Event) -> None:
        await self._events.record(name, await codec.json(Event).encode(payload), compute=payload.compute)

    def _lock(self, compute_id: str) -> asyncio.Lock:
        """One pass per compute at a time.

        The emitter already collapses duplicate wakeups, but the tick and an event
        can arrive from different directions at the same moment, and two passes
        interleaved would each write the deficit the other is already writing.
        """
        return self._locks.setdefault(compute_id, asyncio.Lock())


def census(nodes: Sequence[Node]) -> str:
    """What the rows say, counted by state — the one line that says where a pass stands."""
    counted = Counter(node.state for node in nodes)
    return ", ".join(f"{count} {state}" for state, count in sorted(counted.items())) or "none"


def abandoned(compute: Compute) -> bool:
    """Whether nobody owns this compute and nobody is coming back for it.

    The lease is the only sign of life a client gives: it is claimed at birth and
    renewed for as long as the process holding the SSH connections is alive. A
    compute past the newborn grace with no live lease belongs to a process that is
    gone — a ``Ctrl-C``, a crash, a laptop closed.

    What follows is spelled out on the lease endpoint: ``delete_on_exit`` tears it
    down, anything else sits ownerless until something attaches. Sitting ownerless
    is not being ignored: the machines are still reconciled, because the compute a
    shell created is never leased by anybody and would otherwise stop being brought
    up sixty seconds into its own provisioning. A compute already being deleted is
    never abandoned — teardown must finish no matter who asked.
    """
    if compute.spec.desired == "deleted":
        return False
    if compute.lease.owner is not None and compute.lease.expires_at is not None and compute.lease.expires_at > now():
        return False
    return now() - compute.created_at > timedelta(seconds=ABANDON_SECONDS)


def leavable(node: Node, holding: Counter[str], owed: frozenset[int]) -> bool:
    """Whether this node may ever be reclaimed, before any question of time.

    Only a ``ready`` node. One still being bought, logged into, or bootstrapped has
    had nothing to do since before it existed — an idle clock started at
    ``requested`` reads boot time as idleness, and boot time is minutes on the
    providers where it matters, so an elastic pool would drain the machines it was
    still waiting for and buy them again when the work arrived.

    And not one holding an execution, or standing on a rank a broadcast froze:
    rank 3 is owed work even when nothing has been placed on it yet, and taking
    the machine leaves the broadcast waiting for a peer that is never coming back.
    """
    return node.state == "ready" and node.id not in holding and node.rank not in owed


def demand(compute: Compute, nodes: Sequence[Node], load: int) -> tuple[int, int]:
    """How many machines to buy, and how many to give back.

    ``initial`` is a size the pool is asked for once rather than a target it is held
    to, so it is counted against the rows the request created: a machine that never
    came up is not bought again, because the pool is standing on what it does have
    and ``min`` is what says whether that is enough. Once per generation, and a
    generation is a resize — a pool grown from three to eight asks for five, and
    what is already up counts whichever definition bought it.

    What holds the size afterwards is the floor. Without a ``max`` the compute is
    not elastic and the load says nothing: it is held at ``min`` and tolerated up to
    the size it opened at. Those are the same number for a ``nodes=4`` and they are
    not for a pool that asked for eight and is willing to live on four, which is the
    whole point of the two being different fields — nothing is bought to close that
    gap, and nothing is drained to close it either.

    With a ``max`` the load is what sizes it, between the two. The load is what has
    been asked for and not yet answered — queued and running together, because they
    are the same demand seen a moment apart. Sizing to what is running would size the
    pool to what the pool can already do, and a queue would never be a reason to grow.
    """
    alive = sum(1 for node in nodes if node.state in LIVE)

    if compute.spec.desired == "deleted":
        return 0, alive

    hold, ceiling = bounds(compute.spec)
    if compute.spec.nodes.max is not None:
        slots = compute.spec.worker.concurrency or 1
        hold = ceiling = max(hold, min(ceiling, ceil(load / slots)))

    spent = sum(1 for node in nodes if node.generation == compute.generation or node.state in LIVE)

    return max(hold - alive, compute.spec.nodes.initial - spent, 0), alive - ceiling


def bounds(spec: ComputeSpec) -> tuple[int, int]:
    """The range the pool may size itself within.

    A compute with no bounds is not elastic: ``nodes=4`` means four, and the
    clamp collapses onto it.

    A compute running a collective is not elastic either, whatever it asked for.
    ``init_process_group`` freezes the world when the last rank joins, and taking a
    rank away afterwards does not shrink the job — it hangs it, at the next
    all-reduce, on a peer that is never going to answer. It is also the one pool
    that is held to ``initial`` rather than to ``min``: a world of eight that opened
    on six is not a smaller job, it is a rendezvous two ranks short.
    """
    if plugins.collective(spec.plugins):
        return spec.nodes.initial, spec.nodes.initial

    lower = spec.nodes.min if spec.nodes.min is not None else spec.nodes.initial
    upper = spec.nodes.max if spec.nodes.max is not None else spec.nodes.initial

    return min(lower, upper), max(lower, upper)
