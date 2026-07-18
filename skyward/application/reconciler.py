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
import logging
from collections.abc import Callable
from datetime import timedelta
from math import ceil
from time import monotonic

import msgspec

from skyward import plugins
from skyward.application.errors import NotFoundError, SkywardError
from skyward.application.machines import Machines
from skyward.persistence.computes import ComputeStore, GenerationStore
from skyward.persistence.events import EventStore
from skyward.persistence.nodes import LIVE, NodeStore
from skyward.persistence.store import now
from skyward.persistence.tasks import TaskStore
from skyward.protocol import codec
from skyward.protocol.schemas import Compute, ComputeSpec, ComputeStatus, Error, Node, NodeState

logger = logging.getLogger(__name__)

type Wake = Callable[..., None]

IDLE_SECONDS = 30.0
"""How long a node must have had nothing to do before it is taken away.

Booting a machine costs minutes and most providers bill the hour they started. A pool
that shrank the instant a burst ended would kill the machines it had just paid to
boot, and buy them again on the next burst.

Seconds, and not passes: a pass runs whenever anything happens, so a compute in the
middle of a busy moment would run through a pass-counted delay in milliseconds and
call it patience.
"""

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
                logger.exception("reconcile failed for %s", compute_id)
                await self._degrade(compute_id, exc)

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
        await self._nodes.observe(
            node_id,
            state,
            Error(code="not_found", message=error, retryable=True) if error else None,
        )
        await self._record(f"node.{state}", compute_id, node=node_id, error=error)
        await self.compute(compute_id)

        if state == "ready":
            self._wake("compute.dispatch", compute_id=compute_id)
            await self._machines.bake(compute_id, node_id)

    async def unsettled(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return await self._computes.live(), await self._tasks.unsettled()

    async def _pass(self, compute: Compute) -> None:
        if compute.status.state == "deleted":
            return

        if abandoned(compute):
            if not compute.spec.delete_on_exit:
                return
            await self._computes.delete(compute.id, compute.revision, f"abandoned:{compute.id}")
            await self._record("compute.abandoned", compute.id)
            compute = await self._computes.get(compute.id)

        await self._machines.resolve(compute, await self._nodes.of(compute.id))

        nodes = await self._nodes.of(compute.id)
        desired = await self._desired(compute)
        alive = [node for node in nodes if node.state in LIVE]

        for _ in range(desired - len(alive)):
            node = await self._nodes.request(compute.id, compute.generation)
            await self._announce("requested", compute.id, node.id)

        for node in await self._surplus(compute, alive, len(alive) - desired):
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

    async def _desired(self, compute: Compute) -> int:
        """How many machines the compute should have right now.

        The load is what has been asked for and not yet answered — queued and
        running together, because they are the same demand seen a moment apart.
        Sizing to what is running would size the pool to what the pool can already
        do, and a queue would never be a reason to grow.
        """
        if compute.spec.desired == "deleted":
            return 0

        lower, upper = bounds(compute.spec)
        slots = compute.spec.worker.concurrency or 1
        load = await self._tasks.load(compute.id)

        return max(lower, min(upper, ceil(load / slots)))

    async def _surplus(self, compute: Compute, alive: list[Node], surplus: int) -> list[Node]:
        """Which machines to give back, if any.

        Idle for long enough, newest first. Newest because the oldest are the ones
        most likely to be holding something a query cannot see — a broadcast froze its
        ranks when it was admitted, and the low ranks are the ones it froze.

        A compute being deleted skips the waiting. The patience exists to stop a pool
        from thrashing around a burst; a pool nobody wants any more is not going to
        need these machines in thirty seconds, and making the user watch us hesitate
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
        """Nodes with nothing on them, and nothing owed to them, for long enough."""
        holding, owed = await self._tasks.busy(compute_id)
        now = monotonic()

        idle: set[str] = set()
        for node in await self._nodes.of(compute_id):
            if node.id in holding or node.rank in owed:
                self._idle.pop(node.id, None)
                continue

            since = self._idle.setdefault(node.id, now)
            if now - since >= idle_timeout:
                idle.add(node.id)

        return idle

    async def _status(self, compute: Compute) -> None:
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
                await self._computes.observe(
                    compute.id,
                    ComputeStatus(state="deleting", observed_generation=compute.generation, nodes_ready=0, nodes_total=0),
                )
                return
            await self._computes.observe(
                compute.id,
                ComputeStatus(state="deleted", observed_generation=compute.generation, nodes_ready=0, nodes_total=0),
            )
            await self._record("compute.deleted", compute.id)
            return

        lower, _ = bounds(compute.spec)
        was = compute.status.state
        state = "ready" if len(ready) >= lower else "provisioning"
        if compute.spec.desired == "deleted":
            state = "deleting"

        await self._computes.observe(
            compute.id,
            ComputeStatus(
                state=state,
                observed_generation=compute.generation,
                nodes_ready=len(ready),
                nodes_total=len(live),
            ),
        )

        if state == "ready" and was != "ready":
            await self._generations.apply(compute.id, compute.generation)
            await self._record("compute.ready", compute.id)
            self._wake("compute.dispatch", compute_id=compute.id)

    async def _announce(self, state: str, compute_id: str, node_id: str, record: bool = True) -> None:
        self._wake(f"node.{state}", compute_id=compute_id, node_id=node_id)
        if record:
            await self._record(f"node.{state}", compute_id, node=node_id)

    async def _degrade(self, compute_id: str, exc: Exception) -> None:
        compute = await self._computes.get(compute_id)
        await self._computes.observe(
            compute_id,
            msgspec.structs.replace(
                compute.status,
                state="degraded",
                last_error=Error(code="capability_mismatch", message=str(exc), retryable=True),
            ),
        )
        await self._record("compute.degraded", compute_id, error=str(exc))

    async def _record(self, event: str, compute: str, **payload: str) -> None:
        await self._events.record(
            event,
            await codec.json(dict[str, str]).encode({"compute": compute, **payload}),
            compute=compute,
            task=payload.get("task"),
        )

    def _lock(self, compute_id: str) -> asyncio.Lock:
        """One pass per compute at a time.

        The emitter already collapses duplicate wakeups, but the tick and an event
        can arrive from different directions at the same moment, and two passes
        interleaved would each write the deficit the other is already writing.
        """
        return self._locks.setdefault(compute_id, asyncio.Lock())


def abandoned(compute: Compute) -> bool:
    """Whether nobody owns this compute and nobody is coming back for it.

    The lease is the only sign of life a client gives: it is claimed at birth and
    renewed for as long as the process holding the SSH connections is alive. A
    compute past the newborn grace with no live lease belongs to a process that is
    gone — a ``Ctrl-C``, a crash, a laptop closed.

    What follows is spelled out on the lease endpoint: ``delete_on_exit`` tears it
    down, anything else sits ownerless until something attaches. A compute already
    being deleted is never abandoned — teardown must finish no matter who asked.
    """
    if compute.spec.desired == "deleted":
        return False
    if compute.lease.owner is not None and compute.lease.expires_at is not None and compute.lease.expires_at > now():
        return False
    return now() - compute.created_at > timedelta(seconds=ABANDON_SECONDS)


def bounds(spec: ComputeSpec) -> tuple[int, int]:
    """The range the pool may size itself within.

    A compute with no bounds is not elastic: ``nodes=4`` means four, and the
    clamp collapses onto it.

    A compute running a collective is not elastic either, whatever it asked for.
    ``init_process_group`` freezes the world when the last rank joins, and taking a
    rank away afterwards does not shrink the job — it hangs it, at the next
    all-reduce, on a peer that is never going to answer.
    """
    if any(plugin.collective for plugin in plugins.resolve(spec.plugins)):
        return spec.nodes.desired, spec.nodes.desired

    lower = spec.nodes.min if spec.nodes.min is not None else spec.nodes.desired
    upper = spec.nodes.max if spec.nodes.max is not None else spec.nodes.desired

    return min(lower, upper), max(lower, upper)
