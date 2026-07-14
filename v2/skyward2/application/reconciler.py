"""Closing the gap between what was asked for and what is there.

Called with a key and never with a payload. That is what lets the emitter collapse
a burst of wakeups for one compute into a single run, and it is why a lost event
costs latency rather than correctness: the reconciler reads the world itself, and
the sweep will hand it the same key again.

Everything here is therefore written to be run at any moment, from any state,
twice. The provider is asked what machines exist rather than told; a machine
nobody wrote down is adopted; a node in flight is skipped rather than started
again.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import msgspec

from skyward2.application.errors import CapabilityMismatchError, NotFoundError
from skyward2.application.provider import Binding, Machine, Provider
from skyward2.application.runtimes import Runtime, Runtimes, keypair
from skyward2.persistence.computes import ComputeStore, GenerationStore, Infrastructure
from skyward2.persistence.events import EventStore
from skyward2.persistence.functions import BlobStore
from skyward2.persistence.nodes import NodeStore
from skyward2.persistence.offers import OfferCache
from skyward2.persistence.providers import ProviderStore
from skyward2.persistence.tasks import TaskStore
from skyward2.protocol import codec
from skyward2.protocol.schemas import (
    Compute,
    ComputeSpec,
    ComputeStatus,
    Error,
    Execution,
    Node,
    NodeState,
    Offer,
    Task,
)
from skyward2.runtime import worker
from skyward2.runtime.source import resolve

logger = logging.getLogger(__name__)

type Wake = Callable[..., None]

DEAD = ("lost", "failed", "deleted")


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
        blobs: BlobStore,
        providers: ProviderStore,
        offers: OfferCache,
        events: EventStore,
        runtimes: Runtimes,
        wake: Wakeup,
    ) -> None:
        self._computes = computes
        self._generations = generations
        self._nodes = nodes
        self._tasks = tasks
        self._blobs = blobs
        self._providers = providers
        self._offers = offers
        self._events = events
        self._runtimes = runtimes
        self._wake = wake
        self._locks: dict[str, asyncio.Lock] = {}

    async def compute(self, compute_id: str) -> None:
        async with self._lock(compute_id):
            try:
                await self._reconcile(await self._computes.get(compute_id))
            except NotFoundError:
                await self._runtimes.close(compute_id)
            except Exception as exc:
                logger.exception("reconcile failed for %s", compute_id)
                await self._degrade(compute_id, exc)

    async def task(self, task_id: str) -> None:
        task = await self._tasks.get(task_id)
        if task.state not in ("queued", "running"):
            return

        runtime = self._runtimes.of(task.compute_id)
        if runtime is None or not runtime.ready:
            return

        for execution in task.executions:
            if execution.state == "created" and execution.id not in runtime.dispatched:
                await self._dispatch(task, execution, runtime)

    async def observed(self, compute_id: str, node_id: str, state: NodeState, error: str) -> None:
        """What the node's own lifecycle reported, and what follows from it.

        The only thing the reconciler is told rather than reads. No query can
        answer whether the SSH connection this process is holding got as far as a
        running worker — only the process holding it knows, and this is how it
        says so.
        """
        await self._nodes.observe(
            node_id,
            state,
            Error(code="not_found", message=error, retryable=True) if error else None,
        )
        await self._record(f"node.{state}", compute_id, node=node_id, error=error)
        await self.compute(compute_id)

    async def unsettled(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return await self._computes.live(), await self._tasks.unsettled()

    async def _reconcile(self, compute: Compute) -> None:
        """One pass: observe, adopt, close the deficit, close the surplus, report.

        The order matters. Observation comes first because the provider is the only
        authority on what exists; adoption comes before the deficit because a
        machine we forgot is a machine we must not buy twice.
        """
        if compute.status.state == "deleted":
            return

        infrastructure = await self._computes.infrastructure(compute.id)
        if not infrastructure.provider_id:
            if compute.spec.desired == "deleted":
                await self._settle(compute, deleted=True)
                return
            infrastructure = await self._provision(compute)

        adapter = await self._adapter(infrastructure.provider_id)
        observed = await adapter.machines(infrastructure.binding)
        known = {node.provider_binding["machine_id"]: node for node in await self._nodes.of(compute.id)}

        for machine_id, machine in observed.items():
            if machine_id not in known:
                node = await self._nodes.adopt(compute.id, compute.generation, len(known), machine)
                known[machine_id] = node
                await self._record("node.adopted", compute.id, node=node.id)

        for machine_id, node in known.items():
            if machine_id not in observed and node.state not in DEAD:
                await self._nodes.observe(node.id, "lost", Error(code="not_found", message="the provider no longer has it", retryable=True))
                await self._record("node.lost", compute.id, node=node.id)

        alive = {machine_id: node for machine_id, node in known.items() if machine_id in observed}
        wanted = 0 if compute.spec.desired == "deleted" else compute.spec.nodes.desired

        if (deficit := wanted - len(alive)) > 0:
            await self._grow(compute, infrastructure, adapter, deficit)
            return

        if (surplus := len(alive) - wanted) > 0:
            await self._shrink(compute, infrastructure, adapter, alive, surplus)
            return

        await self._connect(compute, infrastructure, alive, dict(observed))
        await self._settle(compute, deleted=not observed and compute.spec.desired == "deleted", adapter=adapter, binding=infrastructure.binding)

    async def _provision(self, compute: Compute) -> Infrastructure:
        """Buy the compute an address in the world, and write it down before using it.

        The binding is committed before a single machine is launched. The reverse
        order — launch, then record where they are — is how a crash turns into a
        fleet that bills forever and that nothing can find.
        """
        offer = await self._pick(compute.spec)
        adapter = await self._adapter(offer.provider_id)

        private, public = keypair()
        binding = await adapter.initialize(compute.id, compute.spec, offer, public)

        infrastructure = Infrastructure(
            provider_id=offer.provider_id,
            offer_id=offer.id,
            binding=binding,
            private_key=private,
        )
        await self._computes.bind(compute.id, infrastructure)
        await self._record("compute.provisioning", compute.id)

        return infrastructure

    async def _pick(self, spec: ComputeSpec) -> Offer:
        """The hardware, out of everything on offer that would do.

        Each ``Spec`` is an alternative, not a requirement — the point of listing
        several is that the second one is what you get when the first is sold out.
        """
        candidates: list[Offer] = []

        for wanted in spec.specs:
            page = await self._offers.list(
                provider=None,
                kind=wanted.provider.kind,
                accelerator=wanted.accelerator,
                min_count=wanted.accelerator_count if wanted.accelerator else None,
                min_vram=None,
                max_price=None,
                refresh=False,
            )
            fitting = [
                offer for offer in page.items
                if (wanted.cpus is None or offer.cpus >= wanted.cpus)
                and (wanted.memory_gb is None or offer.memory_gb >= wanted.memory_gb)
                and (wanted.region is None or offer.region == wanted.region)
            ]
            if fitting and spec.selection == "first":
                return min(fitting, key=lambda offer: offer.price or 0.0)
            candidates.extend(fitting)

        if not candidates:
            raise CapabilityMismatchError("nothing on offer satisfies the spec")

        return min(candidates, key=lambda offer: offer.price or 0.0)

    async def _grow(self, compute: Compute, infrastructure: Infrastructure, adapter: Provider, deficit: int) -> None:
        minimum = compute.spec.nodes.min or compute.spec.nodes.desired
        binding, machines = await adapter.launch(infrastructure.binding, deficit, min(minimum, deficit))

        if binding != infrastructure.binding:
            await self._computes.bind(compute.id, msgspec.structs.replace(infrastructure, binding=binding))

        existing = len(await self._nodes.of(compute.id))
        for offset, machine in enumerate(machines):
            node = await self._nodes.adopt(compute.id, compute.generation, existing + offset, machine)
            await self._record("node.launched", compute.id, node=node.id)

        self._wake("compute.changed", compute_id=compute.id)

    async def _shrink(
        self,
        compute: Compute,
        infrastructure: Infrastructure,
        adapter: Provider,
        alive: dict[str, Node],
        surplus: int,
    ) -> None:
        """Take the newest machines away first.

        Newest, because the oldest are the ones most likely to be holding something:
        a broadcast froze its ranks when it was admitted, and the low ranks are the
        ones it froze.
        """
        doomed = sorted(alive.items(), key=lambda pair: pair[1].rank, reverse=True)[:surplus]

        await adapter.terminate(infrastructure.binding, tuple(machine_id for machine_id, _ in doomed))

        runtime = self._runtimes.of(compute.id)
        for _, node in doomed:
            await self._nodes.observe(node.id, "deleted")
            await self._record("node.deleted", compute.id, node=node.id)
            if runtime and (held := runtime.nodes.pop(node.id, None)):
                await held.close()

        self._wake("compute.changed", compute_id=compute.id)

    async def _connect(
        self,
        compute: Compute,
        infrastructure: Infrastructure,
        alive: dict[str, Node],
        observed: dict[str, Machine],
    ) -> None:
        """Start a node's lifecycle for every machine that has one to start.

        The seed is the lowest-ranked machine's private address, and it comes from
        the machine rather than from a node that is already up. Nothing is up when
        this runs — every node of a new compute is started in the same pass — so a
        seed conditioned on readiness would be no seed at all, and each worker would
        form a cluster of one and never find the others.

        It is a bootstrap contact and not a head: casty needs an address to knock
        on, and after the knock every member is equal.

        Idempotent by the only means available: a node already being held is a node
        already being bootstrapped, and reconciling twice must not bootstrap twice.
        """
        if not alive or not infrastructure.private_key:
            return

        source = await resolve(compute.spec.image.skyward)
        runtime = self._runtimes.open(compute.id, source, infrastructure.private_key)
        concurrency = compute.spec.worker.concurrency or 1

        ordered = sorted(alive.items(), key=lambda pair: pair[1].rank)
        first, _ = ordered[0]
        seed = _seed(observed[first])

        for machine_id, node in ordered:
            machine = observed[machine_id]
            if node.id in runtime.nodes or machine.state != "running" or not machine.host:
                continue

            await self._runtimes.start(
                runtime,
                node.id,
                machine,
                compute.spec.image,
                seeds=() if machine_id == first else (seed,),
                concurrency=concurrency,
            )

    async def _settle(
        self,
        compute: Compute,
        deleted: bool,
        adapter: Provider | None = None,
        binding: Binding | None = None,
    ) -> None:
        if deleted:
            if adapter is not None and binding is not None:
                await adapter.release(binding)
            await self._runtimes.close(compute.id)
            await self._computes.observe(compute.id, ComputeStatus(state="deleted", observed_generation=compute.generation, nodes_ready=0, nodes_total=0))
            await self._record("compute.deleted", compute.id)
            return

        nodes = await self._nodes.of(compute.id)
        live = [node for node in nodes if node.state not in DEAD]
        ready = [node for node in live if node.state == "ready"]
        minimum = compute.spec.nodes.min or compute.spec.nodes.desired

        was = compute.status.state
        state = "ready" if len(ready) >= minimum and ready else "provisioning"
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
            self._wake("compute.changed", compute_id=compute.id)
            for task_id in await self._tasks.unsettled():
                self._wake("task.changed", task_id=task_id)

    async def _dispatch(self, task: Task, execution: Execution, runtime: Runtime) -> None:
        """Hand one execution to one worker, and stop holding the reconcile open.

        The call is left running as its own task because it lasts as long as the
        user's function does — hours, possibly — and a reconcile that waited for it
        would be a control plane that stops controlling while somebody trains a
        model.
        """
        node = await self._placement(task, execution, runtime)
        if node is None:
            return

        runtime.dispatched.add(execution.id)
        await self._tasks.observe(execution.id, "dispatching", node_id=node)
        asyncio.get_running_loop().create_task(self._run(task, execution, runtime, node))

    async def _placement(self, task: Task, execution: Execution, runtime: Runtime) -> str | None:
        ready = runtime.ready
        if not ready:
            return None

        if task.dispatch == "one":
            return ready[execution.ordinal % len(ready)]

        nodes = {node.rank: node.id for node in await self._nodes.of(task.compute_id)}
        placed = nodes.get(execution.rank)
        return placed if placed in ready else None

    async def _run(self, task: Task, execution: Execution, runtime: Runtime, node_id: str) -> None:
        try:
            code = await self._blobs.get(task.function)
            args = await self._blobs.get(task.args_sha256)

            member = await runtime.member(node_id)
            system = await runtime.system()

            await self._tasks.observe(execution.id, "started", node_id=node_id)
            await self._record("task.started", task.compute_id, task=task.id)

            reply = await system.service(worker.Worker, at=member).run(execution.id, code, args)
            outcome = msgspec.msgpack.decode(reply, type=worker.Outcome)

            match outcome:
                case worker.Done(value=value):
                    await self._tasks.observe(execution.id, "succeeded", result_sha256=await self._blobs.store(value))
                    await self._record("task.succeeded", task.compute_id, task=task.id)
                case worker.Failed(error=error, traceback=trace):
                    await self._tasks.observe(
                        execution.id,
                        "failed",
                        error=Error(code="task_failed", message=error, retryable=False, details={"traceback": trace}),
                    )
                    await self._record("task.failed", task.compute_id, task=task.id)
        except Exception as exc:
            await self._lost(task, execution, exc)
        finally:
            runtime.dispatched.discard(execution.id)
            self._wake("task.changed", task_id=task.id)

    async def _lost(self, task: Task, execution: Execution, exc: Exception) -> None:
        """The call died, and we do not know whether the function did.

        This is the only honest verdict available. The worker may have run the
        user's code to completion and lost the reply on the way back, so calling it
        `failed` — which is retryable without asking — would be the system deciding
        on the user's behalf that a duplicate side effect is acceptable.
        """
        logger.warning("execution %s lost", execution.id, exc_info=exc)
        await self._tasks.observe(
            execution.id,
            "indeterminate",
            error=Error(code="task_indeterminate", message=f"{type(exc).__name__}: {exc}", retryable=False),
        )
        await self._record("task.indeterminate", task.compute_id, task=task.id)

    async def _adapter(self, provider_id: str | None) -> Provider:
        adapter = await self._providers.adapter(provider_id or "")
        if not isinstance(adapter, Provider):
            raise CapabilityMismatchError(f"{adapter.kind} can quote hardware but cannot provision it")
        return adapter

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
        """One reconcile per compute at a time.

        The emitter already collapses duplicate wakeups, but the sweep and an event
        can arrive from different directions at the same moment, and two passes
        interleaved would each see the deficit the other is already filling.
        """
        return self._locks.setdefault(compute_id, asyncio.Lock())


def _seed(machine: Machine) -> str:
    return f"{machine.private_host or machine.host}:{worker.PORT}"
