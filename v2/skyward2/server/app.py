from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, replace
from pathlib import Path

from litestar import Litestar
from litestar.di import Provide
from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import ScalarRenderPlugin

from skyward2.application import mock, ports
from skyward2.application.errors import SkywardError
from skyward2.application.health import Health
from skyward2.application.reconciler import Reconciler, Wakeup
from skyward2.application.runtimes import Runtimes
from skyward2.persistence.computes import ComputeStore, GenerationStore
from skyward2.persistence.db import DEFAULT_PATH, connect
from skyward2.persistence.events import EventStore
from skyward2.persistence.functions import BlobStore, FunctionStore
from skyward2.persistence.nodes import NodeStore
from skyward2.persistence.offers import OfferCache
from skyward2.persistence.providers import ProviderStore
from skyward2.persistence.tasks import ExecutionStore, TaskStore
from skyward2.protocol import codec
from skyward2.server.controllers.blobs import BlobController
from skyward2.server.controllers.computes import ComputeController
from skyward2.server.controllers.events import EventController
from skyward2.server.controllers.functions import FunctionController
from skyward2.server.controllers.health import HealthController
from skyward2.server.controllers.nodes import NodeController
from skyward2.server.controllers.offers import OfferController
from skyward2.server.controllers.providers import ProviderController, ProviderKindController
from skyward2.server.controllers.tasks import TaskController
from skyward2.server.emitter import ReconcilingEventEmitter
from skyward2.server.exceptions import skyward_error_handler
from skyward2.server.listeners import build_listeners

SWEEP_SECONDS = 10


@dataclass(frozen=True, slots=True)
class Services:
    computes: ports.Computes
    generations: ports.Generations
    nodes: ports.Nodes
    functions: ports.Functions
    blobs: ports.Blobs
    tasks: ports.Tasks
    executions: ports.Executions
    events: ports.Events
    providers: ports.Providers
    offers: ports.Offers
    health: ports.Health
    reconciler: ports.Reconciler
    wake: Wakeup
    runtimes: Runtimes | None = None


def mock_services() -> Services:
    return Services(
        computes=mock.MockComputes(),
        generations=mock.MockGenerations(),
        nodes=mock.MockNodes(),
        functions=mock.MockFunctions(),
        blobs=mock.MockBlobs(),
        tasks=mock.MockTasks(),
        executions=mock.MockExecutions(),
        events=mock.MockEvents(),
        providers=mock.MockProviders(),
        offers=mock.MockOffers(),
        health=mock.MockHealth(),
        reconciler=mock.MockReconciler(),
        wake=Wakeup(),
    )


def with_real(**overrides: object) -> Services:
    """Mock everything that is not built yet, keep the real thing for what is."""
    return replace(mock_services(), **overrides)


def services() -> Services:
    """The real thing, wired.

    The store is not opened here: :func:`connect` is the app's business, because a
    daemon that cannot reach its database should fail on startup and not on the
    first request.
    """
    wake = Wakeup()

    computes = ComputeStore()
    nodes = NodeStore()
    blobs = BlobStore()
    providers = ProviderStore()
    events = EventStore()
    tasks = TaskStore(computes, nodes, blobs)

    async def console(compute: str, node: str, content: str, task: str | None) -> None:
        payload = await codec.json(dict[str, str]).encode({"compute": compute, "node": node, "content": content})
        await events.record("node.console", payload, compute=compute, task=task)

    def spoken(recording: Coroutine[None, None, None]) -> None:
        """A node's output goes straight to the log, not through the wakeup bus.

        The emitter coalesces identical payloads, and two identical lines of a
        user's `print` are not a duplicate event — they are two lines. This is the
        one thing a node emits that is data rather than a trigger.
        """
        asyncio.get_running_loop().create_task(recording)

    runtimes = Runtimes(
        listener=lambda compute, node, state, error: wake(
            "node.observed", compute_id=compute, node_id=node, state=state, error=error or "",
        ),
        output=lambda compute, node, content, task: spoken(console(compute, node, content, task)),
    )

    return Services(
        computes=computes,
        generations=GenerationStore(computes),
        nodes=nodes,
        functions=FunctionStore(blobs),
        blobs=blobs,
        tasks=tasks,
        executions=ExecutionStore(tasks),
        events=events,
        providers=providers,
        offers=OfferCache(providers),
        health=Health(providers),
        reconciler=Reconciler(
            computes=computes,
            generations=GenerationStore(computes),
            nodes=nodes,
            tasks=tasks,
            blobs=blobs,
            providers=providers,
            offers=OfferCache(providers),
            events=events,
            runtimes=runtimes,
            wake=wake,
        ),
        wake=wake,
        runtimes=runtimes,
    )


def create_app(svc: Services | None = None, database: Path | None = None) -> Litestar:
    svc = svc or mock_services()

    async def sweep() -> None:
        """The safety net that makes events optional for correctness.

        An event is a wakeup: if one is lost — crash between commit and emit, a
        listener that died, a restart — the intent is still in the store and the
        sweep finds it. This is what buys us the right to skip an outbox, an
        effects table and delivery leases.
        """
        while True:
            await asyncio.sleep(SWEEP_SECONDS)
            computes, tasks = await svc.reconciler.unsettled()
            for compute_id in computes:
                app.emit("compute.changed", compute_id=compute_id)
            for task_id in tasks:
                app.emit("task.changed", task_id=task_id)

    async def on_startup(app: Litestar) -> None:
        if database is not None:
            await connect(database)
        app.state.sweep = asyncio.create_task(sweep())

    async def on_shutdown(app: Litestar) -> None:
        task: asyncio.Task[None] | None = getattr(app.state, "sweep", None)
        if task:
            task.cancel()
        if svc.runtimes:
            await svc.runtimes.shutdown()

    app = Litestar(
        path="/v1",
        route_handlers=[
            ComputeController,
            NodeController,
            FunctionController,
            BlobController,
            TaskController,
            EventController,
            ProviderController,
            ProviderKindController,
            OfferController,
            HealthController,
        ],
        dependencies={
            "computes": Provide(lambda: svc.computes, sync_to_thread=False),
            "generations": Provide(lambda: svc.generations, sync_to_thread=False),
            "nodes": Provide(lambda: svc.nodes, sync_to_thread=False),
            "functions": Provide(lambda: svc.functions, sync_to_thread=False),
            "blobs": Provide(lambda: svc.blobs, sync_to_thread=False),
            "tasks": Provide(lambda: svc.tasks, sync_to_thread=False),
            "executions": Provide(lambda: svc.executions, sync_to_thread=False),
            "events": Provide(lambda: svc.events, sync_to_thread=False),
            "providers": Provide(lambda: svc.providers, sync_to_thread=False),
            "offers": Provide(lambda: svc.offers, sync_to_thread=False),
            "health": Provide(lambda: svc.health, sync_to_thread=False),
            "wake": Provide(lambda: svc.wake, sync_to_thread=False),
        },
        listeners=build_listeners(svc.reconciler),
        event_emitter_backend=ReconcilingEventEmitter,
        exception_handlers={SkywardError: skyward_error_handler},
        on_startup=[on_startup],
        on_shutdown=[on_shutdown],
        openapi_config=OpenAPIConfig(
            title="Skyward Control Plane",
            version="0.1.0",
            description=(
                "Two families of resource.\n\n"
                "**Declarative** (`compute`, `node`) carry `spec` (intent) and `status` (observation). `PATCH` only "
                "touches `spec`; `status` is written by the reconciler. There is no `operation` resource — `generation` "
                "vs `status.observed_generation` is the progress.\n\n"
                "**Imperative** (`task`) are append-only facts with one terminal outcome. `executions` are the physical "
                "attempts; retrying creates an execution, never a task, so a `Future` keeps a stable handle."
            ),
            path="/schema",
            render_plugins=[ScalarRenderPlugin()],
        ),
    )

    svc.wake.bind(app.emit)
    return app


app = create_app(services(), database=DEFAULT_PATH)
