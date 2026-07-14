from __future__ import annotations

import asyncio
from dataclasses import dataclass

from litestar import Litestar
from litestar.di import Provide
from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import ScalarRenderPlugin

from skyward2.application import mock, ports
from skyward2.application.errors import SkywardError
from skyward2.server.controllers.blobs import BlobController
from skyward2.server.controllers.catalog import CatalogController
from skyward2.server.controllers.computes import ComputeController
from skyward2.server.controllers.events import EventController
from skyward2.server.controllers.functions import FunctionController
from skyward2.server.controllers.health import HealthController
from skyward2.server.controllers.nodes import NodeController
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
    catalog: ports.Catalog
    health: ports.Health
    reconciler: ports.Reconciler


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
        catalog=mock.MockCatalog(),
        health=mock.MockHealth(),
        reconciler=mock.MockReconciler(),
    )


def create_app(services: Services | None = None) -> Litestar:
    svc = services or mock_services()

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
        app.state.sweep = asyncio.create_task(sweep())

    async def on_shutdown(app: Litestar) -> None:
        task: asyncio.Task[None] | None = getattr(app.state, "sweep", None)
        if task:
            task.cancel()

    app = Litestar(
        path="/v1",
        route_handlers=[
            ComputeController,
            NodeController,
            FunctionController,
            BlobController,
            TaskController,
            EventController,
            CatalogController,
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
            "catalog": Provide(lambda: svc.catalog, sync_to_thread=False),
            "health": Provide(lambda: svc.health, sync_to_thread=False),
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
    return app


app = create_app()
