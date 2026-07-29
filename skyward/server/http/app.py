from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, replace
from pathlib import Path

import msgspec
from litestar import Litestar
from litestar.di import Provide
from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import ScalarRenderPlugin
from litestar.types import Empty

from skyward.server.application import mock, ports
from skyward.server.application.connector import Connector
from skyward.server.application.dispatcher import Dispatcher
from skyward.server.application.health import Health
from skyward.server.application.machines import Machines
from skyward.server.application.metering import Meter
from skyward.server.application.reconciler import Reconciler, Wakeup
from skyward.server.application.runtimes import Files, Forward, Runtimes, Terminal
from skyward.server.http.controllers.blobs import BlobController
from skyward.server.http.controllers.computes import ComputeController
from skyward.server.http.controllers.events import EventController
from skyward.server.http.controllers.files import FileController
from skyward.server.http.controllers.forward import ForwardController
from skyward.server.http.controllers.functions import FunctionController
from skyward.server.http.controllers.health import HealthController
from skyward.server.http.controllers.nodes import NodeController
from skyward.server.http.controllers.offers import OfferController
from skyward.server.http.controllers.providers import ProviderController, ProviderKindController
from skyward.server.http.controllers.shell import ShellController
from skyward.server.http.controllers.tasks import TaskController
from skyward.server.http.emitter import ReconcilingEventEmitter
from skyward.server.http.exceptions import skyward_error_handler, unhandled_error_handler
from skyward.server.http.listeners import build_listeners
from skyward.server.persistence.computes import ComputeStore, GenerationStore
from skyward.server.persistence.db import DEFAULT_PATH, connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore, FunctionStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.store import now
from skyward.server.persistence.tasks import ExecutionStore, TaskStore
from skyward.shared import codec
from skyward.shared.errors import SkywardError

TICK_SECONDS = 5
METER_SECONDS = 10


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
    dispatcher: ports.Dispatcher
    wake: Wakeup
    machines: Machines | None = None
    connector: Connector | None = None
    runtimes: Runtimes | None = None
    meter: Meter | None = None
    forwarder: ports.Forwarder | None = None
    shell: ports.Shell | None = None
    files: ports.Files | None = None


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
        dispatcher=mock.MockDispatcher(),
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
        line = {"compute": compute, "node": node, "content": content, **({"task": task} if task else {})}
        payload = await codec.json(dict[str, str]).encode(line)
        await events.record("node.console", payload, compute=compute, task=task)

    async def phased(compute: str, node: str, event: str, phase: str, error: str | None) -> None:
        """A bootstrap phase turning over is recorded, so a late subscriber replays the checklist."""
        mark = {"compute": compute, "node": node, "event": event, "phase": phase, "at": now().isoformat(), **({"error": error} if error else {})}
        payload = await codec.json(dict[str, str]).encode(mark)
        await events.record("node.phase", payload, compute=compute)

    async def sampled(compute: str, node: str, name: str, value: float) -> None:
        """A gauge reading goes out to whoever is watching, and is not written down."""
        payload = msgspec.json.encode({"compute": compute, "node": node, "name": name, "value": value})
        await events.publish("node.metrics", payload, compute=compute)

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
        sample=lambda compute, node, name, value: spoken(sampled(compute, node, name, value)),
        phase=lambda compute, node, event, phase, error: spoken(phased(compute, node, event, phase, error)),
    )

    offers = OfferCache(providers)
    generations = GenerationStore(computes)
    machines = Machines(computes=computes, nodes=nodes, providers=providers, offers=offers, blobs=blobs)

    return Services(
        computes=computes,
        generations=generations,
        nodes=nodes,
        functions=FunctionStore(blobs),
        blobs=blobs,
        tasks=tasks,
        executions=ExecutionStore(tasks),
        events=events,
        providers=providers,
        offers=offers,
        health=Health(providers),
        reconciler=Reconciler(
            computes=computes,
            generations=generations,
            nodes=nodes,
            tasks=tasks,
            machines=machines,
            events=events,
            wake=wake,
        ),
        dispatcher=Dispatcher(
            computes=computes,
            tasks=tasks,
            nodes=nodes,
            blobs=blobs,
            events=events,
            runtimes=runtimes,
            wake=wake,
        ),
        wake=wake,
        machines=machines,
        connector=Connector(computes=computes, nodes=nodes, runtimes=runtimes, blobs=blobs),
        runtimes=runtimes,
        meter=Meter(computes=computes, nodes=nodes, events=events),
        forwarder=Forward(runtimes),
        shell=Terminal(runtimes),
        files=Files(runtimes),
    )


def create_app(svc: Services | None = None, database: Path | None = None, logging: bool = False) -> Litestar:
    """The daemon as an ASGI app.

    ``logging`` is off by default because this module is imported into the user's
    process — the embedded client builds an app right here — and Litestar's default
    config turns the *root* logger up to INFO for everyone the moment an app is
    constructed. A guest does not get to do that. A standalone daemon, which owns its
    process, is welcome to ask for it on.
    """
    svc = svc or mock_services()

    async def tick() -> None:
        """The clock, and the safety net that makes events optional for correctness.

        An event is a wakeup: if one is lost — a crash between the commit and the
        emit, a listener that died, a restart — the intent is still in the store and
        this finds it. That is what buys the right to skip an outbox, an effects
        table and delivery leases.

        A deadline is the other thing no event can carry. Nobody writes it down when it
        passes, so the clock is the only thing in a position to notice.
        """
        while True:
            await asyncio.sleep(TICK_SECONDS)
            await svc.tasks.expire()
            computes, tasks = await svc.reconciler.unsettled()
            for compute_id in computes:
                app.emit("compute.changed", compute_id=compute_id)
            for task_id in tasks:
                app.emit("task.changed", task_id=task_id)

    async def metered(meter: Meter) -> None:
        """The cost gauge: every few seconds, what each live compute has accrued."""
        while True:
            await asyncio.sleep(METER_SECONDS)
            await meter.sample()

    async def on_startup(app: Litestar) -> None:
        if database is not None:
            await connect(database)
        app.state.tick = asyncio.create_task(tick())
        if svc.meter:
            app.state.meter = asyncio.create_task(metered(svc.meter))

    async def on_shutdown(app: Litestar) -> None:
        for name in ("tick", "meter"):
            task: asyncio.Task[None] | None = getattr(app.state, name, None)
            if task:
                task.cancel()
        if svc.runtimes:
            await svc.runtimes.shutdown()

    forwarding = [ForwardController] if svc.forwarder else []
    shells = [ShellController] if svc.shell else []
    filing = [FileController] if svc.files else []

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
            *forwarding,
            *shells,
            *filing,
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
            "reconciler": Provide(lambda: svc.reconciler, sync_to_thread=False),
            "dispatcher": Provide(lambda: svc.dispatcher, sync_to_thread=False),
            "wake": Provide(lambda: svc.wake, sync_to_thread=False),
            **({"forwarder": Provide(lambda: svc.forwarder, sync_to_thread=False)} if svc.forwarder else {}),
            **({"shell": Provide(lambda: svc.shell, sync_to_thread=False)} if svc.shell else {}),
            **({"files": Provide(lambda: svc.files, sync_to_thread=False)} if svc.files else {}),
        },
        listeners=build_listeners(svc.reconciler, svc.dispatcher, svc.machines, svc.connector),
        event_emitter_backend=ReconcilingEventEmitter,
        exception_handlers={SkywardError: skyward_error_handler, Exception: unhandled_error_handler},
        on_startup=[on_startup],
        on_shutdown=[on_shutdown],
        logging_config=Empty if logging else None,
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
