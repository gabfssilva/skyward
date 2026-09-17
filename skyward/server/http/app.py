from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path

from litestar import Litestar, Router
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
from skyward.server.http.controllers.console import console
from skyward.server.http.controllers.events import EventController
from skyward.server.http.controllers.files import FileController
from skyward.server.http.controllers.forward import ForwardController
from skyward.server.http.controllers.functions import FunctionController
from skyward.server.http.controllers.health import HealthController
from skyward.server.http.controllers.metrics import MetricController
from skyward.server.http.controllers.nodes import NodeController
from skyward.server.http.controllers.offers import AcceleratorController, OfferController
from skyward.server.http.controllers.providers import ProviderController, ProviderKindController
from skyward.server.http.controllers.shell import ShellController
from skyward.server.http.controllers.tasks import TaskController
from skyward.server.http.emitter import ReconcilingEventEmitter
from skyward.server.http.exceptions import skyward_error_handler, unhandled_error_handler
from skyward.server.http.listeners import build_listeners
from skyward.server.http.openapi import TAGS, describe
from skyward.server.http.references import identified
from skyward.server.persistence.computes import ComputeStore, GenerationStore
from skyward.server.persistence.db import DEFAULT_PATH, connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore, FunctionStore
from skyward.server.persistence.metrics import MetricStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.store import now
from skyward.server.persistence.tasks import ExecutionStore, TaskStore
from skyward.shared.errors import SkywardError
from skyward.shared.events import ConsoleEvent, MetricEvent, PhaseEvent
from skyward.shared.observability import LogConfig, level, logger, setup_logging
from skyward.shared.schemas import MetricSample, PhaseMark
from skyward.worker.journal import Console, Metric

logger = logger.bind(component="daemon")

CONSOLE = Path(__file__).resolve().parent / "console"
"""Where ``task web:build`` puts the browser console, inside the package so the wheel carries it."""

TICK_SECONDS = 5
METER_SECONDS = 10
FLUSH_SECONDS = 2
"""How long a node's reading waits in memory before it is written: the delay a reader polling the metrics sees."""
COMPACT_SECONDS = 60


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
    metrics: ports.Metrics
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
        metrics=mock.MockMetrics(),
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

    events = EventStore()
    metrics = MetricStore()
    nodes = NodeStore()
    computes = ComputeStore(events, nodes)
    blobs = BlobStore()
    providers = ProviderStore()
    tasks = TaskStore(computes, nodes, blobs)

    async def console(compute: str, node: str, lines: tuple[Console, ...]) -> None:
        """A node names the execution it was running; the line is recorded under that execution's task too."""
        owners = await tasks.owners({line.task for line in lines if line.task})
        await events.record_all(
            tuple(
                ConsoleEvent(compute=compute, node=node, content=line.content, task=owners.get(line.task) if line.task else None, execution=line.task)
                for line in lines
            )
        )

    async def phased(compute: str, node: str, event: PhaseMark, phase: str, error: str | None) -> None:
        """A bootstrap phase turning over is recorded, so a late subscriber replays the checklist."""
        await events.record(PhaseEvent(compute=compute, node=node, event=event, phase=phase, at=now(), error=error))

    async def sampled(compute: str, node: str, reading: Metric) -> None:
        """A gauge reading goes out to whoever is watching, and is held for the metric store's next write."""
        metrics.add(compute, (MetricSample(node=node, name=reading.name, at=reading.at, value=reading.value),))
        await events.publish(MetricEvent(compute=compute, node=node, name=reading.name, value=reading.value))

    runtimes = Runtimes(
        listener=lambda compute, node, state, error: wake(
            "node.observed", compute_id=compute, node_id=node, state=state, error=error,
        ),
        output=console,
        sample=sampled,
        phase=phased,
    )

    offers = OfferCache(providers)
    generations = GenerationStore(computes)
    machines = Machines(computes=computes, nodes=nodes, providers=providers, offers=offers, blobs=blobs, events=events)

    return Services(
        computes=computes,
        generations=generations,
        nodes=nodes,
        functions=FunctionStore(blobs),
        blobs=blobs,
        tasks=tasks,
        executions=ExecutionStore(tasks),
        events=events,
        metrics=metrics,
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


def create_app(svc: Services | None = None, database: Path | None = None, logging: bool = False, console_at: Path | None = None) -> Litestar:
    """The daemon as an ASGI app.

    ``logging`` is off by default because this module is imported into the user's
    process — the embedded client builds an app right here — and Litestar's default
    config turns the *root* logger up to INFO for everyone the moment an app is
    constructed. A guest does not get to do that. A standalone daemon, which owns its
    process, is welcome to ask for it on.

    ``console_at`` is a built browser console to serve at the root, beside the API
    under ``/v1``. Only a daemon somebody can open in a browser has a use for one.
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
            await svc.dispatcher.expire()
            computes, tasks = await svc.reconciler.unsettled()
            logger.debug("tick: {} unsettled computes, {} unsettled tasks", len(computes), len(tasks))
            for compute_id in computes:
                app.emit("compute.changed", compute_id=compute_id)
            for task_id in tasks:
                app.emit("task.changed", task_id=task_id)

    async def metered(meter: Meter) -> None:
        """The cost gauge: every few seconds, what each live compute has accrued."""
        while True:
            await asyncio.sleep(METER_SECONDS)
            await meter.sample()

    async def every(seconds: float, work: Callable[[], Awaitable[None]], what: str) -> None:
        """Do ``work`` on a period, and keep doing it after a pass that failed.

        A metric write that meets a locked database costs a couple of seconds of
        samples; a loop that died of it would cost every sample after them.
        """
        while True:
            await asyncio.sleep(seconds)
            try:
                await work()
            except Exception:
                logger.exception("could not {}", what)

    async def on_startup(app: Litestar) -> None:
        if database is not None:
            await connect(database)
        app.state.tick = asyncio.create_task(tick())
        app.state.rechunk = asyncio.create_task(svc.blobs.rechunk())
        app.state.flush = asyncio.create_task(every(FLUSH_SECONDS, svc.metrics.flush, "write metrics"))
        app.state.compact = asyncio.create_task(every(COMPACT_SECONDS, svc.metrics.compact, "compact metrics"))
        if svc.meter:
            app.state.meter = asyncio.create_task(metered(svc.meter))

    async def on_shutdown(app: Litestar) -> None:
        for name in ("tick", "rechunk", "meter", "flush", "compact"):
            task: asyncio.Task[None] | None = getattr(app.state, name, None)
            if task:
                task.cancel()
        if svc.runtimes:
            await svc.runtimes.shutdown()
        await svc.metrics.flush()

    forwarding = [ForwardController] if svc.forwarder else []
    shells = [ShellController] if svc.shell else []
    filing = [FileController] if svc.files else []

    api = Router(
        path="/v1",
        route_handlers=[
            ComputeController,
            NodeController,
            FunctionController,
            BlobController,
            TaskController,
            EventController,
            MetricController,
            ProviderController,
            ProviderKindController,
            OfferController,
            AcceleratorController,
            HealthController,
            *forwarding,
            *shells,
            *filing,
        ],
    )

    app = Litestar(
        route_handlers=[api, *([console(console_at)] if console_at else [])],
        dependencies={
            "computes": Provide(lambda: svc.computes, sync_to_thread=False),
            "compute_id": Provide(identified),
            "generations": Provide(lambda: svc.generations, sync_to_thread=False),
            "nodes": Provide(lambda: svc.nodes, sync_to_thread=False),
            "functions": Provide(lambda: svc.functions, sync_to_thread=False),
            "blobs": Provide(lambda: svc.blobs, sync_to_thread=False),
            "tasks": Provide(lambda: svc.tasks, sync_to_thread=False),
            "executions": Provide(lambda: svc.executions, sync_to_thread=False),
            "events": Provide(lambda: svc.events, sync_to_thread=False),
            "metrics": Provide(lambda: svc.metrics, sync_to_thread=False),
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
            path="/v1/schema",
            render_plugins=[ScalarRenderPlugin()],
            tags=list(TAGS),
        ),
    )

    describe(app)
    svc.wake.bind(app.emit)
    return app


@dataclass(frozen=True, slots=True)
class Standalone:
    """The app a standalone daemon serves, and what its server says to it before cutting its connections."""

    app: Litestar
    closing: Callable[[], None]


def daemon() -> Standalone:
    """The app a standalone daemon serves.

    The one deployment that owns its process is also the only one allowed to say where
    logs go: ``create_app`` is imported into the user's process by the embedded client,
    and a guest does not get to install sinks on the host application's behalf.

    The console sink is attached only on a terminal: a detached daemon's stdout is
    ``server.log``, a file nothing rotates, and the rotating log file already has
    every line the console would print.
    """
    setup_logging(LogConfig(level=level(os.environ.get("SKYWARD_LOG_LEVEL")), console=sys.stdout.isatty()))
    database = Path(env) if (env := os.environ.get("SKYWARD_DATABASE")) else DEFAULT_PATH
    svc = services()
    return Standalone(create_app(svc, database=database, console_at=CONSOLE if (CONSOLE / "index.html").is_file() else None), svc.tasks.close)
