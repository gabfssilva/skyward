from __future__ import annotations

from litestar.events import EventListener, listener

from skyward2.application import ports
from skyward2.application.connector import Connector
from skyward2.application.machines import Machines
from skyward2.protocol.schemas import NodeState


def build_listeners(
    reconciler: ports.Reconciler,
    dispatcher: ports.Dispatcher,
    machines: Machines | None,
    connector: Connector | None,
) -> list[EventListener]:
    """Who reacts to what.

    One write, one event, one reactor, and the reactor's own write is what moves
    the row along. Nobody does two things: the reconciler decides a machine is
    wanted and writes ``requested``; something else buys it; something else logs
    into it. A chain, and every link of it is restartable, because every link reads
    the row before acting on it and does nothing if the row has already moved on.

    A listener that raises never escapes into a task group — the emitter isolates
    it, and the tick will offer the same row again.

    ``machines`` and ``connector`` are absent when the app is stood up against mocks
    to exercise the HTTP surface: there is no cloud to buy from and no machine to log
    into, and the events they would react to are never emitted.
    """

    @listener("compute.changed")
    async def on_compute_changed(compute_id: str) -> None:
        await reconciler.compute(compute_id)

    @listener("node.requested")
    async def on_node_requested(compute_id: str, node_id: str) -> None:
        if machines:
            await machines.create(compute_id, node_id)

    @listener("node.connect")
    async def on_node_connect(compute_id: str, node_id: str) -> None:
        """Somebody should be holding a live connection to this machine, and is not."""
        if connector:
            await connector.connect(compute_id, node_id)

    @listener("node.deleting")
    async def on_node_deleting(compute_id: str, node_id: str) -> None:
        if machines:
            await machines.terminate(compute_id, node_id)

    @listener("node.draining")
    async def on_node_draining(compute_id: str, node_id: str) -> None:
        await reconciler.compute(compute_id)

    @listener("node.observed")
    async def on_node_observed(compute_id: str, node_id: str, state: NodeState, error: str) -> None:
        """A node said what became of it.

        It goes through the bus rather than straight into the reconciler so that a
        machine flapping between states cannot spawn one pass per flap.
        """
        await reconciler.observed(compute_id, node_id, state, error)

    @listener("task.changed")
    async def on_task_changed(task_id: str) -> None:
        await dispatcher.task(task_id)

    @listener("compute.dispatch")
    async def on_compute_dispatch(compute_id: str) -> None:
        """A slot came free, or a machine arrived. Offer the queue what it can now have."""
        await dispatcher.resume(compute_id)

    return [
        on_compute_changed,
        on_node_requested,
        on_node_connect,
        on_node_deleting,
        on_node_draining,
        on_node_observed,
        on_task_changed,
        on_compute_dispatch,
    ]
