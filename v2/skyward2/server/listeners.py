from __future__ import annotations

from litestar.events import EventListener, listener

from skyward2.application import ports
from skyward2.protocol.schemas import NodeState


def build_listeners(reconciler: ports.Reconciler) -> list[EventListener]:
    """Wire the event bus to the reconciler.

    A listener enqueues; it never works and never raises. The work happens in the
    reconciler, where a failure becomes ``status.last_error`` in the store instead
    of an exception that takes the event bus down with it.
    """

    @listener("compute.changed")
    async def on_compute_changed(compute_id: str) -> None:
        await reconciler.compute(compute_id)

    @listener("node.changed")
    async def on_node_changed(compute_id: str, node_id: str) -> None:
        await reconciler.compute(compute_id)

    @listener("task.changed")
    async def on_task_changed(task_id: str) -> None:
        await reconciler.task(task_id)

    @listener("node.observed")
    async def on_node_observed(compute_id: str, node_id: str, state: NodeState, error: str) -> None:
        """A node said what became of it.

        It goes through the bus rather than straight into the reconciler so that a
        machine flapping between states cannot spawn one reconcile per flap.
        """
        await reconciler.observed(compute_id, node_id, state, error)

    return [on_compute_changed, on_node_changed, on_task_changed, on_node_observed]
