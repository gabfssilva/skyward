"""Deleting a compute is event-driven; the tick is only a fallback.

When a node is terminated the compute must progress to ``deleted`` off the back of
the deletion event itself, not by waiting for the next reconcile tick. The real
reconciler and the real stores run here, with the cloud faked at its one boundary
(``Machines``), and no tick is started at all: reaching ``deleted`` proves the
delete path drives itself through events.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skyward.server.application import mock
from skyward.server.application.machines import Machines
from skyward.server.application.reconciler import Reconciler, Wakeup
from skyward.server.persistence.computes import ComputeStore, GenerationStore
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.tasks import TaskStore
from skyward.shared.schemas import (
    Compute,
    ComputeCreate,
    ComputeSpec,
    Image,
    Node,
    NodeBounds,
    ProviderRef,
    Spec,
)
from skyward.server.http.emitter import ReconcilingEventEmitter
from skyward.server.http.listeners import build_listeners

pytestmark = pytest.mark.unit

SPEC = ComputeSpec(
    specs=(Spec(provider=ProviderRef(kind="container"), cpus=1, memory_gb=1),),
    nodes=NodeBounds(desired=1),
    image=Image(python="3.13", skyward="local"),
)


class FakeCloud(Machines):
    """The real ``Machines`` with only the two verbs that touch a provider stubbed.

    ``terminate`` does to the store exactly what the real one does — marks the node
    ``deleted`` — without the EC2 call; ``release`` succeeds. Everything the
    reconciler reads still comes from the real stores.
    """

    def __init__(self, nodes: NodeStore) -> None:
        self._nodes = nodes
        self.released = 0

    async def resolve(self, compute: Compute, nodes: tuple[Node, ...]) -> None:
        return None

    async def terminate(self, compute_id: str, node_id: str) -> None:
        node = await self._nodes.get(compute_id, node_id)
        if node.state == "deleting":
            await self._nodes.observe(node_id, "deleted")

    async def release(self, compute_id: str) -> None:
        self.released += 1


async def test_terminating_the_last_node_deletes_the_compute_without_a_tick(tmp_path: Path):
    await connect(tmp_path / "skyward.sqlite")
    computes = ComputeStore()
    nodes = NodeStore()
    blobs = BlobStore()
    events = EventStore()
    tasks = TaskStore(computes, nodes, blobs)
    generations = GenerationStore(computes)
    machines = FakeCloud(nodes)

    wake = Wakeup()
    reconciler = Reconciler(
        computes=computes,
        generations=generations,
        nodes=nodes,
        tasks=tasks,
        machines=machines,
        events=events,
        wake=wake,
    )
    listeners = build_listeners(reconciler, mock.MockDispatcher(), machines, connector=None)
    on_node_deleting = next(listener for listener in listeners if "node.deleting" in listener.event_ids)

    async with ReconcilingEventEmitter(listeners) as emitter:
        wake.bind(lambda event, **payload: emitter.emit(event, **payload))

        created, _ = await computes.create(ComputeCreate(spec=SPEC, name=None), idempotency_key="k")
        current = await computes.get(created.id)
        await computes.delete(created.id, current.revision, idempotency_key="d")

        node = await nodes.request(created.id, created.generation)
        await nodes.observe(node.id, "deleting")

        await on_node_deleting.fn(compute_id=created.id, node_id=node.id)

        compute = await computes.get(created.id)
        assert compute.status.state == "deleted", "the deletion event drove the release; nothing waited for the tick"
        assert machines.released == 1
