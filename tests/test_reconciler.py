"""One pass of the reconciler: what it reads, and what it lets go of."""

from pathlib import Path

import pytest

from skyward.server.application.machines import Machines
from skyward.server.application.reconciler import Reconciler, Wakeup
from skyward.server.persistence.computes import ComputeStore, GenerationStore
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.tasks import TaskStore
from skyward.shared.schemas import Node
from tests.conftest import given

pytestmark = pytest.mark.local


class CountingNodes(NodeStore):
    """A node store that counts how often the whole list is asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.listed = 0

    async def of(self, compute_id: str) -> tuple[Node, ...]:
        self.listed += 1
        return await super().of(compute_id)


def describe_one_pass() -> None:
    async def it_reads_the_nodes_twice_at_most(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Once after the provider is asked what became of them, once after the pass changed them."""
        computes, compute, nodes, reconciler = await _reconciler(tmp_path, monkeypatch)
        await reconciler.compute(compute)
        nodes.listed = 0

        await reconciler.compute(compute)

        assert nodes.listed == 2

    async def it_forgets_a_compute_once_it_is_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        computes, compute, nodes, reconciler = await _reconciler(tmp_path, monkeypatch)
        await reconciler.compute(compute)
        assert compute in reconciler._locks

        await computes.delete(compute, (await computes.get(compute)).revision, "once")
        for node in await nodes.of(compute):
            await nodes.observe(node.id, "deleted")
        await reconciler.compute(compute)

        assert (await computes.get(compute)).status.state == "deleted"
        assert compute not in reconciler._locks
        assert not any(node.id in reconciler._idle for node in await nodes.of(compute))


async def _reconciler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ComputeStore, str, CountingNodes, Reconciler]:
    """A reconciler over a compute of its own, with a provider that has nothing to say."""
    events = EventStore()
    computes, compute = await given(tmp_path / "skyward.sqlite", events=events)
    nodes, blobs = CountingNodes(), BlobStore()
    machines = Machines(computes, nodes, ProviderStore(), OfferCache(ProviderStore()), blobs, events)

    async def nothing(*_: object) -> None:
        return None

    monkeypatch.setattr(machines, "resolve", nothing)
    reconciler = Reconciler(computes, GenerationStore(computes), nodes, TaskStore(computes, nodes, blobs), machines, events, Wakeup())
    return computes, compute.id, nodes, reconciler
