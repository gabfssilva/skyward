"""What an open compute screen reads and writes, against a daemon in this process."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import msgspec
import pytest

from skyward.api.v1 import ComputeResource, MetricHistoryResource, MetricSeries, NodeResource
from skyward.core import history, writes
from skyward.core.client import Client
from tests.conftest import held, machine

pytestmark = pytest.mark.local

NOW = datetime(2026, 9, 30, 12, 0, 10, tzinfo=UTC)
STEP = history.SPARK_STEP_MS
START = (int(NOW.timestamp() * 1000) // STEP - (history.SPARK_BUCKETS - 1)) * STEP


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[Client]:
    opened = await Client.embedded(tmp_path / "skyward.sqlite")
    try:
        yield opened
    finally:
        await opened.close()


class _Canned(Client):
    """A daemon that answers every read of a compute with the same two values."""

    def __init__(self, compute: ComputeResource, metrics: MetricHistoryResource) -> None:
        self._compute = compute
        self._metrics = metrics

    async def call[T](
        self,
        method: str,
        path: str,
        kind: type[T],
        /,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        urgent: bool = False,
        **query: object,
    ) -> T:
        answer = self._metrics if path.endswith("/metrics") else self._compute
        assert isinstance(answer, kind)
        return answer


def _series(node: str, values: dict[int, float], name: str = "gpu_util") -> MetricSeries:
    return MetricSeries(node=node, name=name, at=tuple(START + bucket * STEP for bucket in values), values=tuple(values.values()))


async def _snapshot(template: ComputeResource, nodes: tuple[NodeResource, ...], *series: MetricSeries) -> history.Snapshot:
    compute = msgspec.structs.replace(template, nodes=nodes)
    return await history.snapshot(_Canned(compute, MetricHistoryResource(series=series, cursor="", reset=False)), compute.id, NOW)


def describe_a_snapshot() -> None:
    async def it_reads_a_compute_the_daemon_holds(client: Client) -> None:
        created = await held(client, "training")

        snapshot = await history.snapshot(client, created.id, NOW)

        assert snapshot.compute.name == "training"
        assert snapshot.sparks == {}
        assert snapshot.averages == {}
        assert snapshot.idle is None
        assert snapshot.at == NOW

    async def it_puts_each_sample_in_the_bucket_its_time_falls_in(client: Client) -> None:
        template = await held(client, "training")
        early = MetricSeries(node="nod_a", name="cpu", at=(START - STEP, START, START + 3 * STEP + 1), values=(99.0, 10.0, 40.0))

        snapshot = await _snapshot(template, (machine("nod_a", 0, NOW),), early)

        buckets = snapshot.sparks["nod_a", "cpu"]
        assert len(buckets) == history.SPARK_BUCKETS
        assert buckets[0] == 10.0
        assert buckets[3] == 40.0
        assert [value for index, value in enumerate(buckets) if index not in (0, 3)] == [None] * (history.SPARK_BUCKETS - 2)

    async def it_averages_each_metric_over_the_nodes_still_alive(client: Client) -> None:
        template = await held(client, "training")
        nodes = (
            machine("nod_a", 0, NOW, metrics={"cpu": {"at": 1, "value": 20.0}, "gpu_util": {"at": 1, "value": 90.0}}),
            machine("nod_b", 1, NOW, metrics={"cpu": {"at": 1, "value": 40.0}}),
            machine("nod_c", 1, NOW, state="deleted", terminated_at=NOW, metrics={"cpu": {"at": 1, "value": 100.0}}),
        )

        snapshot = await _snapshot(template, nodes)

        assert snapshot.averages == {"cpu": 30.0, "gpu_util": 90.0}


def describe_idleness() -> None:
    last = history.SPARK_BUCKETS - 1

    async def a_gpu_that_worked_in_the_last_buckets_is_not_idle(client: Client) -> None:
        template = await held(client, "training")

        snapshot = await _snapshot(template, (machine("nod_a", 0, NOW),), _series("nod_a", {last - 2: 80.0, last: 1.0}))

        assert snapshot.idle is None

    async def it_is_idle_since_the_bucket_after_the_last_busy_one(client: Client) -> None:
        template = await held(client, "training")
        busy = last - history.IDLE_BUCKETS

        snapshot = await _snapshot(template, (machine("nod_a", 0, NOW),), _series("nod_a", {busy: 80.0, last: 1.0}))

        assert snapshot.idle == history.Idle(since=datetime.fromtimestamp((START + (busy + 1) * STEP) / 1000, UTC), capped=False)

    async def a_window_with_no_busy_bucket_began_idle_before_it(client: Client) -> None:
        template = await held(client, "training")

        snapshot = await _snapshot(template, (machine("nod_a", 0, NOW),), _series("nod_a", {last: 1.0}))

        assert snapshot.idle == history.Idle(since=datetime.fromtimestamp(START / 1000, UTC), capped=True)

    async def a_compute_that_reports_no_gpu_is_not_called_idle_by_it(client: Client) -> None:
        template = await held(client, "training")

        snapshot = await _snapshot(template, (machine("nod_a", 0, NOW),), _series("nod_a", {last: 1.0}, name="cpu"))

        assert snapshot.idle is None

    async def the_mean_is_over_the_nodes_alive(client: Client) -> None:
        template = await held(client, "training")
        nodes = (machine("nod_a", 0, NOW), machine("nod_b", 1, NOW, state="deleted", terminated_at=NOW))

        snapshot = await _snapshot(template, nodes, _series("nod_a", {last: 1.0}), _series("nod_b", {last: 95.0}))

        assert snapshot.idle is not None


def describe_grouping_a_node() -> None:
    @pytest.mark.parametrize(
        ("state", "group"),
        [
            ("ready", "active"),
            ("requested", "bootstrapping"),
            ("bootstrapping", "bootstrapping"),
            ("draining", "stopping"),
            ("deleting", "stopping"),
            ("deleted", "stopped"),
            ("lost", "failed"),
            ("failed", "failed"),
        ],
    )
    def it_reads_the_state(state: str, group: str) -> None:
        assert history.group(machine("nod_a", 0, NOW, state=state)) == group

    @pytest.mark.parametrize("state", ["deleting", "deleted"])
    def a_node_on_its_way_out_with_an_error_failed(state: str) -> None:
        error = {"code": "not_found", "message": "the machine went away", "retryable": False, "request_id": None, "details": None}

        assert history.group(machine("nod_a", 0, NOW, state=state, last_error=error)) == "failed"


def describe_the_pages() -> None:
    async def a_compute_with_no_tasks_has_an_empty_last_page(client: Client) -> None:
        created = await held(client, "training")

        page = await history.tasks(client, created.id)

        assert page.items == ()
        assert page.next is None
        assert dict(page.counts) == dict.fromkeys(history.COUNTED, 0)

    async def the_log_holds_what_happened_and_a_term_narrows_it(client: Client) -> None:
        created = await held(client, "training")

        everything = await history.log(client, created.id)
        nothing = await history.log(client, created.id, term="no line says this")

        assert everything.entries
        assert nothing.entries == ()

    async def a_live_compute_is_not_among_the_recent(client: Client) -> None:
        await held(client, "training")

        assert await history.recent(client) == ()


def describe_the_writes() -> None:
    async def scaling_moves_the_bounds_and_keeps_the_size_it_opened_at(client: Client) -> None:
        created = await held(client, "training")

        await writes.scale(client, created, 3, 5)

        scaled = await client.call("GET", f"/v1/computes/{created.id}", ComputeResource)
        assert (scaled.spec.nodes.initial, scaled.spec.nodes.min, scaled.spec.nodes.max) == (created.spec.nodes.initial, 3, 5)

    async def deleting_starts_the_compute_on_its_way_out(client: Client) -> None:
        created = await held(client, "training")

        await writes.delete(client, created.id)

        deleted = await client.call("GET", f"/v1/computes/{created.id}", ComputeResource)
        assert deleted.status.state in ("deleting", "deleted")
        assert not history.deletable(deleted)
