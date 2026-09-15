"""Node metrics as the daemon keeps them: recent samples as rows, closed windows as chunks, read back the same either way."""

import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
from litestar.testing import AsyncTestClient

import skyward as sky
from skyward.server.http.app import create_app, services
from skyward.server.persistence.db import connect
from skyward.server.persistence.metrics import MetricStore
from skyward.server.persistence.tables import MetricChunkRow, MetricSampleRow
from skyward.shared.schemas import Aggregate, MetricHistory, MetricSample
from tests.conftest import given

SECOND = 1_000
MINUTE = 60 * SECOND
WINDOW = 10 * MINUTE
GRACE = MINUTE
T0 = 1_757_887_200_000
"""A window boundary: every window here starts on a multiple of ``WINDOW``."""

COMPUTE = "cmp_metrics"


def reading(name: str, at: int, value: float, node: str = "nod_a") -> MetricSample:
    return MetricSample(node=node, name=name, at=at, value=value)


def points(history: MetricHistory, name: str, node: str = "nod_a") -> list[tuple[int, float]]:
    matching = [series for series in history.series if series.node == node and series.name == name]
    assert len(matching) <= 1, "one series per node and name"
    return [(at, value) for series in matching for at, value in zip(series.at, series.values, strict=True)]


async def recorded(store: MetricStore, *samples: MetricSample, compute: str = COMPUTE) -> None:
    store.add(compute, samples)
    await store.flush()


async def rows() -> int:
    return await MetricSampleRow.count()


async def chunks() -> int:
    return await MetricChunkRow.count()


@pytest.fixture
async def store(tmp_path: Path) -> MetricStore:
    await connect(tmp_path / "metrics.sqlite")
    return MetricStore(window=WINDOW, grace=GRACE)


@pytest.mark.local
def describe_recording() -> None:
    async def it_reads_back_what_was_recorded_in_the_order_it_was_measured(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0 + 4 * SECOND, 30.5), reading("cpu", T0, 10.0), reading("cpu", T0 + 2 * SECOND, 20.25))

        history = await store.series(COMPUTE, since=T0)

        assert points(history, "cpu") == [(T0, 10.0), (T0 + 2 * SECOND, 20.25), (T0 + 4 * SECOND, 30.5)]

    async def a_sample_recorded_twice_is_kept_once(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 10.0))
        await recorded(store, reading("cpu", T0, 10.0))

        history = await store.series(COMPUTE, since=T0)

        assert points(history, "cpu") == [(T0, 10.0)]

    async def another_computes_samples_are_not_this_ones(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 10.0), compute="cmp_other")

        history = await store.series(COMPUTE, since=T0)

        assert history.series == ()

    async def it_narrows_to_the_nodes_and_names_asked_for(store: MetricStore) -> None:
        await recorded(
            store,
            reading("cpu", T0, 1.0, node="nod_a"),
            reading("gpu_util", T0, 2.0, node="nod_a"),
            reading("cpu", T0, 3.0, node="nod_b"),
        )

        history = await store.series(COMPUTE, since=T0, nodes=("nod_b",), names=("cpu",))

        assert [(series.node, series.name) for series in history.series] == [("nod_b", "cpu")]

    async def it_reads_only_inside_the_range(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0 - SECOND, 1.0), reading("cpu", T0, 2.0), reading("cpu", T0 + MINUTE, 3.0))

        history = await store.series(COMPUTE, since=T0, until=T0 + MINUTE)

        assert points(history, "cpu") == [(T0, 2.0)]


@pytest.mark.local
def describe_compaction() -> None:
    async def it_folds_a_closed_window_into_one_chunk_that_reads_back_exactly(store: MetricStore) -> None:
        values = (
            reading("cpu", T0 + 37, 41.3),
            reading("cpu", T0 + 2_041, 0.30000000000000004),
            reading("cpu", T0 + 4_012, -7.0),
            reading("cpu", T0 + 6_990, 1e300),
            reading("gpu_power_w", T0 + 3_001, 287.45),
            reading("mem_used_mb", T0 + 1, 36_123.0),
        )
        await recorded(store, *values)
        before = await store.series(COMPUTE, since=T0)

        await store.compact(now=T0 + WINDOW + GRACE)

        assert (await rows(), await chunks()) == (0, 1)
        after = await store.series(COMPUTE, since=T0)
        assert after.series == before.series

    async def it_leaves_a_window_alone_until_the_grace_has_passed(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 1.0))

        await store.compact(now=T0 + WINDOW + GRACE - 1)

        assert (await rows(), await chunks()) == (1, 0)

    async def it_leaves_the_open_window_as_rows(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 1.0), reading("cpu", T0 + WINDOW, 2.0))

        await store.compact(now=T0 + WINDOW + GRACE)

        assert (await rows(), await chunks()) == (1, 1)
        assert points(await store.series(COMPUTE, since=T0), "cpu") == [(T0, 1.0), (T0 + WINDOW, 2.0)]

    async def a_sample_that_arrives_after_its_window_was_sealed_joins_the_chunk(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 1.0))
        await store.compact(now=T0 + WINDOW + GRACE)

        await recorded(store, reading("cpu", T0 + SECOND, 2.0))
        await store.compact(now=T0 + WINDOW + GRACE)

        assert (await rows(), await chunks()) == (0, 1)
        assert points(await store.series(COMPUTE, since=T0), "cpu") == [(T0, 1.0), (T0 + SECOND, 2.0)]

    async def a_window_read_again_after_it_was_sealed_is_not_counted_twice(store: MetricStore) -> None:
        window = (reading("cpu", T0, 1.0), reading("cpu", T0 + SECOND, 2.0))
        await recorded(store, *window)
        await store.compact(now=T0 + WINDOW + GRACE)

        await recorded(store, *window)
        await store.compact(now=T0 + WINDOW + GRACE)

        assert points(await store.series(COMPUTE, since=T0), "cpu") == [(T0, 1.0), (T0 + SECOND, 2.0)]


@pytest.mark.local
def describe_steps() -> None:
    @pytest.mark.parametrize(
        ("aggregate", "expected"),
        [
            ("avg", [(T0, 2.0), (T0 + MINUTE, 10.0)]),
            ("min", [(T0, 1.0), (T0 + MINUTE, 10.0)]),
            ("max", [(T0, 3.0), (T0 + MINUTE, 10.0)]),
            ("last", [(T0, 2.0), (T0 + MINUTE, 10.0)]),
        ],
    )
    async def a_bucket_is_one_value_per_step_starting_on_it(store: MetricStore, aggregate: Aggregate, expected: list[tuple[int, float]]) -> None:
        await recorded(
            store,
            reading("cpu", T0 + SECOND, 1.0),
            reading("cpu", T0 + 20 * SECOND, 3.0),
            reading("cpu", T0 + 40 * SECOND, 2.0),
            reading("cpu", T0 + MINUTE + SECOND, 10.0),
        )

        history = await store.series(COMPUTE, since=T0, step=MINUTE, aggregate=aggregate)

        assert points(history, "cpu") == expected

    async def buckets_read_chunks_and_rows_alike(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0 + WINDOW - SECOND, 4.0), reading("cpu", T0 + WINDOW, 6.0))
        await store.compact(now=T0 + WINDOW + GRACE)

        history = await store.series(COMPUTE, since=T0, step=2 * WINDOW)

        assert points(history, "cpu") == [(T0, 5.0)]


@pytest.mark.local
def describe_following() -> None:
    async def a_cursor_returns_only_what_was_recorded_after_it(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 1.0))
        first = await store.series(COMPUTE, since=T0)

        await recorded(store, reading("cpu", T0 + SECOND, 2.0), reading("gpu_util", T0 - MINUTE, 90.0))
        second = await store.after(COMPUTE, first.cursor)

        assert points(second, "cpu") == [(T0 + SECOND, 2.0)]
        assert points(second, "gpu_util") == [(T0 - MINUTE, 90.0)], "a sample measured earlier but recorded later is still new"
        assert second.reset is False
        assert (await store.after(COMPUTE, second.cursor)).series == ()

    async def a_cursor_from_a_range_misses_nothing_measured_ahead_of_the_daemon(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0 + 10 * WINDOW, 1.0))

        history = await store.series(COMPUTE, since=T0)

        assert points(history, "cpu") == [(T0 + 10 * WINDOW, 1.0)]

    async def a_cursor_behind_a_compaction_says_to_read_the_range_again(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 1.0))
        cursor = (await store.series(COMPUTE, since=T0)).cursor
        await recorded(store, reading("cpu", T0 + SECOND, 2.0))

        await store.compact(now=T0 + WINDOW + GRACE)

        assert (await store.after(COMPUTE, cursor)).reset is True

    async def a_cursor_still_follows_after_every_row_was_compacted_and_the_daemon_restarted(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 1.0), reading("cpu", T0 + SECOND, 2.0))
        cursor = (await store.series(COMPUTE, since=T0)).cursor
        await store.compact(now=T0 + WINDOW + GRACE)
        assert await rows() == 0

        restarted = MetricStore(window=WINDOW, grace=GRACE)
        await recorded(restarted, reading("cpu", T0 + 2 * WINDOW, 3.0))

        assert points(await restarted.after(COMPUTE, cursor), "cpu") == [(T0 + 2 * WINDOW, 3.0)]

    async def a_cursor_ahead_of_every_compaction_is_not_reset(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 1.0))
        cursor = (await store.series(COMPUTE, since=T0)).cursor

        await store.compact(now=T0 + WINDOW + GRACE)

        assert (await store.after(COMPUTE, cursor)).reset is False


@pytest.mark.local
def describe_latest() -> None:
    async def it_is_the_newest_sample_of_each_node_and_name(store: MetricStore) -> None:
        await recorded(
            store,
            reading("cpu", T0, 1.0),
            reading("cpu", T0 + SECOND, 2.0),
            reading("gpu_util", T0, 90.0),
            reading("cpu", T0, 5.0, node="nod_b"),
        )

        latest = await store.latest(COMPUTE)

        assert set(latest) == {reading("cpu", T0 + SECOND, 2.0), reading("gpu_util", T0, 90.0), reading("cpu", T0, 5.0, node="nod_b")}

    async def it_narrows_to_the_nodes_and_names_asked_for(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 1.0), reading("gpu_util", T0, 90.0), reading("cpu", T0, 5.0, node="nod_b"))

        latest = await store.latest(COMPUTE, nodes=("nod_a",), names=("cpu",))

        assert latest == (reading("cpu", T0, 1.0),)

    async def a_node_gone_quiet_is_answered_from_its_last_chunk(store: MetricStore) -> None:
        await recorded(store, reading("cpu", T0, 1.0), reading("cpu", T0 + WINDOW + SECOND, 2.0), reading("cpu", T0 + SECOND, 7.0, node="nod_b"))
        await store.compact(now=T0 + WINDOW + GRACE)

        latest = await store.latest(COMPUTE)

        assert set(latest) == {reading("cpu", T0 + WINDOW + SECOND, 2.0), reading("cpu", T0 + SECOND, 7.0, node="nod_b")}


@pytest.fixture
async def served(tmp_path: Path) -> AsyncIterator[tuple[AsyncTestClient, MetricStore, str]]:
    _, compute = await given(tmp_path / "skyward.sqlite")
    svc = services()
    async with AsyncTestClient(app=create_app(svc, logging=False)) as http:
        assert isinstance(svc.metrics, MetricStore)
        yield http, svc.metrics, compute.id


@pytest.mark.local
def describe_the_endpoints() -> None:
    async def a_range_answers_series_and_a_cursor_that_follows_it(served: tuple[AsyncTestClient, MetricStore, str]) -> None:
        http, store, compute = served
        start = int(time.time() * 1000) - MINUTE
        await recorded(store, reading("cpu", start, 41.5), compute=compute)

        first = await http.get(f"/v1/computes/{compute}/metrics", params={"since": start})
        await recorded(store, reading("cpu", start + 2 * SECOND, 42.0), compute=compute)
        second = await http.get(f"/v1/computes/{compute}/metrics", params={"after": first.json()["cursor"]})

        assert first.status_code == 200, first.text
        assert first.json()["series"] == [{"node": "nod_a", "name": "cpu", "at": [start], "values": [41.5]}]
        assert second.json()["series"] == [{"node": "nod_a", "name": "cpu", "at": [start + 2 * SECOND], "values": [42.0]}]
        assert second.json()["reset"] is False

    async def a_step_answers_buckets_over_a_long_range(served: tuple[AsyncTestClient, MetricStore, str]) -> None:
        http, store, compute = served
        start = int(time.time() * 1000) // MINUTE * MINUTE - 24 * 60 * MINUTE
        await recorded(store, reading("cpu", start + SECOND, 10.0), reading("cpu", start + 2 * SECOND, 20.0), compute=compute)

        answer = await http.get(f"/v1/computes/{compute}/metrics", params={"since": start, "step": MINUTE})

        assert answer.status_code == 200, answer.text
        assert answer.json()["series"] == [{"node": "nod_a", "name": "cpu", "at": [start], "values": [15.0]}]

    async def latest_answers_one_item_per_node_and_name(served: tuple[AsyncTestClient, MetricStore, str]) -> None:
        http, store, compute = served
        await recorded(store, reading("cpu", T0, 41.5), reading("cpu", T0 + SECOND, 43.0), compute=compute)

        answer = await http.get(f"/v1/computes/{compute}/metrics/latest")

        assert answer.status_code == 200, answer.text
        assert answer.json()["items"] == [{"node": "nod_a", "name": "cpu", "at": T0 + SECOND, "value": 43.0}]

    @pytest.mark.parametrize("path", ["/metrics?since=0", "/metrics/latest"])
    async def an_unknown_compute_is_not_found(served: tuple[AsyncTestClient, MetricStore, str], path: str) -> None:
        http, _, _ = served

        answer = await http.get(f"/v1/computes/cmp_nobody{path}")

        assert answer.status_code == 404

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"since": T0, "after": "1"},
            {"after": "not-a-cursor"},
            {"since": T0, "until": T0 + 61 * MINUTE},
            {"since": T0, "until": T0 + 2001 * SECOND, "step": SECOND},
        ],
        ids=["neither since nor after", "both since and after", "a cursor that is not one", "raw samples over an hour", "more than 2000 buckets"],
    )
    async def a_request_that_asks_for_something_unanswerable_is_refused(served: tuple[AsyncTestClient, MetricStore, str], params: dict[str, object]) -> None:
        http, _, compute = served

        answer = await http.get(f"/v1/computes/{compute}/metrics", params=params)

        assert answer.status_code == 400, answer.text


@pytest.mark.compute
@pytest.mark.xdist_group("pool")
def describe_a_live_pool() -> None:
    def its_nodes_can_be_read_back_and_followed(pool: sky.Compute, daemon: str) -> None:
        with httpx.Client(base_url=f"{daemon}/v1/computes/{pool.id}/metrics", timeout=10) as http:
            latest = _waited(lambda: http.get("/latest", params={"name": "cpu"}).json()["items"], lambda items: len({item["node"] for item in items}) == 2)
            first = http.get("", params={"since": min(item["at"] for item in latest) - MINUTE, "name": "cpu"}).json()
            followed = _waited(lambda: http.get("", params={"after": first["cursor"], "name": "cpu"}).json(), lambda answer: len(answer["series"]) == 2)

        assert {series["node"] for series in first["series"]} == {item["node"] for item in latest}
        for answer in (first, followed):
            assert all(list(series["at"]) == sorted(set(series["at"])) for series in answer["series"])
        assert all(min(later["at"]) > max(earlier["at"]) for earlier in first["series"] for later in followed["series"] if later["node"] == earlier["node"])
        assert followed["reset"] is False


def _waited[T](read: Callable[[], T], done: Callable[[T], bool], timeout: float = 60.0) -> T:
    deadline = time.monotonic() + timeout
    while not done(answer := read()):
        assert time.monotonic() < deadline, f"still waiting after {timeout:.0f}s: {answer}"
        time.sleep(1)
    return answer
