"""The daemon's own log: numbered on the way to the file, read back newest first, filtered, summarized, followed and served."""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from skyward.server.http.app import create_app, with_real
from skyward.server.persistence import daemonlog as store
from skyward.server.persistence.daemonlog import DaemonLogStore
from skyward.shared.observability import Entry, Failure, LogFile, Query, Severity, entries, logger, summarize
from skyward.shared.observability.logfile import failure
from tests.conftest import logfile

pytestmark = pytest.mark.local

WAIT = 5.0
T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)

type Runs = AsyncIterator[tuple[Entry, ...]]


def entry(
    sequence: int,
    *,
    seconds: float = 0,
    level: Severity = "INFO",
    site: str = "reconciler:_pass:197",
    component: str | None = "reconciler",
    compute: str | None = None,
    node: str | None = None,
    message: str = "pass",
    exception: Failure | None = None,
) -> Entry:
    return Entry(
        sequence=sequence,
        at=T0 + timedelta(seconds=seconds),
        level=level,
        site=site,
        logger="skyward.log",
        component=component,
        compute=compute,
        node=node,
        fields={},
        message=message,
        exception=exception,
    )


def refused() -> Failure:
    deployed = ExceptionGroup("runpod deployed 0 pods", [RuntimeError("400 Bad Request: no longer any instances")])
    return failure(ExceptionGroup("no market could place a runpod machine", [deployed]))


async def written(file: LogFile, sequence: int) -> None:
    """Until the file has written ``sequence``, and its followers have been handed it."""
    async with asyncio.timeout(WAIT):
        while file.sequence < sequence:
            await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)


async def run(stream: Runs) -> tuple[Entry, ...]:
    return await asyncio.wait_for(anext(stream), WAIT)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "skyward.log"


def describe_log_file() -> None:
    def it_numbers_every_entry_in_the_order_it_writes_them(path: Path) -> None:
        with logfile(path):
            for index in range(50):
                logger.info("line {index}", index=index)

        read = list(entries(path))
        assert [each.message for each in read] == [f"line {index}" for index in reversed(range(50))]
        assert [each.sequence for each in read] == list(range(50, 0, -1))

    def it_carries_on_the_numbering_when_the_file_is_opened_again(path: Path) -> None:
        with logfile(path):
            logger.info("before")
        with logfile(path) as file:
            assert file.sequence == 1
            logger.info("after")

        assert [each.sequence for each in entries(path)] == [2, 1]

    def it_reads_across_the_files_it_rolled_over_into(path: Path) -> None:
        with logfile(path, size=4096):
            for index in range(200):
                logger.info("line {index} {padding}", index=index, padding="x" * 60)

        assert path.with_name("skyward.log.1.gz").exists()
        sequences = [each.sequence for each in entries(path)]
        assert sequences[0] == 200
        assert sequences == list(range(200, 200 - len(sequences), -1)), "no entry is missing or repeated where a file ends"

    def it_lifts_the_ids_a_reader_filters_on_out_of_the_bound_fields(path: Path) -> None:
        with logfile(path):
            logger.bind(component="machines", compute_id="cmp_a", node_id="nod_a", provider="aws").warning("bought {n}", n=4)

        (written_entry,) = entries(path)
        assert (written_entry.component, written_entry.compute, written_entry.node) == ("machines", "cmp_a", "nod_a")
        assert written_entry.fields == {"provider": "aws"}
        assert (written_entry.level, written_entry.message) == ("WARNING", "bought 4")
        assert written_entry.site.startswith("test_daemon_log:")

    def it_keeps_the_exception_a_line_was_logged_with(path: Path) -> None:
        with logfile(path):
            try:
                raise KeyError("missing")
            except KeyError:
                logger.exception("lookup failed")

        (written_entry,) = entries(path)
        assert written_entry.message == "lookup failed"
        assert written_entry.exception is not None
        assert written_entry.exception.type == "KeyError"
        assert "Traceback" in written_entry.exception.traceback
        assert written_entry.group == f"{written_entry.site}|KeyError"

    def it_tells_a_follower_what_it_writes_until_it_stops_following(path: Path) -> None:
        heard: list[str] = []
        first = threading.Event()

        def follower(written_entry: Entry) -> None:
            heard.append(written_entry.message)
            first.set()

        with logfile(path) as file:
            stop = file.subscribe(follower)
            logger.info("one")
            assert first.wait(WAIT)
            stop()
            logger.info("two")

        assert heard == ["one"]


def describe_entries() -> None:
    def it_leaves_a_last_line_that_is_still_being_written(path: Path) -> None:
        with logfile(path):
            logger.info("whole")
        with path.open("ab") as file:
            file.write(b'{"sequence":2,"at":')

        assert [each.message for each in entries(path)] == ["whole"]

    def it_stops_where_the_file_was_still_text(path: Path) -> None:
        path.parent.mkdir(parents=True)
        path.write_text("2026-09-23 18:22:52.283 | ERROR    | reconciler:compute:134 [component=reconciler] - reconcile failed\n")
        with logfile(path) as file:
            assert file.sequence == 0
            logger.info("the first entry")

        assert [(each.sequence, each.message) for each in entries(path)] == [(1, "the first entry")]

    def it_reads_nothing_where_there_is_no_file(path: Path) -> None:
        assert list(entries(path)) == []


def describe_failure() -> None:
    def it_names_the_innermost_exception_of_a_group_of_groups() -> None:
        failed = refused()

        assert failed.type == "ExceptionGroup"
        assert failed.message == "no market could place a runpod machine (1 sub-exception)"
        assert failed.cause == "RuntimeError: 400 Bad Request: no longer any instances"

    def it_follows_what_an_exception_was_raised_from() -> None:
        try:
            try:
                raise ConnectionRefusedError("connect call failed")
            except ConnectionRefusedError as refusal:
                raise RuntimeError("cannot reach member") from refusal
        except RuntimeError as exc:
            failed = failure(exc)

        assert failed.cause == "ConnectionRefusedError: connect call failed"

    def it_has_no_cause_when_nothing_is_under_it() -> None:
        assert failure(ValueError("plain")).cause is None


def describe_query() -> None:
    def it_keeps_the_level_asked_for_and_the_ones_above() -> None:
        lines = [entry(1, level="DEBUG"), entry(2, level="INFO"), entry(3, level="WARNING"), entry(4, level="ERROR")]

        assert [each.sequence for each in lines if Query(level="WARNING").matches(each)] == [3, 4]

    def it_keeps_any_one_of_the_components_named() -> None:
        lines = [entry(1, component="ssh"), entry(2, component="machines"), entry(3, component="reconciler")]

        assert [each.sequence for each in lines if Query(components=frozenset({"ssh", "machines"})).matches(each)] == [1, 2]

    def it_drops_the_hidden_groups_and_nothing_else() -> None:
        routine = entry(1, site="dispatcher:_launch:347")
        lost = entry(2, site="dispatcher:_lost:569", level="WARNING")

        wanted = Query(hidden=frozenset({routine.group}))
        assert (wanted.matches(routine), wanted.matches(lost)) == (False, True)

    def it_tells_a_group_by_the_exception_it_was_failing_with() -> None:
        refusal = entry(1, site="emitter:_invoke:95", exception=refused())
        unreachable = entry(2, site="emitter:_invoke:95", exception=failure(RuntimeError("cannot reach member")))

        assert refusal.group != unreachable.group
        assert Query(groups=frozenset({refusal.group})).matches(unreachable) is False

    def it_finds_a_term_in_the_exception_as_well_as_the_message_ignoring_case() -> None:
        failed = entry(1, message="event listener failed: on_node_requested", exception=refused())

        assert Query(contains=("NO LONGER ANY INSTANCES",)).matches(failed)
        assert Query(contains=("nothing like it", "on_node_requested")).matches(failed)
        assert not Query(contains=("nothing like it",)).matches(failed)

    def it_ends_before_until() -> None:
        wanted = Query(since=T0, until=T0 + timedelta(seconds=10))

        assert [wanted.matches(entry(1, seconds=seconds)) for seconds in (-1, 0, 9, 10)] == [False, True, True, False]


def describe_summarize() -> None:
    def it_counts_a_repeated_call_as_one_group_and_puts_the_failures_first() -> None:
        lines = [entry(sequence, seconds=sequence) for sequence in range(1, 101)]
        lines.append(entry(101, seconds=101, level="ERROR", site="reconciler:compute:134", message="reconcile failed", compute="cmp_a", exception=refused()))
        newest_first = list(reversed(lines))

        summary = summarize(newest_first, Query(), sequence=101, since=T0, until=T0 + timedelta(minutes=2), step=timedelta(seconds=60))

        failed, routine = summary.groups
        assert (failed.key, failed.level, failed.count, failed.computes) == ("reconciler:compute:134|ExceptionGroup", "ERROR", 1, 1)
        assert (routine.key, routine.count) == ("reconciler:_pass:197", 100)
        assert (routine.first, routine.last) == (T0 + timedelta(seconds=1), T0 + timedelta(seconds=100))
        assert routine.latest.sequence == 100
        assert summary.volume.info == (59, 41)
        assert summary.volume.error == (0, 1)
        assert routine.series == (59, 41)

    def it_counts_the_components_as_if_none_were_chosen() -> None:
        lines = [entry(3, component="ssh"), entry(2, component="machines"), entry(1, component="ssh")]

        summary = summarize(lines, Query(components=frozenset({"machines"})), sequence=3, since=T0, until=T0 + timedelta(seconds=1), step=timedelta(seconds=1))

        assert summary.components == {"ssh": 2, "machines": 1}
        assert [group.count for group in summary.groups] == [1]

    def it_leaves_what_came_after_its_sequence_to_a_follower() -> None:
        lines = [entry(3), entry(2), entry(1)]

        summary = summarize(lines, Query(), sequence=2, since=T0, until=T0 + timedelta(seconds=1), step=timedelta(seconds=1))

        assert summary.sequence == 2
        assert summary.groups[0].count == 2


def describe_capture() -> None:
    def it_writes_a_librarys_warnings_under_the_librarys_name(path: Path) -> None:
        logger.capture("litestar")
        try:
            with logfile(path):
                logging.getLogger("litestar.events.listener").warning("from the library")
                logging.getLogger("litestar.events.listener").info("below the level captured")
        finally:
            logger.release("litestar")

        (written_entry,) = entries(path)
        assert (written_entry.component, written_entry.logger, written_entry.message) == ("litestar", "litestar.events.listener", "from the library")

    def it_lets_a_released_library_go() -> None:
        library = logging.getLogger("somebody.else")
        logger.capture("somebody.else")
        logger.capture("somebody.else")
        logger.release("somebody.else")

        assert library.handlers == []


def describe_store() -> None:
    async def it_pages_newest_first_and_carries_on_from_the_cursor(path: Path) -> None:
        with logfile(path) as file:
            for index in range(5):
                logger.info("line {index}", index=index)
            logger.warning("the only warning")
        log = DaemonLogStore(file)

        first = await log.page(Query(), None, 2)
        second = await log.page(Query(), first.next_cursor, 2)
        warnings = await log.page(Query(level="WARNING"), None, 10)

        assert [each.sequence for each in first.items] == [6, 5]
        assert [each.sequence for each in second.items] == [4, 3]
        assert [each.message for each in warnings.items] == ["the only warning"]
        assert warnings.next_cursor is None

    async def it_follows_from_the_last_id_without_a_gap(path: Path) -> None:
        with logfile(path) as file:
            for index in range(3):
                logger.info("line {index}", index=index)
            await written(file, 3)
            log = DaemonLogStore(file)

            stream = log.follow(Query(), 1)
            try:
                assert [each.sequence for each in await run(stream)] == [2, 3]
                logger.info("line 3")
                assert [each.sequence for each in await run(stream)] == [4]
            finally:
                await stream.aclose()

    async def it_hands_a_follower_only_what_it_asked_for(path: Path) -> None:
        with logfile(path) as file:
            log = DaemonLogStore(file)
            stream = log.follow(Query(level="WARNING"), None)
            try:
                waiting = asyncio.ensure_future(anext(stream))
                await asyncio.sleep(0.05)
                logger.info("routine")
                logger.warning("worth a look")
                assert [each.message for each in await asyncio.wait_for(waiting, WAIT)] == ["worth a look"]
            finally:
                await stream.aclose()

    async def it_hangs_up_on_a_follower_that_fell_behind(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(store, "BACKLOG", 3)
        with logfile(path) as file:
            log = DaemonLogStore(file)
            stream = log.follow(Query(), None)
            waiting = asyncio.ensure_future(anext(stream))
            await asyncio.sleep(0.05)
            logger.info("taken at once")
            first = await asyncio.wait_for(waiting, WAIT)
            for index in range(10):
                logger.info("line {index}", index=index)
            await written(file, 11)

            behind = [each.sequence for each in await run(stream)]
            with pytest.raises(StopAsyncIteration):
                await run(stream)

        assert [each.sequence for each in first] == [1]
        assert behind == [2, 3, 4], "what was held before the hang-up is still handed over"


def describe_http() -> None:
    async def it_answers_404_where_the_daemon_keeps_no_log() -> None:
        async with AsyncTestClient(app=create_app(logging=False)) as http:
            response = await http.get("/v1/daemon/log")

        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    async def it_reads_a_page_and_a_summary(path: Path) -> None:
        with logfile(path) as file:
            for _ in range(3):
                logger.bind(component="reconciler").debug("pass")
            try:
                raise RuntimeError("cannot reach member")
            except RuntimeError:
                logger.bind(component="emitter", compute_id="cmp_a", node_id="nod_a").exception("event listener failed: on_node_connect")

        async with AsyncTestClient(app=create_app(with_real(log=DaemonLogStore(file)), logging=False)) as http:
            page = (await http.get("/v1/daemon/log", params={"level": "ERROR"})).json()
            summary = (await http.get("/v1/daemon/log/summary", params={"since": 0})).json()
            hidden = (await http.get("/v1/daemon/log", params={"hide": page["items"][0]["group"]})).json()

        (failed,) = page["items"]
        assert (failed["compute"], failed["node"], failed["exception"]["type"]) == ("cmp_a", "nod_a", "RuntimeError")
        assert failed["group"] == f"{failed['site']}|RuntimeError"
        assert [(group["level"], group["count"]) for group in summary["groups"]] == [("ERROR", 1), ("DEBUG", 3)]
        assert summary["components"] == {"reconciler": 3, "emitter": 1}
        assert summary["sequence"] == 4
        assert [item["message"] for item in hidden["items"]] == ["pass", "pass", "pass"]

    async def it_refuses_a_window_cut_into_too_many_steps(path: Path) -> None:
        with logfile(path) as file:
            pass

        async with AsyncTestClient(app=create_app(with_real(log=DaemonLogStore(file)), logging=False)) as http:
            response = await http.get("/v1/daemon/log/summary", params={"since": 0, "until": 10_000, "step": 1})

        assert response.status_code == 400
