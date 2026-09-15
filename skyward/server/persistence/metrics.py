"""Node metrics: recent samples as rows, closed windows as compressed chunks.

A node reports a handful of gauges every couple of seconds, which as rows is more
than ten megabytes a day per node. What is recent stays rows — a window is still filling,
late samples are still landing, and readers following along want it by the row. A
window that closed longer than the grace ago is folded into one chunk per node:
each metric a column of its moments and values, stored as changes from the sample
before, packed with msgpack and zlib — about a byte or two a sample, where a row
costs dozens. Every read joins the two, so nothing about compaction shows through it.
"""

from __future__ import annotations

import asyncio
import itertools
import time
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from decimal import Decimal
from statistics import fmean

import msgspec
from msgspec import Struct

from skyward.server.persistence.db import transaction
from skyward.server.persistence.tables import MetricChunkRow, MetricSampleRow
from skyward.shared.schemas import Aggregate, MetricHistory, MetricSample, MetricSeries

WINDOW = 30 * 60 * 1000
"""Milliseconds of samples one chunk holds."""

GRACE = 5 * 60 * 1000
"""How long a closed window stays rows: time for a late sample to land, and for a reader following along to read it."""

BATCH = 1000
"""Rows one INSERT carries. A node's backlog read after a restart arrives all at once, and SQLite caps the parameters of a statement."""

type Points = dict[tuple[str, str], dict[int, float]]
"""Samples by node and name, then by moment."""


class MetricStore:
    """Where node metrics are written, compacted and read back.

    Samples are written in batches: :meth:`add` holds them and :meth:`flush` writes
    what it holds. A sample recorded twice — a node's log read again after the link
    dropped or the daemon restarted — is kept once.

    A row's id is the order it was recorded in, and it is what a cursor counts. Ids
    are handed out here rather than by SQLite, which gives a new row one more than
    the largest id left in the table: compaction deletes the newest rows too once a
    node goes quiet, and ids counted again from below would pass under every cursor
    already handed out. The largest id ever used is the larger of what the rows and
    the chunks remember.
    """

    def __init__(self, window: int = WINDOW, grace: int = GRACE) -> None:
        self._window = window
        self._grace = grace
        self._pending: list[tuple[str, MetricSample]] = []
        self._last: int | None = None
        self._writing = asyncio.Lock()

    def add(self, compute: str, samples: Iterable[MetricSample]) -> None:
        """Hold samples for the next :meth:`flush`."""
        self._pending.extend((compute, sample) for sample in samples)

    async def flush(self) -> None:
        """Write every sample held, in as few statements as the batch size allows."""
        async with self._writing:
            pending, self._pending = self._pending, []
            for batch in itertools.batched(pending, BATCH):
                first = await self._head() + 1
                await MetricSampleRow.insert(
                    *(
                        MetricSampleRow(id=first + offset, compute_id=compute, node_id=sample.node, name=sample.name, at=sample.at, value=sample.value)
                        for offset, (compute, sample) in enumerate(batch)
                    )
                ).on_conflict(action="DO NOTHING").run()
                self._last = first + len(batch) - 1

    async def compact(self, now: int | None = None) -> None:
        """Fold every window that closed more than the grace before ``now`` into its node's chunk.

        A window is sealed in one transaction: its rows are read, merged into the chunk
        already there if a late sample reopened it, written back and deleted. ``now`` is
        milliseconds since the epoch, the wall clock when not given.
        """
        moment = time.time_ns() // 1_000_000 if now is None else now
        boundary = (moment - self._grace) // self._window * self._window
        windows = await MetricSampleRow.raw(
            "SELECT DISTINCT compute_id, node_id, at / {} * {} AS since FROM metric_samples WHERE at < {}",
            self._window,
            self._window,
            boundary,
        ).run()
        for window in windows:
            await self._seal(window["compute_id"], window["node_id"], window["since"])

    async def series(
        self,
        compute: str,
        since: int,
        until: int | None = None,
        step: int | None = None,
        aggregate: Aggregate = "avg",
        nodes: Sequence[str] | None = None,
        names: Sequence[str] | None = None,
    ) -> MetricHistory:
        """A compute's samples measured from ``since`` up to ``until``, or one value per ``step``.

        With no ``until`` the range is open, so a sample from a node whose clock runs
        ahead of the daemon's is not left out of it. The rows are read before the
        chunks: a window sealed between the two reads is then read twice, and merges
        into itself, rather than read never.
        """
        head = await self._head()
        query = MetricSampleRow.select(MetricSampleRow.node_id, MetricSampleRow.name, MetricSampleRow.at, MetricSampleRow.value).where(
            (MetricSampleRow.compute_id == compute) & (MetricSampleRow.at >= since) & (MetricSampleRow.id <= head)
        )
        if until is not None:
            query = query.where(MetricSampleRow.at < until)
        if nodes:
            query = query.where(MetricSampleRow.node_id.is_in(list(nodes)))
        if names:
            query = query.where(MetricSampleRow.name.is_in(list(names)))
        rows = await query

        sealed = MetricChunkRow.select(MetricChunkRow.node_id, MetricChunkRow.data).where(
            (MetricChunkRow.compute_id == compute) & (MetricChunkRow.until > since)
        )
        if until is not None:
            sealed = sealed.where(MetricChunkRow.since < until)
        if nodes:
            sealed = sealed.where(MetricChunkRow.node_id.is_in(list(nodes)))

        points: Points = {}
        for chunk in await sealed:
            for name, samples in _unpack(chunk["data"]).items():
                if names and name not in names:
                    continue
                inside = ((at, value) for at, value in samples.items() if at >= since and (until is None or at < until))
                points.setdefault((chunk["node_id"], name), {}).update(inside)
        for row in rows:
            points.setdefault((row["node_id"], row["name"]), {})[row["at"]] = row["value"]

        return MetricHistory(series=_series(points, step, aggregate), cursor=str(head))

    async def after(self, compute: str, cursor: str, nodes: Sequence[str] | None = None, names: Sequence[str] | None = None) -> MetricHistory:
        """What was recorded for a compute after ``cursor``, whenever it was measured.

        The compacted chunks are checked after the rows are read, so a window sealed in
        between is caught by ``reset`` rather than missed by both.
        """
        head = await self._head()
        start = int(cursor)
        query = MetricSampleRow.select(MetricSampleRow.node_id, MetricSampleRow.name, MetricSampleRow.at, MetricSampleRow.value).where(
            (MetricSampleRow.compute_id == compute) & (MetricSampleRow.id > start) & (MetricSampleRow.id <= head)
        )
        if nodes:
            query = query.where(MetricSampleRow.node_id.is_in(list(nodes)))
        if names:
            query = query.where(MetricSampleRow.name.is_in(list(names)))

        points: Points = {}
        for row in await query:
            points.setdefault((row["node_id"], row["name"]), {})[row["at"]] = row["value"]

        reset = await MetricChunkRow.exists().where((MetricChunkRow.last_id > start) & (MetricChunkRow.compute_id == compute))
        return MetricHistory(series=_series(points, None, "avg"), cursor=str(head), reset=reset)

    async def latest(self, compute: str, nodes: Sequence[str] | None = None, names: Sequence[str] | None = None) -> tuple[MetricSample, ...]:
        """The newest sample of each node and name.

        A node that has been quiet for longer than a window and the grace has no rows
        left, and is answered from the newest chunk it has.
        """
        filters, arguments = _filters(nodes, names)
        rows = await MetricSampleRow.raw(
            f"SELECT node_id, name, max(at) AS at, value FROM metric_samples WHERE compute_id = {{}}{filters} GROUP BY node_id, name",
            compute,
            *arguments,
        ).run()
        found = {(row["node_id"], row["name"]): MetricSample(node=row["node_id"], name=row["name"], at=row["at"], value=row["value"]) for row in rows}

        nodal, nodal_arguments = _filters(nodes, None)
        newest = await MetricChunkRow.raw(
            "SELECT node_id, data FROM metric_chunks AS chunk WHERE compute_id = {}"
            f"{nodal} AND since = (SELECT max(since) FROM metric_chunks WHERE compute_id = chunk.compute_id AND node_id = chunk.node_id)",
            compute,
            *nodal_arguments,
        ).run()
        speaking = {node for node, _ in found}
        for chunk in newest:
            if chunk["node_id"] in speaking:
                continue
            for name, samples in _unpack(chunk["data"]).items():
                if samples and (not names or name in names):
                    at = max(samples)
                    found[(chunk["node_id"], name)] = MetricSample(node=chunk["node_id"], name=name, at=at, value=samples[at])

        return tuple(sample for _, sample in sorted(found.items()))

    async def _seal(self, compute: str, node: str, since: int) -> None:
        until = since + self._window
        window = (
            (MetricSampleRow.compute_id == compute) & (MetricSampleRow.node_id == node) & (MetricSampleRow.at >= since) & (MetricSampleRow.at < until)
        )
        async with transaction():
            rows = await MetricSampleRow.select(MetricSampleRow.id, MetricSampleRow.name, MetricSampleRow.at, MetricSampleRow.value).where(window)
            if not rows:
                return
            existing = await MetricChunkRow.select(MetricChunkRow.data).where(
                (MetricChunkRow.compute_id == compute) & (MetricChunkRow.node_id == node) & (MetricChunkRow.since == since)
            ).first()
            samples = _unpack(existing["data"]) if existing else {}
            for row in rows:
                samples.setdefault(row["name"], {})[row["at"]] = row["value"]
            await MetricChunkRow.raw(
                "INSERT INTO metric_chunks (compute_id, node_id, since, until, last_id, data) VALUES ({}, {}, {}, {}, {}, {}) "
                "ON CONFLICT (compute_id, node_id, since) DO UPDATE SET data = excluded.data, last_id = max(last_id, excluded.last_id)",
                compute,
                node,
                since,
                until,
                max(row["id"] for row in rows),
                _pack(samples),
            ).run()
            await MetricSampleRow.delete().where(window).run()

    async def _head(self) -> int:
        """The largest id recorded so far, the rows gone into chunks included."""
        if self._last is None:
            [row] = await MetricSampleRow.raw(
                "SELECT max(coalesce((SELECT max(id) FROM metric_samples), 0), coalesce((SELECT max(last_id) FROM metric_chunks), 0)) AS head"
            ).run()
            self._last = int(row["head"])
        return self._last


class _Column(Struct, frozen=True, array_like=True):
    """One metric of one chunk.

    ``at`` is the first moment, the first gap, and then how much each gap differs
    from the one before: a sampler on a period barely changes its gaps, and numbers
    that barely change compress to almost nothing. ``values`` are the samples times
    ``10 ** scale``, each stored as its change from the one before; a series those
    integers cannot carry exactly keeps its ``floats`` as they came, with ``scale``
    at ``-1``.
    """

    name: str
    at: tuple[int, ...]
    scale: int
    values: tuple[int, ...] = ()
    floats: tuple[float, ...] = ()


_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(tuple[_Column, ...])


def _pack(samples: Mapping[str, Mapping[int, float]]) -> bytes:
    return zlib.compress(_encoder.encode(tuple(_column(name, sorted(series.items())) for name, series in sorted(samples.items()))))


def _unpack(data: bytes) -> dict[str, dict[int, float]]:
    return {column.name: dict(zip(_moments(column.at), _numbers(column), strict=True)) for column in _decoder.decode(zlib.decompress(data))}


def _column(name: str, samples: Sequence[tuple[int, float]]) -> _Column:
    moments = [at for at, _ in samples]
    gaps = [later - earlier for earlier, later in itertools.pairwise(moments)]
    at = (moments[0], *gaps[:1], *(later - earlier for earlier, later in itertools.pairwise(gaps)))
    values = [value for _, value in samples]
    match _scaled(values):
        case (scale, integers):
            return _Column(name=name, at=at, scale=scale, values=(integers[0], *(later - earlier for earlier, later in itertools.pairwise(integers))))
        case None:
            return _Column(name=name, at=at, scale=-1, floats=tuple(values))


def _scaled(values: Sequence[float]) -> tuple[int, list[int]] | None:
    """The fewest decimal places that carry every value exactly, and the values as integers at that scale."""
    places = 0
    for value in values:
        match Decimal(repr(value)).as_tuple().exponent:
            case int(exponent):
                places = max(places, -exponent)
            case _:
                return None
    if any(abs(value) >= 2**62 / 10**places for value in values):
        return None
    integers = [round(value * 10**places) for value in values]
    if all(integer / 10**places == value for integer, value in zip(integers, values, strict=True)):
        return places, integers
    return None


def _moments(at: Sequence[int]) -> Iterator[int]:
    first, *changes = at
    return itertools.accumulate(itertools.accumulate(changes), initial=first)


def _numbers(column: _Column) -> Iterator[float]:
    if column.scale < 0:
        return iter(column.floats)
    return (integer / 10**column.scale for integer in itertools.accumulate(column.values))


def _series(points: Points, step: int | None, aggregate: Aggregate) -> tuple[MetricSeries, ...]:
    def values(samples: dict[int, float]) -> list[tuple[int, float]]:
        ordered = sorted(samples.items())
        if step is None:
            return ordered
        width = step
        return [
            (start, _aggregated([value for _, value in bucket], aggregate))
            for start, bucket in itertools.groupby(ordered, key=lambda sample: sample[0] // width * width)
        ]

    return tuple(
        MetricSeries(node=node, name=name, at=tuple(at for at, _ in ordered), values=tuple(value for _, value in ordered))
        for (node, name), samples in sorted(points.items())
        if (ordered := values(samples))
    )


def _aggregated(values: list[float], aggregate: Aggregate) -> float:
    match aggregate:
        case "avg":
            return fmean(values)
        case "min":
            return min(values)
        case "max":
            return max(values)
        case "last":
            return values[-1]


def _filters(nodes: Sequence[str] | None, names: Sequence[str] | None) -> tuple[str, tuple[str, ...]]:
    """The ``AND … IN (…)`` clauses a raw query narrows by, and the arguments they bind."""
    clauses = [(column, tuple(wanted)) for column, wanted in (("node_id", nodes), ("name", names)) if wanted]
    sql = "".join(f" AND {column} IN ({', '.join('{}' for _ in wanted)})" for column, wanted in clauses)
    return sql, tuple(value for _, wanted in clauses for value in wanted)
