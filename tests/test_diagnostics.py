"""Why a machine was replaced, asked after it already was.

A node that goes away is replaced by one that works, and the replacement is the
only thing left to look at. These are the two places the reason has to survive to
still be there when somebody asks: the node's own row, and the event log the
console and ``sky log export`` read.
"""

import json
from pathlib import Path

import pytest

from skyward.core.console import render
from skyward.server.application.machines import Machines
from skyward.server.application.mock import SPEC
from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.tables import EventRow
from skyward.shared.schemas import ComputeCreate, Error, Node, NodeEvent, ProgressEvent

pytestmark = pytest.mark.local


async def given(database: Path) -> tuple[NodeStore, EventStore, Node]:
    """One compute with one node asked for, in a database of this test's own."""
    await connect(database)
    computes, nodes, events = ComputeStore(), NodeStore(), EventStore()
    compute, _ = await computes.create(ComputeCreate(spec=SPEC), idempotency_key="given")
    return nodes, events, await nodes.request(compute.id, compute.generation)


def machines_for(nodes: NodeStore, events: EventStore) -> Machines:
    providers = ProviderStore()
    return Machines(
        computes=ComputeStore(),
        nodes=nodes,
        providers=providers,
        offers=OfferCache(providers),
        blobs=BlobStore(),
        events=events,
    )


def describe_a_node_given_up_on() -> None:
    async def it_still_says_why_after_the_row_is_retired(tmp_path: Path) -> None:
        nodes, _, node = await given(tmp_path / "skyward.sqlite")

        await nodes.observe(node.id, "lost", Error(code="not_found", message="the machine went away", retryable=True))
        await nodes.observe(node.id, "deleting")
        await nodes.observe(node.id, "deleted")

        retired = await nodes.get(node.compute_id, node.id)
        assert retired.last_error is not None, "the bookkeeping that followed knew nothing about why"
        assert retired.last_error.message == "the machine went away"

    async def it_is_forgotten_once_a_machine_reports_itself_up(tmp_path: Path) -> None:
        nodes, _, node = await given(tmp_path / "skyward.sqlite")

        await nodes.observe(node.id, "connecting", Error(code="unreachable", message="ssh refused", retryable=True))
        await nodes.observe(node.id, "ready")

        assert (await nodes.get(node.compute_id, node.id)).last_error is None, "an error a working machine outlived is noise"


def describe_a_machine_the_provider_no_longer_has() -> None:
    async def it_is_announced_with_the_reason_it_was_given_up_on(tmp_path: Path) -> None:
        nodes, events, node = await given(tmp_path / "skyward.sqlite")
        machines = machines_for(nodes, events)

        await machines._lost(node, "the machine went away")

        recorded = await EventRow.objects().where(EventRow.compute_id == node.compute_id)
        lost = [row for row in recorded if row.type == "node.lost"]

        assert len(lost) == 1, "a loss nobody announced is a machine that was silently replaced"
        assert json.loads(lost[0].payload)["error"] == "the machine went away"
        assert (await nodes.get(node.compute_id, node.id)).state == "lost"


def describe_a_machine_still_on_its_way_up() -> None:
    """Ten minutes of `provisioning` and no line is a pool nobody can tell from a hung one."""

    async def it_says_what_it_is_doing_without_leaving_a_row(tmp_path: Path) -> None:
        nodes, events, node = await given(tmp_path / "skyward.sqlite")
        machines = machines_for(nodes, events)
        before = len(await EventRow.objects())

        await machines._progressed(node, "downloading (42%)")

        assert len(await EventRow.objects()) == before, "a percentage that moves every couple of seconds is a gauge"

    def it_reads_as_a_line_of_the_node_own() -> None:
        line = render(ProgressEvent(compute="cmp_1", node="nod_1", progress="downloading (42%)"))

        assert line is not None and "downloading (42%)" in line


def describe_the_line_a_lost_machine_leaves() -> None:
    def it_carries_the_reason_and_not_only_the_badge() -> None:
        lost = NodeEvent(compute="cmp_1", node="nod_1", state="lost", error="the machine has been downloading (55%) for 600s")

        line = render(lost)

        assert line is not None
        assert "downloading (55%)" in line, "a badge saying only `lost` sends the reader to the database"
