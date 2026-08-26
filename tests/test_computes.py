"""What the daemon refuses when a compute is asked for, and how it says so.

Both of these were a `500` before: an integrity error and a `RuntimeError` on
their way out through the exception handler. A caller can act on a refusal that
names itself — pick another name, ask the daemon that is holding the compute —
and can do nothing at all with an internal error.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest

from skyward.server.application.machines import Machines
from skyward.server.application.mock import SPEC
from skyward.server.application.runtimes import Files, Runtimes
from skyward.server.persistence.computes import ComputeStore, Infrastructure
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.shared.errors import ComputeNotConnectedError, NameTakenError
from skyward.shared.provider import Machine
from skyward.shared.schemas import Compute, ComputeCreate, Node

pytestmark = pytest.mark.local


def describe_naming_a_compute() -> None:
    async def it_is_refused_when_another_compute_already_has_the_name(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        store = ComputeStore()
        first, _ = await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="first")

        with pytest.raises(NameTakenError) as refused:
            await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="second")

        assert refused.value.details["compute"] == first.id
        assert refused.value.status == 409

    async def it_lets_two_computes_go_unnamed(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        store = ComputeStore()

        first, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="first")
        second, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="second")

        assert first.id != second.id, "a name nobody gave is not a name two computes share"


def describe_reaching_a_compute_this_daemon_is_not_holding() -> None:
    async def it_is_told_rather_than_raised_through(tmp_path: Path) -> None:
        files = Files(Runtimes(listener=lambda *_: None, output=lambda *_: None, sample=lambda *_: None, phase=lambda *_: None))

        with pytest.raises(ComputeNotConnectedError) as refused:
            await files.run("cmp_elsewhere", "all", "echo hello")

        assert refused.value.status == 409
        assert refused.value.retryable, "another daemon holds it, or this one has not picked it up yet"


def describe_a_machine_that_is_bought_and_never_says_where_it_is() -> None:
    async def it_is_given_up_on_once_the_window_closes(tmp_path: Path) -> None:
        machines, compute, node = await _bought(tmp_path / "skyward.sqlite", provision_timeout=0.01)
        await asyncio.sleep(0.05)

        await machines.resolve(compute, await NodeStore().of(compute.id))

        given_up = await NodeStore().get(compute.id, node.id)
        assert given_up.state == "lost"
        assert given_up.last_error is not None
        assert "never published an address" in given_up.last_error.message

    async def it_is_waited_on_while_the_window_is_open(tmp_path: Path) -> None:
        machines, compute, node = await _bought(tmp_path / "skyward.sqlite", provision_timeout=600.0)

        await machines.resolve(compute, await NodeStore().of(compute.id))

        assert (await NodeStore().get(compute.id, node.id)).state == "provisioning", "a machine still coming up is not a machine lost"


async def _bought(database: Path, provision_timeout: float) -> tuple[Machines, Compute, Node]:
    """A compute whose one node has a machine the provider reports without an address."""
    await connect(database)
    computes, nodes, providers = ComputeStore(), NodeStore(), ProviderStore()
    spec = msgspec.structs.replace(SPEC, options=msgspec.structs.replace(SPEC.options, provision_timeout=provision_timeout))
    compute, _ = await computes.create(ComputeCreate(spec=spec), idempotency_key="bought")
    await computes.bind(compute.id, Infrastructure(provider_id="prv_1", binding={"prefix": "skyward-"}))

    node = await nodes.request(compute.id, compute.generation)
    await nodes.launched(node.id, Machine(id="m1", state="running"))

    machines = Machines(
        computes=computes,
        nodes=nodes,
        providers=providers,
        offers=OfferCache(providers),
        blobs=BlobStore(),
        events=EventStore(),
    )
    machines.adapter = lambda _: _answering({"m1": Machine(id="m1", state="running")})
    return machines, await computes.get(compute.id), node


async def _answering(observed: dict[str, Machine]) -> Any:
    """A provider that reports exactly these machines and is asked nothing else."""
    return SimpleNamespace(machines=lambda _: _listing(observed))


async def _listing(observed: dict[str, Machine]) -> dict[str, Machine]:
    return observed
