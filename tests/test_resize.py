"""Changing the size of a compute that already exists.

A resize is the one change that touches a definition without replacing what it
was built into, and the CLI and the SDK both make it through the same ``PATCH``.
What is asserted here is the rule the daemon holds — a new generation, and a
refusal for the computes whose ranks are no longer free to move. Whether the
machines actually arrive is a question for the lifecycle tests, which own a pool.
"""

from pathlib import Path

import msgspec
import pytest

from skyward.server.application.mock import SPEC
from skyward.server.persistence.computes import GenerationStore
from skyward.shared.errors import ComputeNotResizableError
from skyward.shared.schemas import ComputeSpecPatch, NodeBounds
from tests.conftest import cli, given

pytestmark = pytest.mark.local


def describe_asking_a_compute_for_a_different_size() -> None:
    async def the_size_it_already_has_is_not_a_new_definition(tmp_path: Path) -> None:
        store, compute = await given(tmp_path / "skyward.sqlite")

        written = await store.patch(compute.id, ComputeSpecPatch(nodes=SPEC.nodes), compute.revision)

        assert (written.revision, written.generation) == (compute.revision, compute.generation)
        assert len((await GenerationStore(store).list(compute.id)).items) == 1

    async def it_is_a_new_definition_and_leaves_the_rest_of_the_spec_alone(tmp_path: Path) -> None:
        store, compute = await given(tmp_path / "skyward.sqlite")

        resized = await store.patch(compute.id, ComputeSpecPatch(nodes=NodeBounds(initial=16)), compute.revision)

        assert resized.spec.nodes == NodeBounds(initial=16)
        assert resized.generation == compute.generation + 1, "a size is a definition, so it is a generation"
        assert msgspec.structs.replace(resized.spec, nodes=SPEC.nodes) == SPEC, "and nothing else moved with it"

    async def it_is_recorded_as_a_generation_of_its_own(tmp_path: Path) -> None:
        store, compute = await given(tmp_path / "skyward.sqlite")

        resized = await store.patch(compute.id, ComputeSpecPatch(nodes=NodeBounds(initial=2)), compute.revision)
        frozen = await GenerationStore(store).get(compute.id, resized.generation)

        assert frozen.spec.nodes == NodeBounds(initial=2)


def describe_a_compute_running_a_collective() -> None:
    async def it_is_refused_a_resize_by_the_plugin_that_froze_it(tmp_path: Path) -> None:
        store, compute = await given(tmp_path / "skyward.sqlite", "torch")

        with pytest.raises(ComputeNotResizableError) as refused:
            await store.patch(compute.id, ComputeSpecPatch(nodes=NodeBounds(initial=16)), compute.revision)

        assert refused.value.details["plugin"] == "torch"
        assert (await store.get(compute.id)).spec.nodes == SPEC.nodes, "and the definition it had is still the one it has"

    async def it_still_answers_a_request_that_asks_for_the_size_it_has(tmp_path: Path) -> None:
        store, compute = await given(tmp_path / "skyward.sqlite", "torch")

        written = await store.patch(compute.id, ComputeSpecPatch(nodes=SPEC.nodes), compute.revision)

        assert written.spec.nodes == SPEC.nodes, "writing the same size back is a retry, not a resize"


def describe_spelling_a_size_on_the_command_line() -> None:
    @pytest.mark.parametrize("nodes", ["8:2", "0", "banana", "2:", "2:8:16"])
    def it_is_refused_before_a_daemon_is_opened_at_all(nodes: str) -> None:
        ran = cli("compute", "scale", "absent", "--nodes", nodes)

        assert ran.code != 0
        assert "--nodes takes" in ran.err
        assert "Traceback" not in ran.err, "a refusal is an answer, not a crash"

    def it_reaches_the_daemon_once_the_size_makes_sense(alone: str) -> None:
        ran = cli("compute", "scale", "absent", "--nodes", "2:8", "--url", alone)

        assert ran.code != 0
        assert "not_found" in ran.err, "the size parsed, and the compute is what was missing"
