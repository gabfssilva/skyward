"""A mutable image changes on the machines a compute already has.

A pool built with ``Image(mutable=True)`` is asked for other packages and other
local code after it is up, and gets them without a machine being replaced: every
ready node is sent through ``bootstrapping`` and comes back ``ready`` with the new
venv. What is asserted here is the contract end to end — the package is
importable afterwards, a task already running when the change arrives finishes,
the code shipped is the code as it is now, and a process naming the compute
adopts it rather than failing on the name. The daemon-side rules are the store
and reconciler tests' business.
"""

import sys
import time
import uuid
from pathlib import Path
from typing import Any

import cloudpickle
import httpx
import pytest

import skyward as sky
from skyward.core.errors import SkywardError
from tests.conftest import PYTHON, Build, cli

pytestmark = [pytest.mark.compute, pytest.mark.xdist_group("refresh")]

cloudpickle.register_pickle_by_value(sys.modules[__name__])

SETTLE = 300.0
"""Seconds a refresh may take: the worker leaves, pip runs, the worker comes back and joins."""


@sky.function
def installed(name: str) -> bool:
    from importlib.util import find_spec

    return find_spec(name) is not None


@sky.function
def who() -> str:
    import shipped

    return shipped.WHO


@sky.function
def slow(seconds: float) -> str:
    import time

    time.sleep(seconds)
    return "finished"


def mutable(**fields: Any) -> sky.Image:
    return sky.Image(python=PYTHON, skyward="local", mutable=True, **fields)


def compute_of(daemon: str, ref: str) -> dict[str, Any]:
    answer = httpx.get(f"{daemon}/v1/computes/{ref}", timeout=10)
    answer.raise_for_status()
    return answer.json()


def settled(daemon: str, ref: str, generation: int) -> dict[str, Any]:
    """The compute once the machines reflect ``generation`` and every node is ready again."""
    deadline = time.monotonic() + SETTLE
    while time.monotonic() < deadline:
        found = compute_of(daemon, ref)
        caught_up = found["status"]["observed_generation"] >= generation
        if caught_up and found["nodes"] and all(node["state"] == "ready" for node in found["nodes"]):
            return found
        time.sleep(1.0)
    raise AssertionError(f"compute {ref} did not settle on generation {generation} within {SETTLE:.0f}s: {compute_of(daemon, ref)['nodes']}")


@pytest.fixture
def local_module(tmp_path: Path) -> Path:
    module = tmp_path / "code"
    module.mkdir()
    (module / "shipped.py").write_text('WHO = "v1"\n')
    return module


def describe_a_mutable_pool_asked_for_other_packages() -> None:
    def it_installs_them_and_comes_back_ready(compute: Build, daemon: str) -> None:
        with compute(image=mutable()) as pool:
            before = compute_of(daemon, pool.id)
            assert installed("toml") >> pool is False, "the package is not there to begin with"

            pool.update(mutable(pip=("toml",)))

            after = settled(daemon, pool.id, before["generation"] + 1)
            assert installed("toml") >> pool is True, "and it is, once the node is back"
            assert [node["image"] for node in after["nodes"]] != [node["image"] for node in before["nodes"]], "the node says it materialized another image"
            assert after["spec"]["image"]["pip"] == ["toml"]

    def it_lets_a_task_already_running_finish_first(compute: Build, daemon: str) -> None:
        with compute(image=mutable()) as pool:
            before = compute_of(daemon, pool.id)
            running = slow(8.0) > pool
            time.sleep(1.0)

            pool.update(mutable(pip=("toml",)))

            assert running.result(timeout=SETTLE) == "finished", "the refresh waited for the node to hold nothing"
            settled(daemon, pool.id, before["generation"] + 1)
            assert installed("toml") >> pool is True


def describe_a_mutable_pool_whose_shipped_code_changed() -> None:
    def it_runs_the_code_as_it_is_now(compute: Build, daemon: str, local_module: Path) -> None:
        image = mutable(includes=(str(local_module / "shipped.py"),))
        with compute(image=image) as pool:
            before = compute_of(daemon, pool.id)
            assert who() >> pool == "v1"

            (local_module / "shipped.py").write_text('WHO = "v2"\n')
            ran = cli("compute", "update", pool.id, "--include", str(local_module / "shipped.py"), "--url", daemon)

            assert ran.code == 0, ran.err
            settled(daemon, pool.id, before["generation"] + 1)
            assert who() >> pool == "v2", "the worker imports the copy shipped now, not the one it was built with"


def describe_naming_a_compute_that_already_exists() -> None:
    def a_mutable_one_is_adopted_and_brought_to_the_new_definition(compute: Build, daemon: str) -> None:
        """The first process leaves the compute up; the second names it with another image, as a script run again would."""
        name = f"refresh-{uuid.uuid4().hex[:8]}"
        with compute(image=mutable(), name=name, delete_on_exit=False) as first:
            left_up = first.id
            before = compute_of(daemon, left_up)

        with sky.Compute(provider=sky.Container(), cpus=1, memory_gb=1, image=mutable(pip=("toml",)), name=name, url=daemon) as second:
            assert second.id == left_up, "the same compute, not a second one"
            settled(daemon, left_up, before["generation"] + 1)
            assert installed("toml") >> second is True

    def a_fixed_one_still_refuses_the_name(compute: Build, daemon: str) -> None:
        name = f"fixed-{uuid.uuid4().hex[:8]}"
        fixed = sky.Image(python=PYTHON, skyward="local")
        with (
            compute(image=fixed, name=name),
            pytest.raises(SkywardError) as refused,
            sky.Compute(provider=sky.Container(), cpus=1, memory_gb=1, image=fixed, name=name, url=daemon),
        ):
            pass

        assert refused.value.code == "name_taken"


def describe_a_fixed_pool_asked_for_other_packages() -> None:
    def it_is_refused_with_image_fixed(compute: Build) -> None:
        with compute(image=sky.Image(python=PYTHON, skyward="local")) as pool:
            with pytest.raises(SkywardError) as refused:
                pool.update(sky.Image(python=PYTHON, skyward="local", pip=("toml",)))

            assert refused.value.code == "image_fixed"
