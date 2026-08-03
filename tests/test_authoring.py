"""What the SDK refuses before a machine exists.

Every one of these is a mistake a user makes while writing, and the feature is
that it is answered where it was written rather than twenty minutes later on a
machine that is already costing money.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

import skyward as sky

pytestmark = pytest.mark.local


def describe_declaring_a_function() -> None:
    def describe_when_it_is_a_generator() -> None:
        def it_is_sent_back_to_be_written_as_a_stream() -> None:
            with pytest.raises(TypeError, match="@stream"):

                @sky.function
                def wrong() -> Iterator[int]:
                    yield 1

    def it_stays_lazy_until_it_is_dispatched() -> None:
        calls: list[int] = []

        @sky.function
        def counted(value: int) -> int:
            calls.append(value)
            return value

        pending = counted(1)

        assert isinstance(pending, sky.Pending)
        assert calls == [], "building the call ran nothing"

    def it_takes_a_timeout_without_becoming_a_different_call() -> None:
        @sky.function
        def anything() -> int:
            return 1

        call = anything()

        assert call.with_timeout(5) is not call, "the deadline is a new value, not a mutation"


def describe_asking_for_a_pool() -> None:
    def describe_when_it_names_both_a_provider_and_specs() -> None:
        def it_refuses_rather_than_picking_one(tmp_path: Path) -> None:
            with pytest.raises(ValueError):
                sky.Compute(
                    sky.Spec(provider=sky.Container()),
                    provider=sky.Container(),
                    database=tmp_path / "skyward.sqlite",
                )

    def describe_when_it_names_an_accelerator_nobody_sells() -> None:
        def it_is_refused_where_the_name_was_written() -> None:
            with pytest.raises(AttributeError, match="unknown accelerator"):
                sky.accelerators.H999()

    def describe_when_it_attaches_to_a_compute_that_is_not_there() -> None:
        def it_says_so_instead_of_provisioning_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
            monkeypatch.delenv("SKYWARD_URL", raising=False)

            with pytest.raises(sky.SkywardError), sky.Compute.attached("never-existed", database=tmp_path / "skyward.sqlite"):
                pass


def describe_reading_the_topology_off_a_node() -> None:
    def it_says_where_it_is_not_rather_than_guessing() -> None:
        with pytest.raises(RuntimeError, match="node"):
            sky.instance_info()
