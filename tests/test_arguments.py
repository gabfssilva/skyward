"""How a pool is asked for: the constructor refuses what it would otherwise drop."""

import pytest

import skyward as sky

pytestmark = pytest.mark.local


def describe_describing_the_machine() -> None:
    def a_shape_without_a_provider_is_refused() -> None:
        with pytest.raises(ValueError, match="pass them with provider="):
            sky.Compute(sky.Spec(provider=sky.Container()), accelerator="a100")

    def a_shape_with_its_provider_is_a_spec() -> None:
        pool = sky.Compute(provider=sky.Container(), cpus=2)

        assert pool._spec.specs[0].cpus == 2
