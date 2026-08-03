"""The machine the function lands on — built from an image the user described.

One pool for the whole file, because the environment *is* what is under test: an
image is decided before a machine boots, and there is no way to change one
afterwards. Everything the image and a plugin promise is asserted on that one
machine.
"""

import sys
from pathlib import Path

import cloudpickle
import pytest

import skyward as sky
from tests.conftest import Build

pytestmark = [pytest.mark.compute, pytest.mark.xdist_group("environment")]

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@sky.function
def report() -> tuple[str, bool, str, str, bool]:
    import os
    from importlib.util import find_spec

    import shipped

    return (
        os.environ["GREETING"],
        find_spec("toml") is not None,
        shipped.WHO,
        os.environ["HF_TOKEN"],
        find_spec("huggingface_hub") is not None,
    )


@pytest.fixture
def local_module(tmp_path: Path) -> Path:
    module = tmp_path / "code"
    module.mkdir()
    (module / "shipped.py").write_text('WHO = "the local checkout"\n')
    return module


def describe_the_environment_a_pool_boots() -> None:
    def it_carries_the_image_and_rebuilds_the_plugin_from_its_fields(compute: Build, local_module: Path) -> None:
        """The env, the package and the local module come from the image; the token
        and huggingface_hub from a plugin rebuilt on the node from its fields."""
        image = sky.Image(
            python="3.13",
            skyward="local",
            pip=("toml",),
            env={"GREETING": "hello"},
            includes=(str(local_module / "shipped.py"),),
        )

        with compute(image=image, plugins=[sky.plugins.HuggingFace(token="hf_not_a_real_token")]) as pool:
            greeting, installed, shipped, token, hub = report() >> pool

        assert greeting == "hello", "the exported variable reached the worker"
        assert installed, "the package was installed on the machine"
        assert shipped == "the local checkout", "and the local module was importable where the function ran"
        assert (token, hub) == ("hf_not_a_real_token", True), "the plugin was rebuilt on the node from what travelled"
