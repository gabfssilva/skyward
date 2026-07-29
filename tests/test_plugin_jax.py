"""The jax plugin as a value: what it installs, and how it travels."""

from skyward.worker.plugins.jax import Jax
from skyward.shared.schemas import Image, PipIndex, PluginRef


def test_it_installs_the_cuda_wheel_from_the_jax_index():
    built = Jax().image(Image(pip=("numpy",)))

    assert built.pip == ("numpy", "jax[cu124]")
    assert built.pip_indexes == (
        PipIndex(
            url="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
            packages=("jax", "jaxlib"),
        ),
    )


def test_the_cuda_build_is_a_knob():
    built = Jax(cuda="cu12").image(Image())

    assert built.pip == ("jax[cu12]",)


def test_it_travels_as_its_name_and_its_fields():
    assert Jax().ref() == PluginRef(kind="jax", params={"cuda": "cu124"})


def test_forming_the_cluster_is_collective():
    assert Jax.collective is True
