"""The cuML plugin, as the spec sees it: a name, a field, and the image it asks for."""

from skyward.plugins.cuml import Cuml
from skyward.protocol.schemas import Image, PipIndex, PluginRef


def test_the_image_asks_for_the_rapids_wheel_from_nvidias_index():
    built = Cuml().image(Image(pip=("numpy",)))

    assert built.pip == ("numpy", "cuml-cu12")
    assert built.pip_indexes == (PipIndex(url="https://pypi.nvidia.com", packages=("cuml-cu12",)),)


def test_the_cuda_suffix_names_the_package_and_scopes_the_index():
    built = Cuml(cuda="cu11").image(Image())

    assert built.pip == ("cuml-cu11",)
    assert built.pip_indexes == (PipIndex(url="https://pypi.nvidia.com", packages=("cuml-cu11",)),)


def test_the_plugin_travels_as_its_name_and_its_field():
    assert Cuml(cuda="cu11").ref() == PluginRef(kind="cuml", params={"cuda": "cu11"})
