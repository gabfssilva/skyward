"""A plugin is a value that survives being written down.

The user builds one, the spec carries its name and its fields, and the node builds
it back. What is tested here is that round trip, and the two places the rebuilt
plugin gets to speak.
"""

from contextlib import contextmanager
from typing import ClassVar

import pytest
from msgspec.structs import replace

from skyward2.application.errors import UnsupportedPluginError
from skyward2.plugins import HuggingFace, Plugin, Torch, chain, image, resolve
from skyward2.protocol.schemas import Image, PluginRef
from skyward2.runtime.api import Info

INFO = Info(node="nod_1", compute="cmp_1", rank=0, peers=("10.0.0.0",))


class Loud(Plugin, frozen=True):
    kind: ClassVar[str] = "loud"

    mark: str = "!"

    def image(self, image: Image) -> Image:
        return replace(image, packages=(*image.packages, self.mark))

    @contextmanager
    def setup(self, info: Info):
        yield

    def run[T](self, call, info: Info) -> T:
        return f"{self.mark}{call()}{self.mark}"


def test_a_plugin_travels_as_its_name_and_its_fields():
    ref = Torch(backend="gloo").ref()

    assert ref == PluginRef(kind="torch", params={"backend": "gloo", "version": None})
    assert resolve((ref,)) == (Torch(backend="gloo"),), "and comes back the same plugin"


def test_a_kind_nobody_has_heard_of_is_refused_rather_than_ignored():
    with pytest.raises(UnsupportedPluginError):
        resolve((PluginRef(kind="tensorflow"),))


def test_parameters_are_validated_where_the_user_can_still_be_told():
    with pytest.raises(UnsupportedPluginError, match="torch"):
        resolve((PluginRef(kind="torch", params={"backend": "mpi"}),))


def test_the_image_is_handed_to_each_plugin_in_turn():
    built = image(Image(packages=("numpy",)), (Torch(version="2.4.0"), HuggingFace()))

    assert built.packages == ("numpy", "torch==2.4.0", "huggingface_hub")


def test_plugins_wrap_a_call_with_the_first_one_outermost():
    wrapped = chain((Loud(mark="a"), Loud(mark="b")), lambda: "x", INFO)

    assert wrapped() == "abxba"
