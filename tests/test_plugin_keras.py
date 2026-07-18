"""The keras plugin, checked without keras: what it installs and how it travels.

The heavy import lives inside ``setup`` and only on the jax backend across more than
one node, so none of this touches it — the image transform and the wire form are
all pure struct work.
"""

import os

from skyward.plugins.keras import Keras
from skyward.protocol.schemas import Image, PipIndex
from skyward.runtime.api import Info


def test_it_installs_keras_and_the_chosen_backend():
    built = Keras(backend="torch").image(Image(pip=("numpy",)))

    assert built.pip == ("numpy", "keras", "torch")


def test_it_sets_the_backend_in_the_worker_before_keras_imports(monkeypatch):
    monkeypatch.setenv("KERAS_BACKEND", "unset")
    info = Info(node="n", compute="c", rank=0, peers=("host",))

    with Keras(backend="tensorflow").setup(info):
        assert os.environ["KERAS_BACKEND"] == "tensorflow"


def test_the_default_backend_is_jax():
    built = Keras().image(Image())

    assert built.pip == ("keras", "jax")


def test_it_leaves_the_rest_of_the_image_alone():
    base = Image(apt=("git",), pip_indexes=(PipIndex(url="https://example"),))
    built = Keras().image(base)

    assert built.apt == ("git",)
    assert built.pip_indexes == (PipIndex(url="https://example"),)


def test_it_travels_as_its_kind_and_its_backend():
    ref = Keras(backend="torch").ref()

    assert ref.kind == "keras"
    assert ref.params == {"backend": "torch"}
