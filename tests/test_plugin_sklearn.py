"""The sklearn plugin, as a value the spec can carry and the node can rebuild.

The heavy library lives lazily inside ``client``; none of this touches it. What is
tested is the two things the plugin decides on its own — what it puts on the node,
and how it names itself on the wire.
"""

from skyward.worker.plugins.sklearn import Sklearn
from skyward.shared.schemas import Image, PluginRef


def test_the_image_gets_scikit_learn_and_joblib():
    built = Sklearn().image(Image(pip=("numpy",)))

    assert built.pip == ("numpy", "scikit-learn", "joblib")


def test_a_version_pins_only_scikit_learn():
    built = Sklearn(version="1.4.0").image(Image())

    assert built.pip == ("scikit-learn==1.4.0", "joblib")


def test_a_plugin_travels_as_its_name_and_its_fields():
    assert Sklearn(version="1.4.0").ref() == PluginRef(kind="sklearn", params={"version": "1.4.0"})
    assert Sklearn().ref() == PluginRef(kind="sklearn", params={"version": None})
