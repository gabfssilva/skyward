"""The accelerate plugin as a value: what it installs, and how it travels."""

from skyward.plugins.accelerate import Accelerate
from skyward.protocol.schemas import Image, PluginRef


def test_it_appends_accelerate_to_the_pip_list():
    built = Accelerate().image(Image(pip=("numpy",)))

    assert built.pip == ("numpy", "accelerate")


def test_it_travels_as_its_name_and_its_config():
    assert Accelerate().ref() == PluginRef(kind="accelerate", params={"config": {}})


def test_the_config_survives_the_round_trip():
    config = {"mixed_precision": "bf16", "fsdp": {"sharding_strategy": "FULL_SHARD"}}

    assert Accelerate(config=config).ref() == PluginRef(kind="accelerate", params={"config": config})


def test_forming_the_group_is_collective():
    assert Accelerate.collective is True
