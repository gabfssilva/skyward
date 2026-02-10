from dataclasses import FrozenInstanceError
import pytest

from skyward.actors.messages import Provision, Replace


def test_provision_message():
    msg = Provision(cluster_id="c-123", provider="aws")
    assert msg.cluster_id == "c-123"


def test_replace_message():
    msg = Replace(old_instance_id="i-old", reason="spot-interruption")
    assert msg.old_instance_id == "i-old"
    assert msg.reason == "spot-interruption"


def test_provision_is_frozen():
    msg = Provision(cluster_id="c-123", provider="aws")
    with pytest.raises(FrozenInstanceError):
        msg.cluster_id = "other"  # type: ignore
