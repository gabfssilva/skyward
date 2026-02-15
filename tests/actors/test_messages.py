from dataclasses import FrozenInstanceError

import pytest


def test_instance_messages_are_frozen():
    from skyward.actors.messages import Bootstrapping, Execute, Log, Metric, Preempted, Running

    msg = Running(ip="10.0.0.1")
    assert msg.ip == "10.0.0.1"
    with pytest.raises(FrozenInstanceError):
        msg.ip = "changed"  # type: ignore[misc]


def test_node_messages_are_frozen():
    from skyward.actors.messages import (
        ExecuteOnNode,
        InstanceBecameReady,
        InstanceDied,
        InstanceLaunched,
        InstanceRunning,
        Provision,
    )

    msg = Provision(cluster_id="c1", provider_ref=None, cluster_client=None)  # type: ignore[arg-type]
    assert msg.cluster_id == "c1"


def test_task_manager_messages_are_frozen():
    from skyward.actors.messages import NodeAvailable, NodeSlots, NodeUnavailable, SlotFreed, SubmitBroadcast, SubmitTask

    slots = NodeSlots(ref=None, total=4, used=1)  # type: ignore[arg-type]
    assert slots.total == 4
    assert slots.used == 1


def test_pool_messages_are_frozen():
    from skyward.actors.messages import NodeBecameReady, NodeLost, PoolStarted, StartPool, StopPool, SubmitTask
