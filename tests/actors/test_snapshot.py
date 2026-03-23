from skyward.actors.snapshot import (
    BootstrapTimeline,
    NodeSnapshot,
    NodeStatus,
    PoolPhase,
    PoolSnapshot,
    ScalingSnapshot,
    TaskCounters,
)
import pytest


@pytest.mark.unit
class TestPoolSnapshot:
    def test_construction(self):
        snap = PoolSnapshot(
            name="test",
            phase=PoolPhase.READY,
            nodes=(
                NodeSnapshot(node_id=0, instance_id="i-abc", status=NodeStatus.READY),
                NodeSnapshot(
                    node_id=1, instance_id="i-def", status=NodeStatus.BOOTSTRAPPING,
                    bootstrap=BootstrapTimeline(
                        phases=("apt", "uv", "deps"),
                        completed=frozenset({"apt"}),
                        active="uv", output="installing...",
                    ),
                ),
            ),
            tasks=TaskCounters(queued=2, running=3, done=10, failed=1),
            scaling=ScalingSnapshot(desired_nodes=4, is_elastic=True, min_nodes=2, max_nodes=8),
        )
        assert snap.phase == PoolPhase.READY
        assert len(snap.nodes) == 2
        assert snap.nodes[1].bootstrap is not None
        assert snap.nodes[1].bootstrap.active == "uv"
        assert snap.tasks.done == 10
        assert snap.scaling.is_elastic

    def test_defaults(self):
        snap = PoolSnapshot(
            name="minimal",
            phase=PoolPhase.PROVISIONING,
            nodes=(),
            tasks=TaskCounters(),
            scaling=ScalingSnapshot(),
        )
        assert snap.cluster is None
        assert snap.instances == ()
        assert snap.started_at == 0.0
        assert snap.tasks.queued == 0
        assert snap.scaling.reconciler_state == "watching"

    def test_frozen(self):
        snap = PoolSnapshot(
            name="test", phase=PoolPhase.READY, nodes=(),
            tasks=TaskCounters(), scaling=ScalingSnapshot(),
        )
        with pytest.raises(AttributeError):
            snap.name = "other"  # type: ignore[misc]
