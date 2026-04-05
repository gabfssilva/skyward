import cloudpickle
import pytest

from skyward.daemon.protocol import (
    GetPoolView,
    GetPools,
    PoolList,
    PoolSummary,
    PoolViewResponse,
    SubscribeEvents,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestCloudpickleRoundtrip:
    def test_get_pools_roundtrip(self) -> None:
        msg = GetPools()
        restored = cloudpickle.loads(cloudpickle.dumps(msg))
        assert isinstance(restored, GetPools)

    def test_get_pool_view_roundtrip(self) -> None:
        msg = GetPoolView(pool_name="train")
        restored = cloudpickle.loads(cloudpickle.dumps(msg))
        assert isinstance(restored, GetPoolView)
        assert restored.pool_name == "train"

    def test_subscribe_events_roundtrip(self) -> None:
        msg = SubscribeEvents(pool_name="infer")
        restored = cloudpickle.loads(cloudpickle.dumps(msg))
        assert isinstance(restored, SubscribeEvents)
        assert restored.pool_name == "infer"

    def test_pool_summary_roundtrip(self) -> None:
        msg = PoolSummary(
            name="train", phase="READY", nodes_ready=2, nodes_total=4,
            tasks_done=10, tasks_running=3, started_at=1000.0,
        )
        restored = cloudpickle.loads(cloudpickle.dumps(msg))
        assert isinstance(restored, PoolSummary)
        assert restored.name == "train"
        assert restored.nodes_ready == 2

    def test_pool_list_roundtrip(self) -> None:
        summary = PoolSummary(
            name="x", phase="BOOTSTRAP", nodes_ready=0, nodes_total=1,
            tasks_done=0, tasks_running=0, started_at=42.0,
        )
        msg = PoolList(pools=(summary,))
        restored = cloudpickle.loads(cloudpickle.dumps(msg))
        assert isinstance(restored, PoolList)
        assert len(restored.pools) == 1
        assert restored.pools[0].name == "x"

    def test_pool_view_response_roundtrip(self) -> None:
        msg = PoolViewResponse(view={"mock": True})
        restored = cloudpickle.loads(cloudpickle.dumps(msg))
        assert isinstance(restored, PoolViewResponse)
        assert restored.view == {"mock": True}
