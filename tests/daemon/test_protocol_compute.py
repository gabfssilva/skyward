import cloudpickle
import pytest

from skyward.daemon.protocol import (
    DaemonRequest,
    DaemonResponse,
    GetPoolView,
    GetPools,
    PoolList,
    PoolSummary,
    PoolViewResponse,
    SubscribeEvents,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestNewRequestMessages:
    def test_get_pools_is_frozen(self) -> None:
        msg = GetPools()
        with pytest.raises((AttributeError, TypeError)):
            msg.x = 1  # type: ignore[attr-defined]

    def test_get_pool_view_carries_name(self) -> None:
        msg = GetPoolView(pool_name="train")
        assert msg.pool_name == "train"

    def test_get_pool_view_is_frozen(self) -> None:
        msg = GetPoolView(pool_name="train")
        with pytest.raises(AttributeError):
            msg.pool_name = "other"  # type: ignore[misc]

    def test_subscribe_events_carries_name(self) -> None:
        msg = SubscribeEvents(pool_name="infer")
        assert msg.pool_name == "infer"

    def test_subscribe_events_is_frozen(self) -> None:
        msg = SubscribeEvents(pool_name="infer")
        with pytest.raises(AttributeError):
            msg.pool_name = "other"  # type: ignore[misc]


class TestNewResponseMessages:
    def test_pool_summary_fields(self) -> None:
        summary = PoolSummary(
            name="train", phase="READY", nodes_ready=2, nodes_total=4,
            tasks_done=10, tasks_running=3, started_at=1000.0,
        )
        assert summary.name == "train"
        assert summary.phase == "READY"
        assert summary.nodes_ready == 2
        assert summary.nodes_total == 4
        assert summary.tasks_done == 10
        assert summary.tasks_running == 3
        assert summary.started_at == 1000.0

    def test_pool_summary_is_frozen(self) -> None:
        summary = PoolSummary(
            name="train", phase="READY", nodes_ready=2, nodes_total=4,
            tasks_done=10, tasks_running=3, started_at=1000.0,
        )
        with pytest.raises(AttributeError):
            summary.name = "other"  # type: ignore[misc]

    def test_pool_list_carries_tuple(self) -> None:
        s1 = PoolSummary(
            name="a", phase="READY", nodes_ready=1, nodes_total=1,
            tasks_done=0, tasks_running=0, started_at=0.0,
        )
        s2 = PoolSummary(
            name="b", phase="PROVISIONING", nodes_ready=0, nodes_total=2,
            tasks_done=0, tasks_running=0, started_at=0.0,
        )
        msg = PoolList(pools=(s1, s2))
        assert len(msg.pools) == 2
        assert msg.pools[0].name == "a"
        assert msg.pools[1].name == "b"

    def test_pool_list_empty(self) -> None:
        msg = PoolList(pools=())
        assert msg.pools == ()

    def test_pool_view_response_carries_view(self) -> None:
        sentinel = object()
        msg = PoolViewResponse(view=sentinel)
        assert msg.view is sentinel


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


class TestTypeUnionsIncludeNewMessages:
    def test_get_pools_in_request_union(self) -> None:
        msg: DaemonRequest = GetPools()  # type: ignore[assignment]
        assert isinstance(msg, GetPools)

    def test_get_pool_view_in_request_union(self) -> None:
        msg: DaemonRequest = GetPoolView(pool_name="p")  # type: ignore[assignment]
        assert isinstance(msg, GetPoolView)

    def test_subscribe_events_in_request_union(self) -> None:
        msg: DaemonRequest = SubscribeEvents(pool_name="p")  # type: ignore[assignment]
        assert isinstance(msg, SubscribeEvents)

    def test_pool_list_in_response_union(self) -> None:
        msg: DaemonResponse = PoolList(pools=())  # type: ignore[assignment]
        assert isinstance(msg, PoolList)

    def test_pool_view_response_in_response_union(self) -> None:
        msg: DaemonResponse = PoolViewResponse(view=None)  # type: ignore[assignment]
        assert isinstance(msg, PoolViewResponse)
