from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, get_args
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _mock_ref() -> Any:
    return MagicMock()


def _mock_spec() -> Any:
    return MagicMock()


def _mock_provider_config() -> Any:
    return MagicMock()


class TestPoolInfo:
    def test_construction(self) -> None:
        from skyward.actors.session.messages import PoolInfo

        info = PoolInfo(
            name="train",
            ref=_mock_ref(),
            spec=_mock_spec(),
            phase="ready",
            nodes_ready=3,
            nodes_total=4,
        )
        assert info.name == "train"
        assert info.phase == "ready"
        assert info.nodes_ready == 3
        assert info.nodes_total == 4

    def test_frozen(self) -> None:
        from skyward.actors.session.messages import PoolInfo

        info = PoolInfo(
            name="train",
            ref=_mock_ref(),
            spec=_mock_spec(),
            phase="ready",
            nodes_ready=3,
            nodes_total=4,
        )
        with pytest.raises(FrozenInstanceError):
            info.name = "other"  # type: ignore[misc]


class TestSpawnPool:
    def test_construction(self) -> None:
        from skyward.actors.session.messages import SpawnPool

        msg = SpawnPool(
            name="train",
            spec=_mock_spec(),
            provider_config=_mock_provider_config(),
            provider=object(),
            offers=(),
            provision_timeout=300.0,
            reply_to=_mock_ref(),
        )
        assert msg.name == "train"
        assert msg.provision_timeout == 300.0
        assert msg.offers == ()

    def test_frozen(self) -> None:
        from skyward.actors.session.messages import SpawnPool

        msg = SpawnPool(
            name="train",
            spec=_mock_spec(),
            provider_config=_mock_provider_config(),
            provider=object(),
            offers=(),
            provision_timeout=300.0,
            reply_to=_mock_ref(),
        )
        with pytest.raises(FrozenInstanceError):
            msg.name = "other"  # type: ignore[misc]


class TestPoolSpawned:
    def test_construction(self) -> None:
        from skyward.actors.session.messages import PoolSpawned

        msg = PoolSpawned(
            name="train",
            pool_ref=_mock_ref(),
            cluster_id="c-123",
            instances=(object(),),
            cluster=MagicMock(),
        )
        assert msg.name == "train"
        assert msg.cluster_id == "c-123"
        assert len(msg.instances) == 1

    def test_frozen(self) -> None:
        from skyward.actors.session.messages import PoolSpawned

        msg = PoolSpawned(
            name="train",
            pool_ref=_mock_ref(),
            cluster_id="c-123",
            instances=(),
            cluster=MagicMock(),
        )
        with pytest.raises(FrozenInstanceError):
            msg.name = "other"  # type: ignore[misc]


class TestPoolSpawnFailed:
    def test_construction(self) -> None:
        from skyward.actors.session.messages import PoolSpawnFailed

        msg = PoolSpawnFailed(name="train", reason="out of capacity")
        assert msg.name == "train"
        assert msg.reason == "out of capacity"

    def test_frozen(self) -> None:
        from skyward.actors.session.messages import PoolSpawnFailed

        msg = PoolSpawnFailed(name="train", reason="timeout")
        with pytest.raises(FrozenInstanceError):
            msg.reason = "other"  # type: ignore[misc]


class TestPoolStateChanged:
    def test_construction(self) -> None:
        from skyward.actors.session.messages import PoolStateChanged

        msg = PoolStateChanged(
            name="train",
            phase="provisioning",
            nodes_ready=0,
            nodes_total=4,
        )
        assert msg.phase == "provisioning"
        assert msg.nodes_ready == 0

    def test_frozen(self) -> None:
        from skyward.actors.session.messages import PoolStateChanged

        msg = PoolStateChanged(
            name="train", phase="ready", nodes_ready=4, nodes_total=4,
        )
        with pytest.raises(FrozenInstanceError):
            msg.phase = "stopped"  # type: ignore[misc]


class TestStopSession:
    def test_construction(self) -> None:
        from skyward.actors.session.messages import StopSession

        msg = StopSession(reply_to=_mock_ref())
        assert msg.reply_to is not None

    def test_frozen(self) -> None:
        from skyward.actors.session.messages import StopSession

        msg = StopSession(reply_to=_mock_ref())
        with pytest.raises(FrozenInstanceError):
            msg.reply_to = _mock_ref()  # type: ignore[misc]


class TestSessionStopped:
    def test_construction(self) -> None:
        from skyward.actors.session.messages import SessionStopped

        msg = SessionStopped()
        assert msg is not None

    def test_frozen(self) -> None:
        from dataclasses import fields as dc_fields

        from skyward.actors.session.messages import SessionStopped

        msg = SessionStopped()
        assert dc_fields(msg) == ()
        with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
            msg.x = 1  # type: ignore[attr-defined]


class TestGetSessionSnapshot:
    def test_construction(self) -> None:
        from skyward.actors.session.messages import GetSessionSnapshot

        msg = GetSessionSnapshot(reply_to=_mock_ref())
        assert msg.reply_to is not None

    def test_frozen(self) -> None:
        from skyward.actors.session.messages import GetSessionSnapshot

        msg = GetSessionSnapshot(reply_to=_mock_ref())
        with pytest.raises(FrozenInstanceError):
            msg.reply_to = _mock_ref()  # type: ignore[misc]


class TestSessionSnapshot:
    def test_construction(self) -> None:
        from skyward.actors.session.messages import PoolInfo, SessionSnapshot

        info = PoolInfo(
            name="train",
            ref=_mock_ref(),
            spec=_mock_spec(),
            phase="ready",
            nodes_ready=2,
            nodes_total=2,
        )
        snap = SessionSnapshot(pools=(info,))
        assert len(snap.pools) == 1
        assert snap.pools[0].name == "train"

    def test_frozen(self) -> None:
        from skyward.actors.session.messages import SessionSnapshot

        snap = SessionSnapshot(pools=())
        with pytest.raises(FrozenInstanceError):
            snap.pools = ()  # type: ignore[misc]


class TestInternalMessages:
    def test_pool_ready_construction(self) -> None:
        from skyward.actors.session.messages import _PoolReady

        msg = _PoolReady(
            name="train",
            cluster_id="c-123",
            instances=(object(),),
            cluster=MagicMock(),
            pool_ref=_mock_ref(),
        )
        assert msg.name == "train"
        assert msg.cluster_id == "c-123"

    def test_pool_ready_frozen(self) -> None:
        from skyward.actors.session.messages import _PoolReady

        msg = _PoolReady(
            name="train",
            cluster_id="c-123",
            instances=(),
            cluster=MagicMock(),
            pool_ref=_mock_ref(),
        )
        with pytest.raises(FrozenInstanceError):
            msg.name = "other"  # type: ignore[misc]

    def test_pool_failed_construction(self) -> None:
        from skyward.actors.session.messages import _PoolFailed

        msg = _PoolFailed(name="train", reason="timeout")
        assert msg.name == "train"
        assert msg.reason == "timeout"

    def test_pool_failed_frozen(self) -> None:
        from skyward.actors.session.messages import _PoolFailed

        msg = _PoolFailed(name="train", reason="timeout")
        with pytest.raises(FrozenInstanceError):
            msg.reason = "other"  # type: ignore[misc]


class TestSessionMsgTypeAlias:
    def test_includes_all_expected_types(self) -> None:
        from skyward.actors.session.messages import (
            GetSessionSnapshot,
            PoolStateChanged,
            SessionMsg,
            SpawnPool,
            StopSession,
            _PoolFailed,
            _PoolReady,
            _SnapshotReady,
        )

        args = set(get_args(SessionMsg.__value__))
        expected = {
            SpawnPool,
            StopSession,
            PoolStateChanged,
            GetSessionSnapshot,
            _SnapshotReady,
            _PoolReady,
            _PoolFailed,
        }
        assert args == expected


class TestPackageExports:
    def test_public_types_exported(self) -> None:
        import skyward.actors.session as session_pkg

        expected_exports = {
            "PoolInfo",
            "SpawnPool",
            "PoolSpawned",
            "PoolSpawnFailed",
            "PoolStateChanged",
            "StopSession",
            "SessionStopped",
            "GetSessionSnapshot",
            "SessionSnapshot",
            "SessionMsg",
            "PoolPhase",
        }
        assert expected_exports.issubset(set(session_pkg.__all__))

    def test_internal_types_not_exported(self) -> None:
        import skyward.actors.session as session_pkg

        assert "_PoolReady" not in session_pkg.__all__
        assert "_PoolFailed" not in session_pkg.__all__


class TestStartAdapter:
    @pytest.fixture
    async def system(self):
        from casty import ActorSystem

        s = ActorSystem("test-adapter")
        yield s
        await s.shutdown()

    @pytest.mark.asyncio
    async def test_converts_pool_started_to_pool_ready(self, system) -> None:
        import asyncio
        from unittest.mock import MagicMock

        from skyward.actors.pool.messages import PoolStarted
        from skyward.actors.session.adapter import start_adapter
        from skyward.actors.session.messages import _PoolReady

        collected: list[_PoolReady] = []
        session_ref = MagicMock()
        session_ref.tell = lambda msg: collected.append(msg)

        pool_ref = MagicMock()

        cluster = MagicMock()
        instances = (MagicMock(),)

        ref = system.spawn(
            start_adapter(name="train", session=session_ref, pool_ref=pool_ref),
            "adapter-success",
        )
        await asyncio.sleep(0.1)

        ref.tell(PoolStarted(cluster_id="c-42", instances=instances, cluster=cluster))
        await asyncio.sleep(0.2)

        assert len(collected) == 1
        assert collected[0].name == "train"
        assert collected[0].cluster_id == "c-42"
        assert collected[0].instances == instances
        assert collected[0].cluster is cluster
        assert collected[0].pool_ref is pool_ref

    @pytest.mark.asyncio
    async def test_converts_provision_failed_to_pool_failed(self, system) -> None:
        import asyncio
        from unittest.mock import MagicMock

        from skyward.actors.pool.messages import ProvisionFailed
        from skyward.actors.session.adapter import start_adapter
        from skyward.actors.session.messages import _PoolFailed

        collected: list[_PoolFailed] = []
        session_ref = MagicMock()
        session_ref.tell = lambda msg: collected.append(msg)

        pool_ref = MagicMock()

        ref = system.spawn(
            start_adapter(name="train", session=session_ref, pool_ref=pool_ref),
            "adapter-failure",
        )
        await asyncio.sleep(0.1)

        ref.tell(ProvisionFailed(reason="out of capacity"))
        await asyncio.sleep(0.2)

        assert len(collected) == 1
        assert collected[0].name == "train"
        assert collected[0].reason == "out of capacity"


class TestSessionActor:
    @pytest.fixture
    async def system(self):
        from casty import ActorSystem

        s = ActorSystem("test-session")
        yield s
        await s.shutdown()

    @pytest.mark.asyncio
    async def test_empty_snapshot(self, system) -> None:
        import asyncio

        from skyward.actors.session.actor import session_actor
        from skyward.actors.session.messages import (
            GetSessionSnapshot,
            SessionSnapshot,
        )

        ref = system.spawn(session_actor(), "session-empty")
        await asyncio.sleep(0.1)

        collected: list[SessionSnapshot] = []
        reply_ref = MagicMock()
        reply_ref.tell = lambda msg: collected.append(msg)

        ref.tell(GetSessionSnapshot(reply_to=reply_ref))
        await asyncio.sleep(0.2)

        assert len(collected) == 1
        assert isinstance(collected[0], SessionSnapshot)
        assert collected[0].pools == ()

    @pytest.mark.asyncio
    async def test_pool_state_changed_updates_aggregate(self, system) -> None:
        import asyncio

        from casty import Behaviors

        from skyward.actors.messages import GetPoolSnapshot
        from skyward.actors.session.actor import session_actor
        from skyward.actors.session.messages import (
            GetSessionSnapshot,
            PoolSpawned,
            PoolSpawnFailed,
            PoolStateChanged,
            SessionSnapshot,
            SpawnPool,
            _PoolReady,
        )
        from skyward.actors.snapshot import (
            PoolPhase,
            PoolSnapshot,
            ScalingSnapshot,
            TaskCounters,
        )

        stub_snapshot = PoolSnapshot(
            name="train",
            phase=PoolPhase.READY,
            nodes=(),
            tasks=TaskCounters(),
            scaling=ScalingSnapshot(),
        )

        def _stub_pool():
            async def handle(ctx, msg):
                match msg:
                    case GetPoolSnapshot(reply_to=reply_to):
                        reply_to.tell(stub_snapshot)
                return Behaviors.same()
            return Behaviors.receive(handle)

        pool_ref = system.spawn(_stub_pool(), "stub-pool")
        await asyncio.sleep(0.1)

        ref = system.spawn(session_actor(), "session-state")
        await asyncio.sleep(0.1)

        from skyward.core.spec import Nodes

        spec = MagicMock()
        spec.nodes = Nodes(min=4)

        spawn_replies: list[PoolSpawned | PoolSpawnFailed] = []
        spawn_reply_ref = MagicMock()
        spawn_reply_ref.tell = lambda msg: spawn_replies.append(msg)

        provider = MagicMock()
        provider.prepare = MagicMock(return_value=asyncio.sleep(3600))

        ref.tell(SpawnPool(
            name="train",
            spec=spec,
            provider_config=MagicMock(),
            provider=provider,
            offers=(MagicMock(),),
            provision_timeout=300.0,
            reply_to=spawn_reply_ref,
        ))
        await asyncio.sleep(0.3)

        ref.tell(_PoolReady(
            name="train",
            cluster_id="c-99",
            instances=(MagicMock(),),
            cluster=MagicMock(),
            pool_ref=pool_ref,
        ))
        await asyncio.sleep(0.2)

        spawned = [r for r in spawn_replies if isinstance(r, PoolSpawned)]
        assert len(spawned) == 1
        assert spawned[0].name == "train"
        assert spawned[0].cluster_id == "c-99"

        ref.tell(PoolStateChanged(
            name="train",
            phase="ready",
            nodes_ready=4,
            nodes_total=4,
        ))
        await asyncio.sleep(0.2)

        snapshots: list[SessionSnapshot] = []
        snap_ref = MagicMock()
        snap_ref.tell = lambda msg: snapshots.append(msg)
        ref.tell(GetSessionSnapshot(reply_to=snap_ref))
        await asyncio.sleep(0.5)

        assert len(snapshots) == 1
        assert len(snapshots[0].pools) == 1
        pool = snapshots[0].pools[0]
        assert pool.name == "train"
        assert pool.phase == PoolPhase.READY

    @pytest.mark.asyncio
    async def test_stop_session(self, system) -> None:
        import asyncio

        from skyward.actors.session.actor import session_actor
        from skyward.actors.session.messages import (
            SessionStopped,
            StopSession,
        )

        ref = system.spawn(session_actor(), "session-stop")
        await asyncio.sleep(0.1)

        collected: list[SessionStopped] = []
        reply_ref = MagicMock()
        reply_ref.tell = lambda msg: collected.append(msg)

        ref.tell(StopSession(reply_to=reply_ref))
        await asyncio.sleep(0.2)

        assert len(collected) == 1
        assert isinstance(collected[0], SessionStopped)


class TestSessionLifecycle:
    def test_session_enter_exit(self) -> None:
        from skyward.core.session import Session

        session = Session(console=False, logging=False)
        session.__enter__()
        assert session.is_active
        assert session._system is not None
        assert session._session_ref is not None
        session.__exit__(None, None, None)
        assert not session.is_active

    def test_session_context_manager(self) -> None:
        from skyward.core.session import Session

        with Session(console=False, logging=False) as session:
            assert session.is_active
        assert not session.is_active

    def test_session_sets_contextvar(self) -> None:
        from skyward.core.context import get_session
        from skyward.core.session import Session

        assert get_session() is None
        with Session(console=False, logging=False) as session:
            assert get_session() is session
        assert get_session() is None


class TestSessionCompute:
    def test_compute_raises_when_inactive(self) -> None:
        from unittest.mock import MagicMock

        from skyward.core.session import Session
        from skyward.core.spec import Spec

        session = Session(console=False, logging=False)
        with pytest.raises(RuntimeError, match="Session is not active"):
            session.compute(Spec(provider=MagicMock()))

    def test_compute_validates_no_specs(self) -> None:
        from unittest.mock import MagicMock

        from skyward.core.session import Session

        session = Session(console=False, logging=False)
        session._active = True
        session._loop = MagicMock()
        session._system = MagicMock()
        session._session_ref = MagicMock()

        with pytest.raises(
            ValueError, match="Either Spec objects or keyword arguments",
        ):
            session.compute()


class TestPoolActorSessionRef:
    def test_pool_actor_accepts_session_ref(self) -> None:
        from skyward.actors.pool.actor import pool_actor
        behavior = pool_actor()
        assert behavior is not None
        behavior_with_ref = pool_actor(session_ref=None, pool_name="test")
        assert behavior_with_ref is not None


class TestComputeSugar:
    def test_compute_is_importable(self) -> None:
        from skyward.core.compute import Compute

        assert callable(Compute)

    def test_compute_is_context_manager(self) -> None:
        from skyward.core.compute import Compute

        assert hasattr(Compute, '__call__')
