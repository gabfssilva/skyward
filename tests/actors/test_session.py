from __future__ import annotations

from typing import get_args
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _mock_ref() -> MagicMock:
    return MagicMock()


def _mock_spec() -> MagicMock:
    return MagicMock()


def _mock_provider_config() -> MagicMock:
    return MagicMock()


class TestSessionMsgTypeAlias:
    def test_includes_all_expected_types(self) -> None:
        from skyward.actors.session.messages import (
            CreatePool,
            RecoverExistingPool,
            SessionMsg,
            SpawnPool,
            StopSession,
            _PoolFailed,
            _PoolReady,
        )

        args = set(get_args(SessionMsg.__value__))
        expected = {
            CreatePool,
            SpawnPool,
            RecoverExistingPool,
            StopSession,
            _PoolReady,
            _PoolFailed,
        }
        assert args == expected


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
    async def test_pool_ready_updates_pool_info(self, system) -> None:
        import asyncio

        from casty import Behaviors

        from skyward.actors.session.actor import session_actor
        from skyward.actors.session.messages import (
            PoolSpawned,
            PoolSpawnFailed,
            SpawnPool,
            _PoolReady,
        )

        def _stub_pool():
            async def handle(ctx, msg):
                return Behaviors.same()
            return Behaviors.receive(handle)

        from skyward.server.host.store import Store

        store = Store(":memory:")
        await store.open()

        pool_ref = system.spawn(_stub_pool(), "stub-pool")
        await asyncio.sleep(0.1)

        ref = system.spawn(session_actor(store=store), "session-state")
        await asyncio.sleep(0.1)

        from skyward.core.spec import Nodes

        spec = MagicMock()
        spec.nodes = Nodes(desired=4)

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
            compute_spec=MagicMock(),
            chosen_spec=MagicMock(),
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

    @pytest.mark.asyncio
    async def test_stop_session(self, system) -> None:
        import asyncio

        from skyward.actors.session.actor import session_actor
        from skyward.actors.session.messages import (
            SessionStopped,
            StopSession,
        )
        from skyward.server.host.store import Store

        store = Store(":memory:")
        await store.open()

        ref = system.spawn(session_actor(store=store), "session-stop")
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
