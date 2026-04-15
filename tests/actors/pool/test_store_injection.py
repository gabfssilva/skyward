"""F1: verify every actor factory accepts a closure-injected Store."""
from __future__ import annotations

import inspect

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestStoreInjection:
    def test_pool_actor_accepts_store_kw(self) -> None:
        from skyward.actors.pool.actor import pool_actor

        sig = inspect.signature(pool_actor)
        assert "store" in sig.parameters
        p = sig.parameters["store"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY

    def test_reconciler_actor_accepts_store_kw(self) -> None:
        from skyward.actors.reconciler.actor import reconciler_actor

        sig = inspect.signature(reconciler_actor)
        assert "store" in sig.parameters
        assert sig.parameters["store"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_task_manager_actor_accepts_store_kw(self) -> None:
        from skyward.actors.task_manager.actor import task_manager_actor

        sig = inspect.signature(task_manager_actor)
        assert "store" in sig.parameters
        assert sig.parameters["store"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_node_actor_accepts_store_kw(self) -> None:
        from skyward.actors.node.actor import node_actor

        sig = inspect.signature(node_actor)
        assert "store" in sig.parameters
        assert sig.parameters["store"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_session_actor_accepts_store_kw(self) -> None:
        from skyward.actors.session.actor import session_actor

        sig = inspect.signature(session_actor)
        assert "store" in sig.parameters
        assert sig.parameters["store"].kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.asyncio
    async def test_pool_actor_can_be_spawned_with_store(self) -> None:
        """Pool actor spawns without error when a real :memory: Store is passed."""
        from casty import ActorSystem

        from skyward.actors.pool.actor import pool_actor
        from skyward.server.host.store import Store

        store = Store(":memory:")
        await store.open()
        try:
            system = ActorSystem("test-f1-pool")
            async with system:
                ref = system.spawn(
                    pool_actor(pool_name="unit", store=store),
                    "pool-unit",
                )
                assert ref is not None
        finally:
            await store.close()
