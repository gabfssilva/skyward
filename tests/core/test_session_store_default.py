"""F1: Session accepts Store | None and manages Store lifecycle correctly.

When ``store=None`` the Session constructs a private in-memory Store and
closes it on exit. When a caller supplies an open Store the Session must
not close it.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestSessionStoreDefault:
    def test_session_creates_default_store_when_none(self) -> None:
        from skyward.core.session import Session

        with Session(console=False, logging=False) as session:
            assert session._store is not None
            assert session._owns_store is True

    def test_session_closes_owned_store_on_exit(self) -> None:
        from skyward.core.session import Session

        with Session(console=False, logging=False) as session:
            store = session._store

        assert store is not None
        assert store._write is None
        assert store._read is None

    @pytest.mark.asyncio
    async def test_session_preserves_caller_store(self) -> None:
        import asyncio

        from skyward.core.session import Session
        from skyward.server.host.store import Store

        store = Store(":memory:")
        await store.open()
        try:
            loop = asyncio.get_running_loop()

            def _run() -> None:
                with Session(console=False, logging=False, store=store) as session:
                    assert session._store is store
                    assert session._owns_store is False

            await loop.run_in_executor(None, _run)
            assert store._write is not None
            assert store._read is not None
        finally:
            await store.close()
