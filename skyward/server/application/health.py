from __future__ import annotations

from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.tables import ComputeRow
from skyward.shared.schemas import DependencyState


class Health:
    """Whether the daemon works, and whether what it depends on does.

    A provider that is down degrades a compute; it never makes the process
    unhealthy. Killing a daemon because Vast.ai is having an afternoon would take
    every other compute down with it.
    """

    def __init__(self, providers: ProviderStore) -> None:
        self._providers = providers

    async def live(self) -> bool:
        return True

    async def ready(self) -> bool:
        return await self._store()

    async def dependencies(self) -> dict[str, DependencyState]:
        return {"store": "ok" if await self._store() else "unreachable"}

    async def _store(self) -> bool:
        try:
            await ComputeRow.count()
            return True
        except Exception:
            return False
