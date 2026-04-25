"""In-memory registries for the HTTP server."""

from __future__ import annotations

import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.api.pool import Pool
    from skyward.api.session import Session


@dataclass
class ServerState:
    session: Session
    pools: dict[str, Pool] = field(default_factory=dict)
    executions: dict[str, Future] = field(default_factory=dict)
    broadcast_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=8, thread_name_prefix="sky-bcast"),
    )

    def register_pool(self, name: str, pool: Pool) -> None:
        self.pools[name] = pool

    def get_pool(self, name: str) -> Pool | None:
        return self.pools.get(name)

    def drop_pool(self, name: str) -> Pool | None:
        return self.pools.pop(name, None)

    def register_execution(self, future: Future) -> str:
        eid = uuid.uuid4().hex
        self.executions[eid] = future
        return eid

    def get_execution(self, eid: str) -> Future | None:
        return self.executions.get(eid)

    def drop_execution(self, eid: str) -> Future | None:
        return self.executions.pop(eid, None)
