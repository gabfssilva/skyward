"""Registry for distributed collections."""

from __future__ import annotations

import ray

from .actors import (
    CounterActor,
    DictActor,
    ListActor,
    SetActor,
    QueueActor,
    BarrierActor,
    LockActor,
)
from .proxies import (
    CounterProxy,
    DictProxy,
    ListProxy,
    SetProxy,
    QueueProxy,
    BarrierProxy,
    LockProxy,
)
from .types import Consistency


NAMESPACE = "skyward"


class DistributedRegistry:
    """Registry for distributed collections.

    Manages get-or-create semantics and cleanup of Ray Actors.
    All actors are created in a fixed "skyward" namespace so they're
    accessible from any Ray Job in the cluster.
    """

    __slots__ = ("_actors",)

    def __init__(self) -> None:
        self._actors: dict[str, ray.ActorHandle] = {}

    def _get_or_create(
        self,
        actor_cls,
        name: str,
        *args,
        **kwargs,
    ) -> ray.ActorHandle:
        """Get existing actor or create new one."""
        cls_name = actor_cls.__ray_metadata__.class_name.lower()
        full_name = f"skyward:{cls_name}:{name}"

        if full_name in self._actors:
            return self._actors[full_name]

        # Use get_if_exists=True to handle race conditions where multiple
        # workers try to create the same actor simultaneously
        actor = actor_cls.options(
            name=full_name,
            namespace=NAMESPACE,
            lifetime="detached",
            get_if_exists=True,
        ).remote(*args, **kwargs)

        self._actors[full_name] = actor
        return actor

    def dict(
        self,
        name: str,
        *,
        consistency: Consistency | None = None,
    ) -> DictProxy:
        """Get or create a distributed dict."""
        actor = self._get_or_create(DictActor, name)
        return DictProxy(actor, consistency=consistency or "eventual")

    def list(
        self,
        name: str,
        *,
        consistency: Consistency | None = None,
    ) -> ListProxy:
        """Get or create a distributed list."""
        actor = self._get_or_create(ListActor, name)
        return ListProxy(actor, consistency=consistency or "eventual")

    def set(
        self,
        name: str,
        *,
        consistency: Consistency | None = None,
    ) -> SetProxy:
        """Get or create a distributed set."""
        actor = self._get_or_create(SetActor, name)
        return SetProxy(actor, consistency=consistency or "eventual")

    def counter(
        self,
        name: str,
        *,
        consistency: Consistency | None = None,
    ) -> CounterProxy:
        """Get or create a distributed counter."""
        actor = self._get_or_create(CounterActor, name)
        return CounterProxy(actor, consistency=consistency or "eventual")

    def queue(self, name: str) -> QueueProxy:
        """Get or create a distributed queue."""
        actor = self._get_or_create(QueueActor, name)
        return QueueProxy(actor)

    def barrier(self, name: str, n: int) -> BarrierProxy:
        """Get or create a distributed barrier."""
        actor = self._get_or_create(BarrierActor, name, n)
        return BarrierProxy(actor)

    def lock(self, name: str) -> LockProxy:
        """Get or create a distributed lock."""
        actor = self._get_or_create(LockActor, name)
        return LockProxy(actor)

    def cleanup(self) -> None:
        """Destroy all managed actors."""
        for actor in self._actors.values():
            try:
                ray.kill(actor)
            except Exception:
                pass  # Actor may already be dead
        self._actors.clear()


__all__ = ["DistributedRegistry"]
