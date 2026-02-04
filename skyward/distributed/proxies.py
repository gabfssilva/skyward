"""Proxy wrappers for distributed collections."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

import ray

from .types import Consistency

if TYPE_CHECKING:
    from ray.actor import ActorHandle


class CounterProxy:
    """Sync/async proxy for CounterActor."""

    __slots__ = ("_actor", "_consistency")

    def __init__(self, actor: ActorHandle, consistency: Consistency = "eventual") -> None:
        self._actor = actor
        self._consistency = consistency

    @property
    def value(self) -> int:
        return ray.get(self._actor.get.remote())

    def increment(self, n: int = 1) -> None:
        ref = self._actor.increment.remote(n)
        if self._consistency == "strong":
            ray.get(ref)

    def decrement(self, n: int = 1) -> None:
        ref = self._actor.decrement.remote(n)
        if self._consistency == "strong":
            ray.get(ref)

    def reset(self, value: int = 0) -> None:
        ref = self._actor.reset.remote(value)
        if self._consistency == "strong":
            ray.get(ref)

    def __int__(self) -> int:
        return self.value

    # Async methods
    async def value_async(self) -> int:
        return await self._actor.get.remote()

    async def increment_async(self, n: int = 1) -> None:
        ref = self._actor.increment.remote(n)
        if self._consistency == "strong":
            await ref

    async def decrement_async(self, n: int = 1) -> None:
        ref = self._actor.decrement.remote(n)
        if self._consistency == "strong":
            await ref

    async def reset_async(self, value: int = 0) -> None:
        ref = self._actor.reset.remote(value)
        if self._consistency == "strong":
            await ref


class DictProxy:
    """Sync/async proxy for DictActor."""

    __slots__ = ("_actor", "_consistency")

    def __init__(self, actor: ActorHandle, consistency: Consistency = "eventual") -> None:
        self._actor = actor
        self._consistency = consistency

    def __getitem__(self, key: str):
        return ray.get(self._actor.get.remote(key))

    def __setitem__(self, key: str, value) -> None:
        ref = self._actor.set.remote(key, value)
        if self._consistency == "strong":
            ray.get(ref)

    def __delitem__(self, key: str) -> None:
        ref = self._actor.delete.remote(key)
        if self._consistency == "strong":
            ray.get(ref)

    def __contains__(self, key: str) -> bool:
        return ray.get(self._actor.contains.remote(key))

    def __len__(self) -> int:
        return ray.get(self._actor.length.remote())

    def get(self, key: str, default=None):
        return ray.get(self._actor.get.remote(key, default))

    def update(self, items: dict) -> None:
        ref = self._actor.update.remote(items)
        if self._consistency == "strong":
            ray.get(ref)

    def keys(self) -> list:
        return ray.get(self._actor.keys.remote())

    def values(self) -> list:
        return ray.get(self._actor.values.remote())

    def items(self) -> list[tuple]:
        return ray.get(self._actor.items.remote())

    def clear(self) -> None:
        ref = self._actor.clear.remote()
        if self._consistency == "strong":
            ray.get(ref)

    def pop(self, key: str, default=None):
        return ray.get(self._actor.pop.remote(key, default))

    # Async methods
    async def get_async(self, key: str, default=None):
        return await self._actor.get.remote(key, default)

    async def set_async(self, key: str, value) -> None:
        ref = self._actor.set.remote(key, value)
        if self._consistency == "strong":
            await ref

    async def update_async(self, items: dict) -> None:
        ref = self._actor.update.remote(items)
        if self._consistency == "strong":
            await ref

    async def pop_async(self, key: str, default=None):
        return await self._actor.pop.remote(key, default)

    async def clear_async(self) -> None:
        ref = self._actor.clear.remote()
        if self._consistency == "strong":
            await ref

    async def keys_async(self) -> list:
        return await self._actor.keys.remote()

    async def values_async(self) -> list:
        return await self._actor.values.remote()

    async def items_async(self) -> list[tuple]:
        return await self._actor.items.remote()


class ListProxy:
    """Sync/async proxy for ListActor."""

    __slots__ = ("_actor", "_consistency")

    def __init__(self, actor: ActorHandle, consistency: Consistency = "eventual") -> None:
        self._actor = actor
        self._consistency = consistency

    def __getitem__(self, index: int):
        return ray.get(self._actor.get.remote(index))

    def __len__(self) -> int:
        return ray.get(self._actor.length.remote())

    def append(self, value) -> None:
        ref = self._actor.append.remote(value)
        if self._consistency == "strong":
            ray.get(ref)

    def extend(self, values: list) -> None:
        ref = self._actor.extend.remote(values)
        if self._consistency == "strong":
            ray.get(ref)

    def pop(self, index: int = -1):
        return ray.get(self._actor.pop.remote(index))

    def slice(self, start: int, end: int) -> list:
        return ray.get(self._actor.slice.remote(start, end))

    def clear(self) -> None:
        ref = self._actor.clear.remote()
        if self._consistency == "strong":
            ray.get(ref)

    # Async methods
    async def append_async(self, value) -> None:
        ref = self._actor.append.remote(value)
        if self._consistency == "strong":
            await ref

    async def extend_async(self, values: list) -> None:
        ref = self._actor.extend.remote(values)
        if self._consistency == "strong":
            await ref

    async def pop_async(self, index: int = -1):
        return await self._actor.pop.remote(index)

    async def slice_async(self, start: int, end: int) -> list:
        return await self._actor.slice.remote(start, end)


class SetProxy:
    """Sync/async proxy for SetActor."""

    __slots__ = ("_actor", "_consistency")

    def __init__(self, actor: ActorHandle, consistency: Consistency = "eventual") -> None:
        self._actor = actor
        self._consistency = consistency

    def __contains__(self, value) -> bool:
        return ray.get(self._actor.contains.remote(value))

    def __len__(self) -> int:
        return ray.get(self._actor.length.remote())

    def add(self, value) -> None:
        ref = self._actor.add.remote(value)
        if self._consistency == "strong":
            ray.get(ref)

    def discard(self, value) -> None:
        ref = self._actor.discard.remote(value)
        if self._consistency == "strong":
            ray.get(ref)

    def clear(self) -> None:
        ref = self._actor.clear.remote()
        if self._consistency == "strong":
            ray.get(ref)

    # Async methods
    async def add_async(self, value) -> None:
        ref = self._actor.add.remote(value)
        if self._consistency == "strong":
            await ref

    async def discard_async(self, value) -> None:
        ref = self._actor.discard.remote(value)
        if self._consistency == "strong":
            await ref

    async def contains_async(self, value) -> bool:
        return await self._actor.contains.remote(value)


class QueueProxy:
    """Sync/async proxy for QueueActor.

    Note: Queue always uses strong consistency for FIFO semantics.
    """

    __slots__ = ("_actor",)

    def __init__(self, actor: ActorHandle) -> None:
        self._actor = actor

    def __len__(self) -> int:
        return ray.get(self._actor.length.remote())

    def put(self, value) -> None:
        ray.get(self._actor.put.remote(value))

    def get(self, timeout: float | None = None):
        """Get item from queue, blocking until available or timeout."""
        start = time.monotonic()
        while True:
            result = ray.get(self._actor.get_nowait.remote())
            if result is not None:
                return result
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    return None
            time.sleep(0.01)  # Small sleep to avoid busy-waiting

    def empty(self) -> bool:
        return ray.get(self._actor.empty.remote())

    # Async methods
    async def put_async(self, value) -> None:
        await self._actor.put.remote(value)

    async def get_async(self, timeout: float | None = None):
        """Get item from queue asynchronously."""
        start = time.monotonic()
        while True:
            result = await self._actor.get_nowait.remote()
            if result is not None:
                return result
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    return None
            await asyncio.sleep(0.01)


class BarrierProxy:
    """Sync/async proxy for BarrierActor.

    Note: Barrier always uses strong consistency.
    """

    __slots__ = ("_actor",)

    def __init__(self, actor: ActorHandle) -> None:
        self._actor = actor

    def wait(self) -> None:
        """Wait for all parties to arrive at barrier."""
        _, generation = ray.get(self._actor.arrive.remote())

        while True:
            current_n, current_count, current_gen = ray.get(self._actor.get_state.remote())
            if current_gen > generation:
                # Barrier was reset, we're released
                return
            if current_count >= current_n:
                return
            time.sleep(0.01)

    def reset(self) -> None:
        """Reset barrier for reuse."""
        ray.get(self._actor.reset.remote())

    # Async methods
    async def wait_async(self) -> None:
        """Wait for all parties asynchronously."""
        _, generation = await self._actor.arrive.remote()

        while True:
            current_n, current_count, current_gen = await self._actor.get_state.remote()
            if current_gen > generation:
                return
            if current_count >= current_n:
                return
            await asyncio.sleep(0.01)


class LockProxy:
    """Sync/async proxy for LockActor.

    Note: Lock always uses strong consistency.
    """

    __slots__ = ("_actor", "_holder_id")

    def __init__(self, actor: ActorHandle) -> None:
        self._actor = actor
        self._holder_id = str(uuid.uuid4())

    def acquire(self) -> bool:
        """Acquire lock, blocking until available."""
        while True:
            if ray.get(self._actor.acquire.remote(self._holder_id)):
                return True
            time.sleep(0.01)

    def release(self) -> None:
        """Release lock."""
        ray.get(self._actor.release.remote(self._holder_id))

    def __enter__(self) -> LockProxy:
        self.acquire()
        return self

    def __exit__(self, *args) -> None:
        self.release()

    # Async methods
    async def acquire_async(self) -> bool:
        """Acquire lock asynchronously."""
        while True:
            if await self._actor.acquire.remote(self._holder_id):
                return True
            await asyncio.sleep(0.01)

    async def release_async(self) -> None:
        """Release lock asynchronously."""
        await self._actor.release.remote(self._holder_id)

    async def __aenter__(self) -> LockProxy:
        await self.acquire_async()
        return self

    async def __aexit__(self, *args) -> None:
        await self.release_async()


__all__ = [
    "CounterProxy",
    "DictProxy",
    "ListProxy",
    "SetProxy",
    "QueueProxy",
    "BarrierProxy",
    "LockProxy",
]
