"""Ray Actors for distributed collections."""

from __future__ import annotations

import ray


@ray.remote
class CounterActor:
    """Distributed counter backed by Ray Actor."""

    def __init__(self) -> None:
        self._value: int = 0

    def get(self) -> int:
        return self._value

    def increment(self, n: int = 1) -> int:
        self._value += n
        return self._value

    def decrement(self, n: int = 1) -> int:
        self._value -= n
        return self._value

    def reset(self, value: int = 0) -> None:
        self._value = value


@ray.remote
class DictActor:
    """Distributed dict backed by Ray Actor."""

    def __init__(self) -> None:
        self._data: dict = {}

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        del self._data[key]

    def contains(self, key: str) -> bool:
        return key in self._data

    def length(self) -> int:
        return len(self._data)

    def update(self, items: dict) -> None:
        self._data.update(items)

    def keys(self) -> list:
        return list(self._data.keys())

    def values(self) -> list:
        return list(self._data.values())

    def items(self) -> list[tuple]:
        return list(self._data.items())

    def clear(self) -> None:
        self._data.clear()

    def pop(self, key: str, default=None):
        return self._data.pop(key, default)


@ray.remote
class ListActor:
    """Distributed list backed by Ray Actor."""

    def __init__(self) -> None:
        self._data: list = []

    def get(self, index: int):
        return self._data[index]

    def append(self, value) -> None:
        self._data.append(value)

    def extend(self, values: list) -> None:
        self._data.extend(values)

    def pop(self, index: int = -1):
        return self._data.pop(index)

    def length(self) -> int:
        return len(self._data)

    def slice(self, start: int, end: int) -> list:
        return self._data[start:end]

    def clear(self) -> None:
        self._data.clear()


@ray.remote
class SetActor:
    """Distributed set backed by Ray Actor."""

    def __init__(self) -> None:
        self._data: set = set()

    def add(self, value) -> None:
        self._data.add(value)

    def discard(self, value) -> None:
        self._data.discard(value)

    def contains(self, value) -> bool:
        return value in self._data

    def length(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


@ray.remote
class QueueActor:
    """Distributed FIFO queue backed by Ray Actor."""

    def __init__(self) -> None:
        self._data: list = []

    def put(self, value) -> None:
        self._data.append(value)

    def get_nowait(self):
        if self._data:
            return self._data.pop(0)
        return None

    def length(self) -> int:
        return len(self._data)

    def empty(self) -> bool:
        return len(self._data) == 0


@ray.remote
class BarrierActor:
    """Distributed barrier backed by Ray Actor."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._count = 0
        self._generation = 0

    def arrive(self) -> tuple[int, int]:
        """Arrive at barrier, return (count, generation)."""
        self._count += 1
        return self._count, self._generation

    def get_state(self) -> tuple[int, int, int]:
        """Get (n, count, generation)."""
        return self._n, self._count, self._generation

    def reset(self) -> None:
        """Reset barrier for reuse."""
        self._count = 0
        self._generation += 1


@ray.remote
class LockActor:
    """Distributed lock backed by Ray Actor."""

    def __init__(self) -> None:
        self._holder: str | None = None

    def acquire(self, holder_id: str) -> bool:
        """Try to acquire lock. Returns True if acquired."""
        if self._holder is None:
            self._holder = holder_id
            return True
        return self._holder == holder_id  # Reentrant

    def release(self, holder_id: str) -> bool:
        """Release lock. Returns True if released."""
        if self._holder == holder_id:
            self._holder = None
            return True
        return False

    def is_locked(self) -> bool:
        return self._holder is not None


__all__ = [
    "CounterActor",
    "DictActor",
    "ListActor",
    "SetActor",
    "QueueActor",
    "BarrierActor",
    "LockActor",
]
