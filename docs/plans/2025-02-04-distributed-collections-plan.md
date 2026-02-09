# Distributed Collections Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `sky.dict()`, `sky.list()`, `sky.set()`, `sky.counter()`, `sky.queue()`, `sky.barrier()`, `sky.lock()` as distributed data structures using Ray Actors.

**Architecture:** Each structure is a Ray Actor with a proxy wrapper. Proxies provide sync/async interface with magic methods. Structures are get-or-created by name and live while the pool exists.

**Tech Stack:** Ray Actors, asyncio, Python 3.12+ generics

---

## Task 1: Types Module

**Files:**
- Create: `skyward/distributed/types.py`
- Test: `tests/distributed/test_types.py`

**Step 1: Write the test**

```python
# tests/distributed/test_types.py
from skyward.distributed.types import Consistency


def test_consistency_literal():
    """Consistency type accepts valid values."""
    strong: Consistency = "strong"
    eventual: Consistency = "eventual"
    assert strong == "strong"
    assert eventual == "eventual"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_types.py -v`
Expected: FAIL (module not found)

**Step 3: Write the implementation**

```python
# skyward/distributed/types.py
"""Type definitions for distributed collections."""

from __future__ import annotations

from typing import Literal

type Consistency = Literal["strong", "eventual"]

__all__ = ["Consistency"]
```

**Step 4: Create `__init__.py` for tests**

```python
# tests/__init__.py
# (empty file)
```

```python
# tests/distributed/__init__.py
# (empty file)
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/distributed/test_types.py -v`
Expected: PASS

---

## Task 2: Counter Actor and Proxy

**Files:**
- Create: `skyward/distributed/actors.py`
- Create: `skyward/distributed/proxies.py`
- Test: `tests/distributed/test_counter.py`

**Step 1: Write the test**

```python
# tests/distributed/test_counter.py
import pytest
import ray

from skyward.distributed.actors import CounterActor
from skyward.distributed.proxies import CounterProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    """Initialize Ray for tests."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_counter_increment():
    """Counter increments correctly."""
    actor = CounterActor.options(name="test:counter:inc").remote()
    proxy = CounterProxy(actor, consistency="strong")

    assert proxy.value == 0
    proxy.increment()
    assert proxy.value == 1
    proxy.increment(5)
    assert proxy.value == 6

    ray.kill(actor)


def test_counter_decrement():
    """Counter decrements correctly."""
    actor = CounterActor.options(name="test:counter:dec").remote()
    proxy = CounterProxy(actor, consistency="strong")

    proxy.increment(10)
    proxy.decrement()
    assert proxy.value == 9
    proxy.decrement(4)
    assert proxy.value == 5

    ray.kill(actor)


def test_counter_reset():
    """Counter resets correctly."""
    actor = CounterActor.options(name="test:counter:reset").remote()
    proxy = CounterProxy(actor, consistency="strong")

    proxy.increment(100)
    proxy.reset()
    assert proxy.value == 0
    proxy.reset(50)
    assert proxy.value == 50

    ray.kill(actor)


def test_counter_int_conversion():
    """Counter converts to int."""
    actor = CounterActor.options(name="test:counter:int").remote()
    proxy = CounterProxy(actor, consistency="strong")

    proxy.increment(42)
    assert int(proxy) == 42

    ray.kill(actor)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_counter.py -v`
Expected: FAIL (module not found)

**Step 3: Write the Actor**

```python
# skyward/distributed/actors.py
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


__all__ = ["CounterActor"]
```

**Step 4: Write the Proxy**

```python
# skyward/distributed/proxies.py
"""Proxy wrappers for distributed collections."""

from __future__ import annotations

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


__all__ = ["CounterProxy"]
```

**Step 5: Create distributed `__init__.py`**

```python
# skyward/distributed/__init__.py
"""Distributed collections for Skyward."""

from .types import Consistency
from .proxies import CounterProxy

__all__ = ["Consistency", "CounterProxy"]
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/distributed/test_counter.py -v`
Expected: PASS

---

## Task 3: Dict Actor and Proxy

**Files:**
- Modify: `skyward/distributed/actors.py`
- Modify: `skyward/distributed/proxies.py`
- Test: `tests/distributed/test_dict.py`

**Step 1: Write the test**

```python
# tests/distributed/test_dict.py
import pytest
import ray

from skyward.distributed.actors import DictActor
from skyward.distributed.proxies import DictProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_dict_setitem_getitem():
    """Dict set and get items."""
    actor = DictActor.options(name="test:dict:basic").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy["key1"] = "value1"
    assert proxy["key1"] == "value1"

    proxy["key2"] = 42
    assert proxy["key2"] == 42

    ray.kill(actor)


def test_dict_contains():
    """Dict membership test."""
    actor = DictActor.options(name="test:dict:contains").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy["exists"] = True
    assert "exists" in proxy
    assert "missing" not in proxy

    ray.kill(actor)


def test_dict_len():
    """Dict length."""
    actor = DictActor.options(name="test:dict:len").remote()
    proxy = DictProxy(actor, consistency="strong")

    assert len(proxy) == 0
    proxy["a"] = 1
    proxy["b"] = 2
    assert len(proxy) == 2

    ray.kill(actor)


def test_dict_delete():
    """Dict delete item."""
    actor = DictActor.options(name="test:dict:del").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy["key"] = "value"
    del proxy["key"]
    assert "key" not in proxy

    ray.kill(actor)


def test_dict_get_default():
    """Dict get with default."""
    actor = DictActor.options(name="test:dict:get").remote()
    proxy = DictProxy(actor, consistency="strong")

    assert proxy.get("missing") is None
    assert proxy.get("missing", "default") == "default"
    proxy["key"] = "value"
    assert proxy.get("key") == "value"

    ray.kill(actor)


def test_dict_update():
    """Dict update multiple keys."""
    actor = DictActor.options(name="test:dict:update").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy.update({"a": 1, "b": 2, "c": 3})
    assert proxy["a"] == 1
    assert proxy["b"] == 2
    assert proxy["c"] == 3

    ray.kill(actor)


def test_dict_keys_values_items():
    """Dict keys, values, items."""
    actor = DictActor.options(name="test:dict:kvi").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy.update({"x": 10, "y": 20})

    assert set(proxy.keys()) == {"x", "y"}
    assert set(proxy.values()) == {10, 20}
    assert set(proxy.items()) == {("x", 10), ("y", 20)}

    ray.kill(actor)


def test_dict_clear():
    """Dict clear all items."""
    actor = DictActor.options(name="test:dict:clear").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy.update({"a": 1, "b": 2})
    proxy.clear()
    assert len(proxy) == 0

    ray.kill(actor)


def test_dict_pop():
    """Dict pop item."""
    actor = DictActor.options(name="test:dict:pop").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy["key"] = "value"
    result = proxy.pop("key")
    assert result == "value"
    assert "key" not in proxy
    assert proxy.pop("missing", "default") == "default"

    ray.kill(actor)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_dict.py -v`
Expected: FAIL (DictActor not found)

**Step 3: Add DictActor**

```python
# Add to skyward/distributed/actors.py

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
```

**Step 4: Add DictProxy**

```python
# Add to skyward/distributed/proxies.py

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
```

**Step 5: Update exports**

```python
# skyward/distributed/actors.py - update __all__
__all__ = ["CounterActor", "DictActor"]

# skyward/distributed/proxies.py - update __all__
__all__ = ["CounterProxy", "DictProxy"]

# skyward/distributed/__init__.py - update exports
from .proxies import CounterProxy, DictProxy
__all__ = ["Consistency", "CounterProxy", "DictProxy"]
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/distributed/test_dict.py -v`
Expected: PASS

---

## Task 4: List Actor and Proxy

**Files:**
- Modify: `skyward/distributed/actors.py`
- Modify: `skyward/distributed/proxies.py`
- Test: `tests/distributed/test_list.py`

**Step 1: Write the test**

```python
# tests/distributed/test_list.py
import pytest
import ray

from skyward.distributed.actors import ListActor
from skyward.distributed.proxies import ListProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_list_append_getitem():
    """List append and get."""
    actor = ListActor.options(name="test:list:basic").remote()
    proxy = ListProxy(actor, consistency="strong")

    proxy.append("first")
    proxy.append("second")
    assert proxy[0] == "first"
    assert proxy[1] == "second"

    ray.kill(actor)


def test_list_len():
    """List length."""
    actor = ListActor.options(name="test:list:len").remote()
    proxy = ListProxy(actor, consistency="strong")

    assert len(proxy) == 0
    proxy.append(1)
    proxy.append(2)
    assert len(proxy) == 2

    ray.kill(actor)


def test_list_extend():
    """List extend."""
    actor = ListActor.options(name="test:list:extend").remote()
    proxy = ListProxy(actor, consistency="strong")

    proxy.extend([1, 2, 3])
    assert len(proxy) == 3
    assert proxy[0] == 1
    assert proxy[2] == 3

    ray.kill(actor)


def test_list_pop():
    """List pop."""
    actor = ListActor.options(name="test:list:pop").remote()
    proxy = ListProxy(actor, consistency="strong")

    proxy.extend([1, 2, 3])
    assert proxy.pop() == 3
    assert len(proxy) == 2
    assert proxy.pop(0) == 1
    assert len(proxy) == 1

    ray.kill(actor)


def test_list_slice():
    """List slice."""
    actor = ListActor.options(name="test:list:slice").remote()
    proxy = ListProxy(actor, consistency="strong")

    proxy.extend([0, 1, 2, 3, 4, 5])
    result = proxy.slice(1, 4)
    assert result == [1, 2, 3]

    ray.kill(actor)


def test_list_clear():
    """List clear."""
    actor = ListActor.options(name="test:list:clear").remote()
    proxy = ListProxy(actor, consistency="strong")

    proxy.extend([1, 2, 3])
    proxy.clear()
    assert len(proxy) == 0

    ray.kill(actor)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_list.py -v`
Expected: FAIL

**Step 3: Add ListActor**

```python
# Add to skyward/distributed/actors.py

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
```

**Step 4: Add ListProxy**

```python
# Add to skyward/distributed/proxies.py

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
```

**Step 5: Update exports and run test**

Run: `uv run pytest tests/distributed/test_list.py -v`
Expected: PASS

---

## Task 5: Set Actor and Proxy

**Files:**
- Modify: `skyward/distributed/actors.py`
- Modify: `skyward/distributed/proxies.py`
- Test: `tests/distributed/test_set.py`

**Step 1: Write the test**

```python
# tests/distributed/test_set.py
import pytest
import ray

from skyward.distributed.actors import SetActor
from skyward.distributed.proxies import SetProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_set_add_contains():
    """Set add and contains."""
    actor = SetActor.options(name="test:set:basic").remote()
    proxy = SetProxy(actor, consistency="strong")

    proxy.add("item1")
    assert "item1" in proxy
    assert "item2" not in proxy

    ray.kill(actor)


def test_set_len():
    """Set length."""
    actor = SetActor.options(name="test:set:len").remote()
    proxy = SetProxy(actor, consistency="strong")

    assert len(proxy) == 0
    proxy.add("a")
    proxy.add("b")
    proxy.add("a")  # duplicate
    assert len(proxy) == 2

    ray.kill(actor)


def test_set_discard():
    """Set discard."""
    actor = SetActor.options(name="test:set:discard").remote()
    proxy = SetProxy(actor, consistency="strong")

    proxy.add("item")
    proxy.discard("item")
    assert "item" not in proxy
    proxy.discard("missing")  # should not raise

    ray.kill(actor)


def test_set_clear():
    """Set clear."""
    actor = SetActor.options(name="test:set:clear").remote()
    proxy = SetProxy(actor, consistency="strong")

    proxy.add("a")
    proxy.add("b")
    proxy.clear()
    assert len(proxy) == 0

    ray.kill(actor)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_set.py -v`
Expected: FAIL

**Step 3: Add SetActor**

```python
# Add to skyward/distributed/actors.py

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
```

**Step 4: Add SetProxy**

```python
# Add to skyward/distributed/proxies.py

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
```

**Step 5: Update exports and run test**

Run: `uv run pytest tests/distributed/test_set.py -v`
Expected: PASS

---

## Task 6: Queue Actor and Proxy

**Files:**
- Modify: `skyward/distributed/actors.py`
- Modify: `skyward/distributed/proxies.py`
- Test: `tests/distributed/test_queue.py`

**Step 1: Write the test**

```python
# tests/distributed/test_queue.py
import pytest
import ray

from skyward.distributed.actors import QueueActor
from skyward.distributed.proxies import QueueProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_queue_put_get():
    """Queue put and get."""
    actor = QueueActor.options(name="test:queue:basic").remote()
    proxy = QueueProxy(actor)

    proxy.put("first")
    proxy.put("second")
    assert proxy.get() == "first"
    assert proxy.get() == "second"

    ray.kill(actor)


def test_queue_len():
    """Queue length."""
    actor = QueueActor.options(name="test:queue:len").remote()
    proxy = QueueProxy(actor)

    assert len(proxy) == 0
    proxy.put(1)
    proxy.put(2)
    assert len(proxy) == 2
    proxy.get()
    assert len(proxy) == 1

    ray.kill(actor)


def test_queue_empty():
    """Queue empty check."""
    actor = QueueActor.options(name="test:queue:empty").remote()
    proxy = QueueProxy(actor)

    assert proxy.empty()
    proxy.put("item")
    assert not proxy.empty()

    ray.kill(actor)


def test_queue_get_timeout():
    """Queue get with timeout returns None when empty."""
    actor = QueueActor.options(name="test:queue:timeout").remote()
    proxy = QueueProxy(actor)

    result = proxy.get(timeout=0.1)
    assert result is None

    ray.kill(actor)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_queue.py -v`
Expected: FAIL

**Step 3: Add QueueActor**

```python
# Add to skyward/distributed/actors.py

import asyncio


@ray.remote
class QueueActor:
    """Distributed FIFO queue backed by Ray Actor."""

    def __init__(self) -> None:
        self._data: list = []
        self._waiters: list[asyncio.Event] = []

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
```

**Step 4: Add QueueProxy**

```python
# Add to skyward/distributed/proxies.py

import time


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
```

**Step 5: Update exports and run test**

Run: `uv run pytest tests/distributed/test_queue.py -v`
Expected: PASS

---

## Task 7: Barrier Actor and Proxy

**Files:**
- Modify: `skyward/distributed/actors.py`
- Modify: `skyward/distributed/proxies.py`
- Test: `tests/distributed/test_barrier.py`

**Step 1: Write the test**

```python
# tests/distributed/test_barrier.py
import pytest
import ray
import threading
import time

from skyward.distributed.actors import BarrierActor
from skyward.distributed.proxies import BarrierProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_barrier_wait():
    """Barrier releases when n parties arrive."""
    actor = BarrierActor.options(name="test:barrier:basic").remote(n=2)
    proxy1 = BarrierProxy(actor)
    proxy2 = BarrierProxy(actor)

    results = []

    def wait_and_record(proxy, name):
        proxy.wait()
        results.append(name)

    t1 = threading.Thread(target=wait_and_record, args=(proxy1, "t1"))
    t2 = threading.Thread(target=wait_and_record, args=(proxy2, "t2"))

    t1.start()
    time.sleep(0.1)  # t1 should be waiting
    assert len(results) == 0

    t2.start()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert len(results) == 2

    ray.kill(actor)


def test_barrier_reset():
    """Barrier can be reset and reused."""
    actor = BarrierActor.options(name="test:barrier:reset").remote(n=1)
    proxy = BarrierProxy(actor)

    proxy.wait()  # First use
    proxy.reset()
    proxy.wait()  # Second use after reset

    ray.kill(actor)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_barrier.py -v`
Expected: FAIL

**Step 3: Add BarrierActor**

```python
# Add to skyward/distributed/actors.py

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
```

**Step 4: Add BarrierProxy**

```python
# Add to skyward/distributed/proxies.py

class BarrierProxy:
    """Sync/async proxy for BarrierActor.

    Note: Barrier always uses strong consistency.
    """

    __slots__ = ("_actor",)

    def __init__(self, actor: ActorHandle) -> None:
        self._actor = actor

    def wait(self) -> None:
        """Wait for all parties to arrive at barrier."""
        count, generation = ray.get(self._actor.arrive.remote())
        n, _, _ = ray.get(self._actor.get_state.remote())

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
        count, generation = await self._actor.arrive.remote()
        n, _, _ = await self._actor.get_state.remote()

        while True:
            current_n, current_count, current_gen = await self._actor.get_state.remote()
            if current_gen > generation:
                return
            if current_count >= current_n:
                return
            await asyncio.sleep(0.01)
```

**Step 5: Update exports and run test**

Run: `uv run pytest tests/distributed/test_barrier.py -v`
Expected: PASS

---

## Task 8: Lock Actor and Proxy

**Files:**
- Modify: `skyward/distributed/actors.py`
- Modify: `skyward/distributed/proxies.py`
- Test: `tests/distributed/test_lock.py`

**Step 1: Write the test**

```python
# tests/distributed/test_lock.py
import pytest
import ray
import threading
import time

from skyward.distributed.actors import LockActor
from skyward.distributed.proxies import LockProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_lock_acquire_release():
    """Lock acquire and release."""
    actor = LockActor.options(name="test:lock:basic").remote()
    proxy = LockProxy(actor)

    assert proxy.acquire()
    proxy.release()

    ray.kill(actor)


def test_lock_context_manager():
    """Lock as context manager."""
    actor = LockActor.options(name="test:lock:ctx").remote()
    proxy = LockProxy(actor)

    with proxy:
        pass  # Lock held here

    # Lock released, can acquire again
    assert proxy.acquire()
    proxy.release()

    ray.kill(actor)


def test_lock_mutual_exclusion():
    """Lock provides mutual exclusion."""
    actor = LockActor.options(name="test:lock:mutex").remote()
    proxy = LockProxy(actor)

    results = []

    def critical_section(name):
        with proxy:
            results.append(f"{name}_enter")
            time.sleep(0.1)
            results.append(f"{name}_exit")

    t1 = threading.Thread(target=critical_section, args=("t1",))
    t2 = threading.Thread(target=critical_section, args=("t2",))

    t1.start()
    time.sleep(0.01)  # Let t1 enter first
    t2.start()

    t1.join()
    t2.join()

    # Should be sequential: t1_enter, t1_exit, t2_enter, t2_exit
    # or t2_enter, t2_exit, t1_enter, t1_exit
    assert results[0].endswith("_enter")
    assert results[1].endswith("_exit")
    assert results[0][:2] == results[1][:2]  # Same thread

    ray.kill(actor)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_lock.py -v`
Expected: FAIL

**Step 3: Add LockActor**

```python
# Add to skyward/distributed/actors.py

import uuid


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
```

**Step 4: Add LockProxy**

```python
# Add to skyward/distributed/proxies.py

import uuid


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
```

**Step 5: Update exports and run test**

Run: `uv run pytest tests/distributed/test_lock.py -v`
Expected: PASS

---

## Task 9: Registry (Get-or-Create)

**Files:**
- Create: `skyward/distributed/registry.py`
- Test: `tests/distributed/test_registry.py`

**Step 1: Write the test**

```python
# tests/distributed/test_registry.py
import pytest
import ray

from skyward.distributed.registry import DistributedRegistry
from skyward.distributed.proxies import DictProxy, CounterProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def registry():
    reg = DistributedRegistry()
    yield reg
    reg.cleanup()


def test_registry_get_or_create_dict(registry):
    """Registry creates dict on first access."""
    d1 = registry.dict("test_dict")
    assert isinstance(d1, DictProxy)

    d2 = registry.dict("test_dict")
    assert d1._actor == d2._actor  # Same actor


def test_registry_get_or_create_counter(registry):
    """Registry creates counter on first access."""
    c1 = registry.counter("test_counter")
    assert isinstance(c1, CounterProxy)

    c2 = registry.counter("test_counter")
    assert c1._actor == c2._actor  # Same actor


def test_registry_different_names(registry):
    """Different names create different actors."""
    d1 = registry.dict("dict_a")
    d2 = registry.dict("dict_b")
    assert d1._actor != d2._actor


def test_registry_consistency_override(registry):
    """Registry respects consistency override."""
    d1 = registry.dict("cons_dict", consistency="strong")
    assert d1._consistency == "strong"

    d2 = registry.dict("cons_dict2", consistency="eventual")
    assert d2._consistency == "eventual"


def test_registry_cleanup(registry):
    """Registry cleanup destroys all actors."""
    registry.dict("cleanup_dict")
    registry.counter("cleanup_counter")

    registry.cleanup()

    # After cleanup, actors should be gone
    # Next access creates new actors
    d = registry.dict("cleanup_dict")
    d["key"] = "value"
    assert d["key"] == "value"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_registry.py -v`
Expected: FAIL

**Step 3: Write the registry**

```python
# skyward/distributed/registry.py
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


class DistributedRegistry:
    """Registry for distributed collections.

    Manages get-or-create semantics and cleanup of Ray Actors.
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
        full_name = f"skyward:{actor_cls.__ray_metadata__.class_name.lower()}:{name}"

        if full_name in self._actors:
            return self._actors[full_name]

        try:
            actor = ray.get_actor(full_name)
        except ValueError:
            actor = actor_cls.options(name=full_name).remote(*args, **kwargs)

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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/distributed/test_registry.py -v`
Expected: PASS

---

## Task 10: Public API Functions

**Files:**
- Modify: `skyward/distributed/__init__.py`
- Modify: `skyward/facade.py`
- Modify: `skyward/__init__.py`
- Test: `tests/distributed/test_api.py`

**Step 1: Write the test**

```python
# tests/distributed/test_api.py
import pytest
import ray


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_dict_function():
    """sky.dict() function works."""
    import skyward as sky
    from skyward.distributed import _set_active_registry
    from skyward.distributed.registry import DistributedRegistry

    reg = DistributedRegistry()
    _set_active_registry(reg)

    try:
        d = sky.dict("api_test_dict")
        d["key"] = "value"
        assert d["key"] == "value"
    finally:
        reg.cleanup()
        _set_active_registry(None)


def test_counter_function():
    """sky.counter() function works."""
    import skyward as sky
    from skyward.distributed import _set_active_registry
    from skyward.distributed.registry import DistributedRegistry

    reg = DistributedRegistry()
    _set_active_registry(reg)

    try:
        c = sky.counter("api_test_counter")
        c.increment(5)
        assert c.value == 5
    finally:
        reg.cleanup()
        _set_active_registry(None)


def test_all_functions_exist():
    """All distributed functions are exported."""
    import skyward as sky

    assert hasattr(sky, "dict")
    assert hasattr(sky, "list")
    assert hasattr(sky, "set")
    assert hasattr(sky, "counter")
    assert hasattr(sky, "queue")
    assert hasattr(sky, "barrier")
    assert hasattr(sky, "lock")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_api.py -v`
Expected: FAIL

**Step 3: Update distributed/__init__.py**

```python
# skyward/distributed/__init__.py
"""Distributed collections for Skyward.

Public API:
    sky.dict("name")     - Distributed dict
    sky.list("name")     - Distributed list
    sky.set("name")      - Distributed set
    sky.counter("name")  - Distributed counter
    sky.queue("name")    - Distributed queue
    sky.barrier("name", n=N)  - Synchronization barrier
    sky.lock("name")     - Distributed lock
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

from .types import Consistency
from .proxies import (
    CounterProxy,
    DictProxy,
    ListProxy,
    SetProxy,
    QueueProxy,
    BarrierProxy,
    LockProxy,
)
from .registry import DistributedRegistry

if TYPE_CHECKING:
    pass


# Context variable for active registry
_active_registry: ContextVar[DistributedRegistry | None] = ContextVar(
    "active_registry", default=None
)


def _get_active_registry() -> DistributedRegistry:
    """Get the active registry."""
    reg = _active_registry.get()
    if reg is None:
        raise RuntimeError(
            "No active pool. Use within a @pool decorated function or 'with pool():' block."
        )
    return reg


def _set_active_registry(registry: DistributedRegistry | None) -> None:
    """Set the active registry (for internal use)."""
    _active_registry.set(registry)


# Public API functions


def dict(name: str, *, consistency: Consistency | None = None) -> DictProxy:
    """Get or create a distributed dict.

    Args:
        name: Unique name for the dict.
        consistency: "strong" (wait for writes) or "eventual" (fire-and-forget).

    Returns:
        DictProxy with dict-like interface.

    Example:
        cache = sky.dict("embeddings")
        cache["key"] = value
        v = cache["key"]
    """
    return _get_active_registry().dict(name, consistency=consistency)


def list(name: str, *, consistency: Consistency | None = None) -> ListProxy:
    """Get or create a distributed list.

    Args:
        name: Unique name for the list.
        consistency: "strong" (wait for writes) or "eventual" (fire-and-forget).

    Returns:
        ListProxy with list-like interface.

    Example:
        results = sky.list("outputs")
        results.append(value)
    """
    return _get_active_registry().list(name, consistency=consistency)


def set(name: str, *, consistency: Consistency | None = None) -> SetProxy:
    """Get or create a distributed set.

    Args:
        name: Unique name for the set.
        consistency: "strong" (wait for writes) or "eventual" (fire-and-forget).

    Returns:
        SetProxy with set-like interface.

    Example:
        seen = sky.set("processed")
        seen.add(item)
        if item in seen: ...
    """
    return _get_active_registry().set(name, consistency=consistency)


def counter(name: str, *, consistency: Consistency | None = None) -> CounterProxy:
    """Get or create a distributed counter.

    Args:
        name: Unique name for the counter.
        consistency: "strong" (wait for writes) or "eventual" (fire-and-forget).

    Returns:
        CounterProxy with counter interface.

    Example:
        progress = sky.counter("steps")
        progress.increment()
        print(progress.value)
    """
    return _get_active_registry().counter(name, consistency=consistency)


def queue(name: str) -> QueueProxy:
    """Get or create a distributed queue.

    Note: Queue always uses strong consistency for FIFO semantics.

    Args:
        name: Unique name for the queue.

    Returns:
        QueueProxy with queue interface.

    Example:
        tasks = sky.queue("work")
        tasks.put(item)
        item = tasks.get()
    """
    return _get_active_registry().queue(name)


def barrier(name: str, n: int) -> BarrierProxy:
    """Get or create a distributed barrier.

    Note: Barrier always uses strong consistency.

    Args:
        name: Unique name for the barrier.
        n: Number of parties that must arrive before release.

    Returns:
        BarrierProxy with barrier interface.

    Example:
        sync = sky.barrier("epoch", n=4)
        sync.wait()  # Blocks until 4 arrive
    """
    return _get_active_registry().barrier(name, n)


def lock(name: str) -> LockProxy:
    """Get or create a distributed lock.

    Note: Lock always uses strong consistency.

    Args:
        name: Unique name for the lock.

    Returns:
        LockProxy with lock interface.

    Example:
        lock = sky.lock("critical")
        with lock:
            # Critical section
    """
    return _get_active_registry().lock(name)


__all__ = [
    # Types
    "Consistency",
    # Proxies
    "CounterProxy",
    "DictProxy",
    "ListProxy",
    "SetProxy",
    "QueueProxy",
    "BarrierProxy",
    "LockProxy",
    # Registry
    "DistributedRegistry",
    # Functions
    "dict",
    "list",
    "set",
    "counter",
    "queue",
    "barrier",
    "lock",
    # Internal
    "_get_active_registry",
    "_set_active_registry",
]
```

**Step 4: Update skyward/__init__.py**

Add to imports section:

```python
# Add after other imports
from .distributed import (
    dict,
    list,
    set,
    counter,
    queue,
    barrier,
    lock,
)
```

Add to `__all__`:

```python
# Add to __all__ list
"dict",
"list",
"set",
"counter",
"queue",
"barrier",
"lock",
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/distributed/test_api.py -v`
Expected: PASS

---

## Task 11: Pool Integration

**Files:**
- Modify: `skyward/facade.py`
- Test: `tests/distributed/test_pool_integration.py`

**Step 1: Write the test**

```python
# tests/distributed/test_pool_integration.py
import pytest
import ray


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_pool_creates_registry():
    """SyncComputePool creates and manages registry."""
    from skyward.facade import SyncComputePool
    from skyward.distributed import _get_active_registry
    from unittest.mock import patch, MagicMock

    # We can't fully test pool without cloud infra, but we can test registry setup
    pool = SyncComputePool.__new__(SyncComputePool)
    pool._registry = None

    # Verify registry attribute exists after __init__ pattern
    assert hasattr(pool, "_registry")


def test_pool_has_collection_methods():
    """SyncComputePool has dict, list, etc. methods."""
    from skyward.facade import SyncComputePool

    assert hasattr(SyncComputePool, "dict")
    assert hasattr(SyncComputePool, "list")
    assert hasattr(SyncComputePool, "set")
    assert hasattr(SyncComputePool, "counter")
    assert hasattr(SyncComputePool, "queue")
    assert hasattr(SyncComputePool, "barrier")
    assert hasattr(SyncComputePool, "lock")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/distributed/test_pool_integration.py -v`
Expected: FAIL

**Step 3: Modify facade.py**

Add imports at top:

```python
from .distributed import (
    DistributedRegistry,
    _set_active_registry,
)
from .distributed import (
    DictProxy,
    ListProxy,
    SetProxy,
    CounterProxy,
    QueueProxy,
    BarrierProxy,
    LockProxy,
)
from .distributed.types import Consistency
```

Add to SyncComputePool class:

```python
# Add to field declarations
_registry: DistributedRegistry | None = field(default=None, init=False, repr=False)

# Add methods to SyncComputePool class

def dict(self, name: str, *, consistency: Consistency | None = None) -> DictProxy:
    """Get or create a distributed dict."""
    if self._registry is None:
        raise RuntimeError("Pool is not active")
    return self._registry.dict(name, consistency=consistency)

def list(self, name: str, *, consistency: Consistency | None = None) -> ListProxy:
    """Get or create a distributed list."""
    if self._registry is None:
        raise RuntimeError("Pool is not active")
    return self._registry.list(name, consistency=consistency)

def set(self, name: str, *, consistency: Consistency | None = None) -> SetProxy:
    """Get or create a distributed set."""
    if self._registry is None:
        raise RuntimeError("Pool is not active")
    return self._registry.set(name, consistency=consistency)

def counter(self, name: str, *, consistency: Consistency | None = None) -> CounterProxy:
    """Get or create a distributed counter."""
    if self._registry is None:
        raise RuntimeError("Pool is not active")
    return self._registry.counter(name, consistency=consistency)

def queue(self, name: str) -> QueueProxy:
    """Get or create a distributed queue."""
    if self._registry is None:
        raise RuntimeError("Pool is not active")
    return self._registry.queue(name)

def barrier(self, name: str, n: int) -> BarrierProxy:
    """Get or create a distributed barrier."""
    if self._registry is None:
        raise RuntimeError("Pool is not active")
    return self._registry.barrier(name, n)

def lock(self, name: str) -> LockProxy:
    """Get or create a distributed lock."""
    if self._registry is None:
        raise RuntimeError("Pool is not active")
    return self._registry.lock(name)
```

Modify `__enter__` to create registry:

```python
# In __enter__, after self._active = True:
self._registry = DistributedRegistry()
_set_active_registry(self._registry)
```

Modify `__exit__` to cleanup registry:

```python
# In __exit__, before _cleanup():
if self._registry is not None:
    self._registry.cleanup()
    _set_active_registry(None)
    self._registry = None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/distributed/test_pool_integration.py -v`
Expected: PASS

---

## Task 12: Run All Tests

**Step 1: Run all distributed tests**

Run: `uv run pytest tests/distributed/ -v`
Expected: All PASS

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All PASS

