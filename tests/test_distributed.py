# pyright: reportOptionalMemberAccess=false, reportUnusedExpression=false
from __future__ import annotations

import pytest

import skyward as sky

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(120),
    pytest.mark.xdist_group("pool"),
    pytest.mark.xfail(
        reason="Distributed collections ContextVar (_active_registry) does not propagate "
        "through Casty actor system to @compute functions. "
        "worker.py sets _active_registry in main(), but Casty spawns actors in separate "
        "contexts, so asyncio.to_thread(_run) in worker_behavior doesn't inherit it.",
        strict=False,
    ),
]


class TestDistributedDict:
    def test_set_and_get_across_nodes(self, pool):
        @sky.compute
        def writer():
            d = sky.dict("cross_dict")
            info = sky.instance_info()
            d[f"node-{info.node}"] = info.node
            return True

        writer() @ pool

        @sky.compute
        def reader():
            d = sky.dict("cross_dict")
            return d.get("node-0"), d.get("node-1")

        result = reader() >> pool
        assert result == (0, 1)

    def test_dict_operations(self, pool):
        @sky.compute
        def dict_ops():
            d = sky.dict("ops_dict")
            d["key"] = "value"
            d["key"] = "updated"
            return d["key"]

        result = dict_ops() >> pool
        assert result == "updated"


class TestDistributedCounter:
    def test_increment_from_multiple_nodes(self, pool):
        @sky.compute
        def increment():
            c = sky.counter("multi_counter")
            c.increment(1)
            return True

        increment() @ pool

        @sky.compute
        def read_counter():
            c = sky.counter("multi_counter")
            return c.value

        value = read_counter() >> pool
        assert value == 2

    def test_counter_operations(self, pool):
        @sky.compute
        def counter_ops():
            c = sky.counter("ops_counter")
            c.increment(10)
            c.decrement(3)
            val = c.value
            c.reset()
            return val, c.value

        result = counter_ops() >> pool
        assert result == (7, 0)


class TestDistributedQueue:
    def test_put_and_get(self, pool):
        @sky.compute
        def queue_ops():
            q = sky.queue("test_q")
            q.put("hello")
            q.put("world")
            first = q.get(timeout=5)
            second = q.get(timeout=5)
            return first, second

        result = queue_ops() >> pool
        assert result == ("hello", "world")

    def test_get_timeout_returns_none(self, pool):
        @sky.compute
        def queue_timeout():
            q = sky.queue("empty_q")
            return q.get(timeout=0.5)

        result = queue_timeout() >> pool
        assert result is None


class TestDistributedLock:
    def test_lock_context_manager(self, pool):
        @sky.compute
        def lock_ops():
            lock = sky.lock("test_lock")
            c = sky.counter("lock_counter")
            with lock:
                c.increment()
            return c.value

        result = lock_ops() >> pool
        assert result == 1


class TestDistributedBarrier:
    def test_barrier_synchronizes_all_nodes(self, pool):
        @sky.compute
        def wait_at_barrier():
            b = sky.barrier("sync_point", 2)
            b.wait()
            return sky.instance_info().node

        results = wait_at_barrier() @ pool
        assert sorted(results) == [0, 1]


class TestDistributedSet:
    def test_set_operations(self, pool):
        @sky.compute
        def set_ops():
            s = sky.set("test_set")
            s.add("a")
            s.add("b")
            s.add("c")
            s.discard("b")
            return len(s), "a" in s, "b" not in s

        result = set_ops() >> pool
        assert result == (2, True, True)
