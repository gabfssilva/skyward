from __future__ import annotations

import pytest

from skyward.infra.serialization import deserialize, serialize

pytestmark = [pytest.mark.xdist_group("unit")]


class TestSerialization:
    def test_simple_function(self):
        def add(a, b):
            return a + b

        data = serialize(add)
        fn = deserialize(data)
        assert fn(2, 3) == 5

    def test_closure_with_state(self):
        factor = 10

        def multiply(x):
            return x * factor

        data = serialize(multiply)
        fn = deserialize(data)
        assert fn(5) == 50

    def test_lambda(self):
        fn = lambda x: x**2  # noqa: E731
        data = serialize(fn)
        restored = deserialize(data)
        assert restored(7) == 49

    def test_dict_roundtrip(self):
        obj = {"key": "value", "nested": [1, 2, 3]}
        data = serialize(obj)
        result = deserialize(data)
        assert result == obj

    def test_large_payload_compresses(self):
        large = list(range(100_000))
        compressed = serialize(large, compress=True)
        uncompressed = serialize(large, compress=False)
        assert len(compressed) < len(uncompressed)
        assert deserialize(compressed) == large

    def test_uncompressed_mode(self):
        obj = {"small": "data"}
        data = serialize(obj, compress=False)
        assert deserialize(data) == obj

    def test_dataclass_roundtrip(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Point:
            x: float
            y: float

        point = Point(1.0, 2.0)
        data = serialize(point)
        result = deserialize(data)
        assert result == point
        assert result.x == 1.0
        assert result.y == 2.0
