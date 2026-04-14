"""Unit tests for the wire codec."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import pytest
from pydantic import BaseModel

from skyward.server.wire.codec import from_dict, register, to_dict


@pytest.mark.parametrize(
    "value",
    [1, -3, 0, 3.14, -0.5, "hello", "", True, False, None],
)
def test_primitives_roundtrip(value: object) -> None:
    encoded = to_dict(value)
    assert encoded == value
    assert from_dict(encoded) == value


def test_bytes_roundtrip() -> None:
    raw = b"\x00\x01\x02abc\xff"
    encoded = to_dict(raw)
    assert encoded == {"$b64": base64.b64encode(raw).decode("ascii")}
    assert from_dict(encoded) == raw


def test_datetime_roundtrip() -> None:
    dt = datetime(2026, 4, 14, 12, 30, 45, tzinfo=UTC)
    encoded = to_dict(dt)
    assert encoded == {"$dt": dt.isoformat()}
    assert from_dict(encoded) == dt


def test_timedelta_roundtrip() -> None:
    td = timedelta(seconds=90, microseconds=500)
    encoded = to_dict(td)
    assert encoded == {"$td": td.total_seconds()}
    assert from_dict(encoded) == td


class Color(Enum):
    RED = "red"
    BLUE = "blue"


def test_enum_roundtrip() -> None:
    encoded = to_dict(Color.RED)
    assert encoded["name"] == "RED"
    assert encoded["$enum"].endswith("Color")
    assert from_dict(encoded) is Color.RED


def test_list_roundtrip() -> None:
    value = [1, "a", True, None]
    encoded = to_dict(value)
    assert encoded == [1, "a", True, None]
    assert from_dict(encoded) == value


def test_tuple_roundtrip() -> None:
    value = (1, "a", 2.5)
    encoded = to_dict(value)
    assert encoded == {"$tuple": [1, "a", 2.5]}
    assert from_dict(encoded) == value


def test_set_roundtrip() -> None:
    value = {1, 2, 3}
    encoded = to_dict(value)
    assert isinstance(encoded, dict) and "$set" in encoded
    assert from_dict(encoded) == value


def test_frozenset_roundtrip() -> None:
    value = frozenset({"a", "b"})
    encoded = to_dict(value)
    assert isinstance(encoded, dict) and "$frozenset" in encoded
    assert from_dict(encoded) == value


def test_dict_roundtrip() -> None:
    value = {"a": 1, "b": [1, 2, 3]}
    encoded = to_dict(value)
    assert encoded == {"a": 1, "b": [1, 2, 3]}
    assert from_dict(encoded) == value


def test_mapping_proxy_roundtrip() -> None:
    value = MappingProxyType({"x": 1})
    encoded = to_dict(value)
    assert from_dict(encoded) == dict(value)


@dataclass(frozen=True, slots=True)
class Inner:
    n: int
    tag: str


@dataclass(frozen=True, slots=True)
class Outer:
    inner: Inner
    count: int


def test_frozen_dataclass_nested_roundtrip() -> None:
    value = Outer(inner=Inner(n=7, tag="hi"), count=3)
    encoded = to_dict(value)
    assert encoded == {"inner": {"n": 7, "tag": "hi"}, "count": 3}
    assert from_dict(encoded, Outer) == value


@runtime_checkable
class Base(Protocol):
    """Example discriminated-union base."""

    ...


@dataclass(frozen=True, slots=True)
class A:
    x: int


@dataclass(frozen=True, slots=True)
class B:
    y: str


register(Base, a=A, b=B)


def test_discriminated_union_roundtrip() -> None:
    value = A(x=42)
    encoded = to_dict(value)
    assert encoded == {"type": "a", "x": 42}
    decoded = from_dict(encoded, Base)
    assert decoded == value


def test_discriminated_union_second_variant() -> None:
    value = B(y="hello")
    encoded = to_dict(value)
    assert encoded == {"type": "b", "y": "hello"}
    assert from_dict(encoded, Base) == value


class PydModel(BaseModel):
    name: str
    count: int


def test_pydantic_roundtrip() -> None:
    value = PydModel(name="foo", count=3)
    encoded = to_dict(value)
    assert encoded["name"] == "foo"
    assert encoded["count"] == 3
    assert from_dict(encoded, PydModel) == value


@dataclass(frozen=True, slots=True)
class WithCollections:
    tags: tuple[str, ...]
    data: bytes


def test_dataclass_with_collections_and_bytes() -> None:
    value = WithCollections(tags=("a", "b"), data=b"xyz")
    encoded = to_dict(value)
    decoded = from_dict(encoded, WithCollections)
    assert decoded == value
