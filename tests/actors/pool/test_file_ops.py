"""Pool-level node selection and file-op fan-out."""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import MagicMock

import pytest

from skyward.actors.pool.actor import _gather_file_op, _select_node_refs

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _state(ready: set[int], refs: dict[int, object]):
    s = MagicMock()
    s.ready_nodes = frozenset(ready)
    s.node_refs = MappingProxyType(refs)
    return s


def test_select_head():
    refs = {0: "r0", 1: "r1"}
    s = _state({0, 1}, refs)
    assert _select_node_refs(s, "head") == ((0, "r0"),)


def test_select_all_sorted():
    refs = {0: "r0", 1: "r1", 2: "r2"}
    s = _state({0, 2, 1}, refs)
    assert _select_node_refs(s, "all") == ((0, "r0"), (1, "r1"), (2, "r2"))


def test_select_rank_present():
    s = _state({0, 1}, {0: "r0", 1: "r1"})
    assert _select_node_refs(s, 1) == ((1, "r1"),)


def test_select_rank_not_ready_empty():
    s = _state({0}, {0: "r0", 1: "r1"})
    assert _select_node_refs(s, 1) == ()


def test_select_head_not_ready_empty():
    s = _state(set(), {})
    assert _select_node_refs(s, "head") == ()


async def test_gather_file_op_collects_results():
    from skyward.actors.messages import NodeFileResult

    class _System:
        async def ask(self, ref, factory, timeout):  # noqa: ANN001
            return NodeFileResult(node_id=ref, success=True, listing=f"node {ref}")

    targets = ((0, 0), (1, 1))
    results = await _gather_file_op(_System(), targets, "ls", "/x", b"", 5.0)
    assert {r.node_id for r in results} == {0, 1}
    assert all(r.success for r in results)


async def test_gather_file_op_converts_errors():
    class _System:
        async def ask(self, ref, factory, timeout):  # noqa: ANN001
            raise TimeoutError("slow")

    results = await _gather_file_op(_System(), ((3, 3),), "rm", "/x", b"", 1.0)
    assert results[0].node_id == 3
    assert results[0].success is False
    assert "slow" in (results[0].error or "")
