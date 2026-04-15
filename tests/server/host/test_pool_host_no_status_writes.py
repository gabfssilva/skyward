"""Workstream B invariant — PoolHost only writes the initial Queued row.

All post-insert transitions belong to the task_manager actor (§3.2
single-writer). The host keeps ``put_execution(... status=Queued())``
for the initial create but must not call ``update_execution_status``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_HOST = Path(__file__).resolve().parents[3] / "skyward" / "server" / "host" / "pool_host.py"


def _method_body(source: str, method_name: str) -> str:
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == method_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{method_name} not found in pool_host.py")


def test_submit_task_never_writes_terminal_status() -> None:
    body = _method_body(_HOST.read_text(encoding="utf-8"), "submit_task")
    assert "update_execution_status" not in body, body


def test_broadcast_never_writes_terminal_status() -> None:
    body = _method_body(_HOST.read_text(encoding="utf-8"), "broadcast")
    assert "update_execution_status" not in body, body


def test_submit_task_still_creates_queued_row() -> None:
    body = _method_body(_HOST.read_text(encoding="utf-8"), "submit_task")
    assert "put_execution" in body
    assert "Queued()" in body
