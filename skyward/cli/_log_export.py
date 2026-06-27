"""Serialize recorded execution history to JSONL / Markdown / ipynb.

Pure functions over ``list[{"type", "fields"}]`` event dicts (plus the
``{eid: source}`` map for ipynb). Notebook output uses stdlib ``json``
against the nbformat-4 schema — no ``nbformat`` dependency.
"""

from __future__ import annotations

import json
from collections import defaultdict


def to_jsonl(events: list[dict]) -> str:
    """One ``json.dumps`` per event, newline-separated."""
    if not events:
        return ""
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _group_logs(events: list[dict]) -> tuple[list[str], dict[str, dict[int, list[str]]], list[dict]]:
    """Split events into task-ordered log groups + leftover session events.

    Returns ``(task_order, logs, session_events)`` where ``logs[tid][node_id]``
    is the list of stdout lines for that task on that node.
    """
    task_order: list[str] = []
    logs: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    session: list[dict] = []
    for e in events:
        fields = e.get("fields", {})
        tid = fields.get("task_id")
        etype = e.get("type")
        if etype == "Log.Emitted" and tid:
            if tid not in logs:
                task_order.append(tid)
            logs[tid][fields.get("node_id", 0)].append(fields.get("message", ""))
        elif etype == "Log.Emitted" or not (isinstance(etype, str) and etype.startswith("Task.")):
            session.append(e)
    return task_order, logs, session


def to_markdown(events: list[dict]) -> str:
    """Group ``Log.Emitted`` by task (then node), with session events last."""
    task_order, logs, session = _group_logs(events)
    out: list[str] = ["# Execution log", ""]
    for tid in task_order:
        out.append(f"## Task `{tid}`")
        out.append("")
        by_node = logs[tid]
        for node_id in sorted(by_node):
            if len(by_node) > 1:
                out.append(f"### node {node_id}")
            out.append("```")
            out.extend(by_node[node_id])
            out.append("```")
            out.append("")
    if session:
        out.append("## Session events")
        out.append("")
        for e in session:
            out.append(f"- **{e.get('type')}** {json.dumps(e.get('fields', {}))}")
    return "\n".join(out) + "\n"


def _markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(source: str, stdout: str) -> dict:
    outputs = (
        [{"output_type": "stream", "name": "stdout", "text": stdout.splitlines(keepends=True)}]
        if stdout
        else []
    )
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True),
    }


def to_ipynb(events: list[dict], sources: dict[str, str]) -> str:
    """nbformat-4 notebook: one markdown + code cell per task.

    Each code cell carries the captured script source (joined by task id);
    executions without captured source fall back to an empty-source cell.
    """
    task_order, logs, _session = _group_logs(events)
    cells: list[dict] = []
    for tid in task_order:
        cells.append(_markdown_cell(f"## Task `{tid}`"))
        by_node = logs[tid]
        stdout = "\n".join(
            line for node_id in sorted(by_node) for line in by_node[node_id]
        )
        cells.append(_code_cell(sources.get(tid, ""), stdout))
    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return json.dumps(notebook, indent=1)
