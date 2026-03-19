"""Shared output helpers for the Skyward CLI."""

from __future__ import annotations

import json as _json
import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def print_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    as_json: bool = False,
) -> None:
    """Render tabular data as a Rich table or JSON array.

    Parameters
    ----------
    columns
        Column header names.
    rows
        Row data — each row is a sequence of values aligned with *columns*.
    as_json
        When ``True``, emit a JSON array of dicts to stdout instead of a table.
    """
    if as_json:
        data = [dict(zip(columns, row, strict=True)) for row in rows]
        sys.stdout.write(_json.dumps(data, indent=2, default=str) + "\n")
        return

    table = Table(show_header=True, header_style="bold", expand=True)
    for col in columns:
        table.add_column(col, ratio=2 if col == "Details" else None)
    for row in rows:
        table.add_row(*(str(v) for v in row))
    console.print(table)


_STATUS_ICONS = {"ok": "[green]ok[/green]", "fail": "[red]fail[/red]", "-": "[dim]-[/dim]"}


def print_status(label: str, status: str, detail: str = "", *, as_json: bool = False) -> None:
    """Print a single status line with an ok/fail/- indicator.

    Parameters
    ----------
    label
        Left-hand label (e.g. ``"Global config"``).
    status
        One of ``"ok"``, ``"fail"``, or ``"-"``.
    detail
        Optional right-hand detail text.
    as_json
        Emit as a JSON object instead of a Rich-formatted line.
    """
    if as_json:
        sys.stdout.write(_json.dumps({"label": label, "status": status, "detail": detail}) + "\n")
        return
    icon = _STATUS_ICONS.get(status, status)
    detail_str = f"  {detail}" if detail else ""
    console.print(f" {icon}   {label}{detail_str}")


def format_price(price: float | None, unit: str = "hr") -> str:
    """Format a price as ``$1.23/hr`` or ``-`` when absent."""
    if price is None:
        return "-"
    return f"${price:.2f}/{unit}"
