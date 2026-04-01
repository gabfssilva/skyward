"""Shared output helpers for the Skyward CLI."""

from __future__ import annotations

import json as _json
import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

# ── Visual constants ────────────────────────────────────────────

SUCCESS = "[green]✓[/green]"
ACTIVE = "[green]●[/green]"
PROGRESS = "[yellow]●[/yellow]"
INACTIVE = "[dim]◌[/dim]"
ERROR = "[red]✗[/red]"
BRANCH = "[dim]├─[/dim]"
BRANCH_LAST = "[dim]└─[/dim]"

PHASE_INDICATOR: dict[str, str] = {
    "READY": ACTIVE,
    "PROVISIONING": PROGRESS,
    "SSH": PROGRESS,
    "BOOTSTRAP": PROGRESS,
    "WORKERS": PROGRESS,
    "STOPPED": INACTIVE,
}


def phase_label(phase: str) -> str:
    """Return indicator + lowercase phase name."""
    indicator = PHASE_INDICATOR.get(phase, INACTIVE)
    return f"{indicator} {phase.lower()}"


def print_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    as_json: bool = False,
) -> None:
    if as_json:
        data = [dict(zip(columns, row, strict=True)) for row in rows]
        sys.stdout.write(_json.dumps(data, indent=2, default=str) + "\n")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*(str(v) for v in row))
    console.print(table)


def print_status(label: str, status: str, detail: str = "", *, as_json: bool = False) -> None:
    if as_json:
        sys.stdout.write(_json.dumps({"label": label, "status": status, "detail": detail}) + "\n")
        return
    icon = {"ok": "[green]ok[/green]", "fail": "[red]fail[/red]", "-": "[dim]-[/dim]"}.get(status, status)
    detail_str = f"  {detail}" if detail else ""
    console.print(f" {icon}   {label}{detail_str}")


def format_price(price: float | None, unit: str = "hr") -> str:
    if price is None:
        return "-"
    return f"${price:.2f}/{unit}"
