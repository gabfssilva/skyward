"""What a command prints.

Two renderings of the same rows: aligned columns for a person, JSON for a
program. Commands take ``--output`` and pass it here rather than branching
themselves, so ``--output json`` means the same thing everywhere.

No ``rich``. The live dashboard is an extra and the CLI must run without it;
padded strings are enough for a table.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Literal

type Output = Literal["table", "json"]

EMPTY = "-"


def cell(value: object) -> str:
    """Render one value, with None as a dash rather than the word."""
    return EMPTY if value is None else str(value)


def dump(data: object) -> None:
    """Write a JSON document to stdout."""
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")


def render(columns: Sequence[str], rows: Sequence[Sequence[object]], *, output: Output = "table") -> None:
    """Print rows as an aligned table, or as a list of JSON objects.

    Parameters
    ----------
    columns
        Header names, one per value in every row.
    rows
        The values, in column order.
    output
        ``"table"`` for a person, ``"json"`` for a program.
    """
    match output:
        case "json":
            dump([dict(zip(columns, map(cell, row), strict=True)) for row in rows])
        case "table":
            _print_table(columns, [[cell(value) for value in row] for row in rows])


def _print_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(column), *(len(row[index]) for row in rows)) if rows else len(column) for index, column in enumerate(columns)]
    lines = [columns, *rows]
    sys.stdout.write("".join(f"{'  '.join(value.ljust(width) for value, width in zip(line, widths, strict=True)).rstrip()}\n" for line in lines))


__all__ = ["EMPTY", "Output", "cell", "dump", "render"]
