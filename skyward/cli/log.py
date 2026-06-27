"""sky log — fetch and export a session's recorded execution history."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import httpx
from cyclopts import Parameter

from . import log_app
from ._client import format_http_error, make_client, resolve_server_url
from ._log_export import to_ipynb, to_jsonl, to_markdown
from ._output import console, print_status


def _resolve_format(fmt: str | None, output: Path | None) -> str:
    if fmt:
        return fmt
    if output is not None and (suffix := output.suffix.lstrip(".")):
        return suffix
    return "jsonl"


@log_app.default
def export_log(
    name: Annotated[str, Parameter(help="Session/pool name")],
    *,
    n: Annotated[int | None, Parameter(name=("-n", "--limit"), help="Last N events only")] = None,
    output: Annotated[Path | None, Parameter(name=("-o", "--output"), help="Write to file (format inferred from suffix)")] = None,
    format: Annotated[str | None, Parameter(name="--format", help="jsonl | md | ipynb")] = None,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Fetch and export a session's execution history (jsonl | md | ipynb)."""
    target = resolve_server_url(url)
    params = {"limit": str(n)} if n else {}
    try:
        with make_client(url) as client:
            r = client.get(f"/compute/{name}/log", params=params)
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None

    if r.status_code == 404:
        console.print(f"[red]No history for session {name!r}[/red]")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    payload = r.json()
    events = payload["events"]
    sources = payload.get("sources", {})

    fmt = _resolve_format(format, output)
    match fmt:
        case "jsonl" | "json":
            text = to_jsonl(events)
        case "md" | "markdown":
            text = to_markdown(events)
        case "ipynb":
            text = to_ipynb(events, sources)
        case _:
            console.print(f"[red]Unknown format '{fmt}'. Use jsonl, md, or ipynb.[/red]")
            raise SystemExit(2)

    if output is not None:
        output.write_text(text)
        print_status(str(output), "ok", f"{len(events)} events")
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
