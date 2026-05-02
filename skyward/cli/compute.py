"""sky compute — manage remote pools and run scripts via the HTTP server."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import httpx
from cyclopts import Parameter

from . import compute_app
from ._client import format_http_error, make_client, resolve_server_url
from ._output import console, print_status, print_table

_POOL_COLUMNS = ["Name", "Status", "Nodes", "Concurrency", "Active"]


def _pool_row(info: dict) -> tuple:
    return (
        info["name"],
        info.get("status", "ready"),
        info["current_nodes"],
        info["concurrency"],
        info["is_active"],
    )


@compute_app.command(name="create")
def create_pool(
    pool: Annotated[str | None, Parameter(help="Named pool from skyward.toml")] = None,
    *,
    name: Annotated[str | None, Parameter(name="--name", help="Server-side pool name (defaults to TOML name or auto-generated)")] = None,
    provider: Annotated[str | None, Parameter(name="--provider", help="Provider type (aws, vastai, runpod, …) when not using a TOML pool")] = None,
    region: Annotated[str | None, Parameter(name="--region", help="Override region on the resolved spec")] = None,
    nodes: Annotated[int | None, Parameter(name="--nodes", help="Override node count")] = None,
    accelerator: Annotated[str | None, Parameter(name="--accelerator", help="Override accelerator (e.g. A100, H100)")] = None,
    allocation: Annotated[
        str | None,
        Parameter(name="--allocation", help="Override allocation (spot, on-demand, spot-if-available, cheapest)"),
    ] = None,
    pip: Annotated[
        list[str] | None,
        Parameter(name="--pip", help="Extra pip packages to install (repeatable; appends to TOML image.pip)"),
    ] = None,
    apt: Annotated[
        list[str] | None,
        Parameter(name="--apt", help="Extra apt packages to install (repeatable; appends to TOML image.apt)"),
    ] = None,
    python: Annotated[
        str | None,
        Parameter(name="--python", help="Python version on the worker (e.g. 3.12, 3.13). Overrides TOML"),
    ] = None,
    watch: Annotated[
        bool,
        Parameter(name="--watch", help="After creating, follow pool events live until ready/failed"),
    ] = False,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Create a remote compute pool."""
    from skyward.api.spec import Options, Spec
    from skyward.config import _get_provider_map, resolve_pool_specs
    from skyward.server.wire import encode

    if pool is None and provider is None:
        console.print(
            "[red]Missing pool source.[/red] Pass a [bold]POOL[/bold] name from "
            "skyward.toml, or [bold]--provider[/bold] to build a spec inline.\n\n"
            "[bold]Examples[/bold]\n"
            "  sky compute create demo                  "
            r"[dim]# uses \[pools.demo] from skyward.toml[/dim]"
            "\n"
            "  sky compute create --provider runpod     [dim]# inline, single spec[/dim]\n"
            "  sky compute create demo --nodes 4        [dim]# TOML + inline override[/dim]\n\n"
            "Run [bold]sky compute create --help[/bold] for the full flag list."
        )
        raise SystemExit(2)

    if pool is not None:
        try:
            specs, options = resolve_pool_specs(pool)
        except (KeyError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise SystemExit(1) from None
        server_name = name or pool
    else:
        provider_map = _get_provider_map()
        cls = provider_map.get(provider or "")
        if cls is None:
            console.print(
                f"[red]Unknown provider '{provider}'. "
                f"Valid: {', '.join(sorted(provider_map))}[/red]"
            )
            raise SystemExit(2)
        specs = (Spec(provider=cls()),)
        options = Options()
        server_name = name

    spec = specs[0]
    overrides: dict = {}
    if nodes is not None:
        overrides["nodes"] = nodes
    if allocation is not None:
        overrides["allocation"] = allocation
    if region is not None:
        overrides["region"] = region
    if accelerator is not None:
        from skyward.accelerators import Accelerator

        overrides["accelerator"] = Accelerator.from_name(accelerator)
    if pip or apt or python is not None:
        base = spec.image
        new_pip = (*base.pip, *(pip or ()))
        new_apt = (*base.apt, *(apt or ()))
        overrides["image"] = replace(
            base,
            pip=new_pip,
            apt=new_apt,
            python=python if python is not None else base.python,
        )
    if overrides:
        if len(specs) > 1:
            console.print("[red]Inline overrides require a single spec; the resolved pool has multiple[/red]")
            raise SystemExit(2)
        spec = replace(spec, **overrides)
        specs = (spec,)

    body = encode((specs, options))
    params: dict[str, str] = {"name": server_name} if server_name else {}

    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.post(
                "/compute",
                params=params,
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None

    if r.status_code not in (201, 202):
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    info = r.json()
    if json:
        import json as _json

        sys.stdout.write(_json.dumps(info) + "\n")
        return

    if watch:
        import asyncio

        from ._view import render as _render

        exit_code = asyncio.run(_render(target, info["name"], json_mode=False, once=False))
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    print_table(_POOL_COLUMNS, [_pool_row(info)])
    if info.get("status") == "creating":
        console.print(
            f"[dim]Provisioning in background.[/dim] "
            f"Follow with: [bold]sky compute view {info['name']}[/bold]"
        )


@compute_app.command(name="list")
def list_pools(
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """List all pools registered on the server."""
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.get("/compute")
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None

    if r.status_code != 200:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    pools = r.json()
    if json:
        import json as _json

        sys.stdout.write(_json.dumps(pools) + "\n")
        return
    if not pools:
        console.print("[dim]No pools[/dim]")
        return
    print_table(_POOL_COLUMNS, [_pool_row(p) for p in pools])


@compute_app.command(name="get")
def get_pool(
    name: Annotated[str, Parameter(help="Pool name on the server")],
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Show info for a single remote pool."""
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.get(f"/compute/{name}")
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None

    if r.status_code == 404:
        console.print(f"[red]Pool {name!r} not found[/red]")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    info = r.json()
    if json:
        import json as _json

        sys.stdout.write(_json.dumps(info) + "\n")
        return
    print_table(_POOL_COLUMNS, [_pool_row(info)])


@compute_app.command(name="delete")
def delete_pool(
    name: Annotated[str, Parameter(help="Pool name on the server")],
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Tear down a remote pool."""
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.delete(f"/compute/{name}")
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None

    if r.status_code == 404:
        console.print(f"[red]Pool {name!r} not found[/red]")
        raise SystemExit(1)
    if r.status_code != 204:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)
    print_status(f"Pool '{name}'", "ok", "deleted")


@compute_app.command(name="view")
def view_pool(
    name: Annotated[str, Parameter(help="Pool name on the server")],
    *,
    once: Annotated[bool, Parameter(name="--once", help="Print one snapshot and exit")] = False,
    json: Annotated[bool, Parameter(name="--json", help="Emit NDJSON (snapshot + each event)")] = False,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Follow live pool/node events from the server.

    Connects to ``GET /compute/{name}/events`` (Server-Sent Events). The
    default mode renders a Rich live layout; use ``--once`` for a static
    snapshot or ``--json`` for NDJSON suitable for piping to ``jq``.
    """
    import asyncio

    from ._view import render as _render

    target = resolve_server_url(url)
    try:
        exit_code = asyncio.run(_render(target, name, json_mode=json, once=once))
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None
    if exit_code != 0:
        raise SystemExit(exit_code)


@compute_app.command(name="run")
def run_script(
    name: Annotated[str, Parameter(help="Pool name on the server")],
    script: Annotated[Path, Parameter(help="Path to the .py script to execute")],
    args: Annotated[list[str] | None, Parameter(help="Arguments forwarded as sys.argv to the script")] = None,
    *,
    broadcast: Annotated[bool, Parameter(name="--broadcast", help="Run on every node instead of one (round-robin)")] = False,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Run a local Python script remotely on a pool.

    The script source is sent to the server, exec'd inside a worker, and the
    captured stdout/stderr are streamed back at the end.
    """
    import skyward as sky

    from ._script import build_exec_pending

    if not script.is_file():
        console.print(f"[red]Script not found: {script}[/red]")
        raise SystemExit(2)

    pending = build_exec_pending(script, list(args or []))
    target = resolve_server_url(url)

    try:
        with sky.Client(name, url=target) as client:
            results = client.broadcast(pending) if broadcast else [client.run(pending)]
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from None

    exit_code = 0
    for idx, result in enumerate(results):
        if broadcast and len(results) > 1:
            console.print(f"[dim]── node {idx} ──[/dim]")
        sys.stdout.write(result["stdout"])
        sys.stderr.write(result["stderr"])
        if result["exit"] != 0:
            exit_code = result["exit"]
    if exit_code != 0:
        raise SystemExit(exit_code)
