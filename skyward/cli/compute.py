"""sky compute — manage remote pools and run scripts via the HTTP server."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import httpx
from cyclopts import Parameter

from . import app, compute_app
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


def _create_session(
    *,
    pool: str | None,
    name: str | None,
    provider: str | None,
    region: str | None,
    nodes: int | None,
    accelerator: str | None,
    allocation: str | None,
    pip: list[str] | None,
    apt: list[str] | None,
    python: str | None,
    url: str | None,
) -> dict:
    """Resolve specs, ``POST /compute``, and return the server's pool-info dict.

    Shared by ``sky compute create`` and ``sky new``. Exits the process
    on bad input, an unreachable server, or a non-201/202 response — the
    established CLI error pattern.
    """
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

    return r.json()


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
    info = _create_session(
        pool=pool, name=name, provider=provider, region=region, nodes=nodes,
        accelerator=accelerator, allocation=allocation, pip=pip, apt=apt,
        python=python, url=url,
    )
    _present_pool_created(info, target=resolve_server_url(url), watch=watch, json=json)


def _present_pool_created(info: dict, *, target: str, watch: bool, json: bool) -> None:
    """Render the result of a pool creation (json | watch | table)."""
    if json:
        import json as _json

        sys.stdout.write(_json.dumps(info) + "\n")
        return

    if watch:
        import asyncio

        from ._view import render as _render

        watch_mode = "rich" if sys.stderr.isatty() else "log"
        exit_code = asyncio.run(_render(target, info["name"], mode=watch_mode))
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    print_table(_POOL_COLUMNS, [_pool_row(info)])
    if info.get("status") == "creating":
        console.print(
            f"[dim]Provisioning in background.[/dim] "
            f"Follow with: [bold]sky status {info['name']}[/bold]"
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
    mode: Annotated[
        str,
        Parameter(
            name="--mode",
            help="rich | log | json | once. 'auto' picks rich on a TTY, log otherwise.",
        ),
    ] = "auto",
    once: Annotated[bool, Parameter(name="--once", help="Alias for --mode=once")] = False,
    json: Annotated[bool, Parameter(name="--json", help="Alias for --mode=json")] = False,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Follow live pool/node events from the server.

    Connects to ``GET /compute/{name}/events`` (Server-Sent Events).

    Modes
    -----
    - ``rich`` — full Rich live layout (TTY).
    - ``log``  — plain ``HH:MM:SS  label  message`` lines (grep-friendly).
    - ``json`` — NDJSON (snapshot + each event), suitable for piping to ``jq``.
    - ``once`` — render the snapshot once and exit.
    - ``auto`` (default) — ``rich`` if stderr is a TTY, ``log`` otherwise.
    """
    import asyncio

    from ._view import ViewMode
    from ._view import render as _render

    resolved: ViewMode
    if json:
        resolved = "json"
    elif once:
        resolved = "once"
    elif mode == "auto":
        resolved = "rich" if sys.stderr.isatty() else "log"
    elif mode in ("rich", "log", "json", "once"):
        resolved = mode  # type: ignore[assignment]
    else:
        console.print(
            f"[red]Unknown --mode '{mode}'. Valid: rich, log, json, once, auto.[/red]"
        )
        raise SystemExit(2)

    target = resolve_server_url(url)
    try:
        exit_code = asyncio.run(_render(target, name, mode=resolved))
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None
    if exit_code != 0:
        raise SystemExit(exit_code)


def _run_params(node: str | None, broadcast: bool) -> dict[str, str]:
    """Resolve ``--node``/``--broadcast`` to ``executions`` query params.

    ``all`` (or legacy ``--broadcast``) → ``mode=broadcast``; ``head`` or a
    rank number → ``mode=run`` plus ``node=``; omitted → round-robin ``run``.
    """
    if node is None:
        return {"mode": "broadcast"} if broadcast else {"mode": "run"}
    if node == "all":
        return {"mode": "broadcast"}
    if node == "head":
        return {"mode": "run", "node": "head"}
    if node.lstrip("-").isdigit():
        return {"mode": "run", "node": node}
    console.print(f"[red]Invalid --node '{node}'. Use head, all, or a rank number.[/red]")
    raise SystemExit(2)


@compute_app.command(name="run")
def run_script(
    name: Annotated[str, Parameter(help="Pool name on the server")],
    script: Annotated[Path, Parameter(help="Path to the .py script to execute")],
    args: Annotated[list[str] | None, Parameter(help="Arguments forwarded as sys.argv to the script")] = None,
    *,
    node: Annotated[
        str | None,
        Parameter(name="--node", help="Target a node: head, all, or a rank number (default round-robin)"),
    ] = None,
    broadcast: Annotated[bool, Parameter(name="--broadcast", help="Alias for --node all")] = False,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Run a local Python script remotely on a pool.

    The script source is sent to the server, exec'd inside a worker, and
    stdout/stderr stream back live via SSE — the CLI subscribes to the
    pool's event stream and prints each ``Log.Emitted`` whose ``task_id``
    matches this execution. The CLI exits with the script's exit code.
    """
    import asyncio

    from skyward.server.wire import encode

    from ._script import build_exec_pending

    if not script.is_file():
        console.print(f"[red]Script not found: {script}[/red]")
        raise SystemExit(2)

    import base64

    pending = build_exec_pending(script, list(args or []))
    body = encode(pending)
    target = resolve_server_url(url)
    params = _run_params(node, broadcast)
    source_header = base64.b64encode(script.read_text().encode()).decode("ascii")

    try:
        with make_client(url) as client:
            r = client.post(
                f"/compute/{name}/executions",
                params=params,
                content=body,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Skyward-Source": source_header,
                },
            )
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None

    if r.status_code == 404:
        console.print(f"[red]Pool {name!r} not found[/red]")
        raise SystemExit(1)
    if r.status_code != 202:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    eid = r.json()["id"]

    exit_code = asyncio.run(_stream_run(target, name, eid))
    if exit_code != 0:
        raise SystemExit(exit_code)


async def _stream_run(target: str, pool_name: str, eid: str) -> int:
    """Tail the pool's SSE for ``Log.Emitted`` events tagged with ``eid``,
    while polling the execution endpoint for the final result.

    Returns the script's exit code (max across nodes for broadcast).
    """
    import asyncio
    import contextlib

    from skyward.api.events import Log
    from skyward.server.wire import decode, event_from_json

    from ._view import iter_sse

    async def stream_logs() -> None:
        try:
            async for event_type, payload in iter_sse(target, pool_name):
                if event_type == "done":
                    return
                event = event_from_json(payload or {})
                if isinstance(event, Log.Emitted) and event.task_id == eid:
                    sys.stdout.write(event.message + "\n")
                    sys.stdout.flush()
        except (httpx.ConnectError, RuntimeError):
            return

    async def wait_result() -> int:
        backoff = 0.1
        async with httpx.AsyncClient(
            base_url=target, timeout=httpx.Timeout(30.0, read=None),
        ) as client:
            while True:
                r = await client.get(f"/compute/{pool_name}/executions/{eid}")
                if r.status_code == 200:
                    payload = decode(r.content)
                    if isinstance(payload, list):
                        return max((int(p.get("exit", 0)) for p in payload), default=0)
                    return int(payload.get("exit", 0))
                if r.status_code == 202:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, 2.0)
                    continue
                if r.status_code == 500 and r.headers.get("X-Skyward-Error") == "1":
                    err = decode(r.content)
                    sys.stderr.write(f"execution error: {err!r}\n")
                    return 1
                console.print(f"[red]{format_http_error(r)}[/red]")
                return 1

    log_task = asyncio.create_task(stream_logs())
    try:
        exit_code = await wait_result()
        # Brief grace period so any in-flight log lines flush before we cancel.
        await asyncio.sleep(0.3)
    finally:
        log_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await log_task
    return exit_code


# ── file operations ──────────────────────────────────────────────


def _raise_for_file_error(r: httpx.Response, name: str, *, expected: int = 200) -> None:
    """Exit with a clean message on the file-route 404/409/other errors."""
    if r.status_code == 404:
        console.print(f"[red]Session {name!r} not found[/red]")
        raise SystemExit(1)
    if r.status_code == 409:
        try:
            status = r.json().get("status", "?")
        except ValueError:
            status = "?"
        console.print(f"[red]session not ready ({status})[/red]")
        raise SystemExit(1)
    if r.status_code != expected:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)


def _print_file_results(results: list[dict]) -> None:
    print_table(
        ["Node", "Status", "Error"],
        [(res["node_id"], "ok" if res["success"] else "fail", res.get("error") or "") for res in results],
    )


@compute_app.command(name="ls")
def ls_files(
    name: Annotated[str, Parameter(help="Session/pool name on the server")],
    path: Annotated[str, Parameter(help="Remote path to list")],
    *,
    node: Annotated[str, Parameter(name="--node", help="head, all, or a rank number")] = "head",
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """List a remote path on the session's node(s)."""
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.get(f"/compute/{name}/files", params={"path": path, "node": node})
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None
    _raise_for_file_error(r, name)
    results = r.json()["results"]
    if json:
        import json as _json

        sys.stdout.write(_json.dumps(results) + "\n")
        return
    for res in results:
        if res["success"]:
            console.print(f"[bold]node {res['node_id']}[/bold]")
            console.print(res["listing"].rstrip())
        else:
            console.print(f"[red]node {res['node_id']}: {res['error']}[/red]")


@compute_app.command(name="rm")
def rm_files(
    name: Annotated[str, Parameter(help="Session/pool name on the server")],
    path: Annotated[str, Parameter(help="Remote path to remove (rm -rf)")],
    *,
    node: Annotated[str, Parameter(name="--node", help="head, all, or a rank number")] = "all",
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Remove a remote path on the session's node(s)."""
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.delete(f"/compute/{name}/files", params={"path": path, "node": node})
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None
    _raise_for_file_error(r, name)
    results = r.json()["results"]
    if json:
        import json as _json

        sys.stdout.write(_json.dumps(results) + "\n")
        return
    _print_file_results(results)


@compute_app.command(name="upload")
def upload(
    name: Annotated[str, Parameter(help="Session/pool name on the server")],
    local: Annotated[Path, Parameter(help="Local file to upload")],
    remote: Annotated[str, Parameter(help="Destination path on the node(s)")],
    *,
    node: Annotated[str, Parameter(name="--node", help="head, all, or a rank number")] = "all",
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Upload a local file to the session's node(s)."""
    if not local.is_file():
        console.print(f"[red]File not found: {local}[/red]")
        raise SystemExit(2)
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.put(
                f"/compute/{name}/files",
                params={"path": remote, "node": node},
                content=local.read_bytes(),
                headers={"Content-Type": "application/octet-stream"},
            )
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None
    _raise_for_file_error(r, name)
    results = r.json()["results"]
    if json:
        import json as _json

        sys.stdout.write(_json.dumps(results) + "\n")
        return
    _print_file_results(results)


@compute_app.command(name="download")
def download(
    name: Annotated[str, Parameter(help="Session/pool name on the server")],
    remote: Annotated[str, Parameter(help="Remote path to download")],
    local: Annotated[Path, Parameter(help="Local destination file")],
    *,
    node: Annotated[str, Parameter(name="--node", help="head or a rank number")] = "head",
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Download a file from a single session node to a local path."""
    target = resolve_server_url(url)
    try:
        with make_client(url) as client, client.stream(
            "GET", f"/compute/{name}/files/content",
            params={"path": remote, "node": node},
        ) as r:
            if r.status_code != 200:
                r.read()
                _raise_for_file_error(r, name)
            with local.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None
    print_status(f"{remote}", "ok", f"→ {local}")


# ── imperative install ───────────────────────────────────────────


def _resolve_specifiers(packages: list[str] | None, requirements: Path | None) -> tuple[str, ...]:
    """Merge positional specifiers with non-comment requirements lines."""
    specs: list[str] = list(packages or [])
    if requirements is not None:
        if not requirements.is_file():
            console.print(f"[red]Requirements file not found: {requirements}[/red]")
            raise SystemExit(2)
        specs.extend(
            line.strip()
            for line in requirements.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not specs:
        console.print("[red]No packages to install[/red]")
        raise SystemExit(2)
    return tuple(specs)


@app.command(name="install")
def install(
    pool: Annotated[str, Parameter(help="Session/pool name")],
    packages: Annotated[list[str] | None, Parameter(help="Package specifiers")] = None,
    *,
    requirements: Annotated[
        Path | None, Parameter(name=("-r", "--requirements"), help="requirements.txt to install"),
    ] = None,
    one: Annotated[bool, Parameter(name="--one", help="Install on a single node (inspection only)")] = False,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Install dependencies into a live session's worker venv via uv.

    Broadcasts ``uv add`` to every node by default (each node has its own
    ``/opt/skyward/.venv``); ``--one`` targets a single node. Output streams
    back live and the CLI exits with the worst node's exit code.
    """
    import asyncio

    from skyward.server.wire import encode

    from ._install import build_install_pending

    specs = _resolve_specifiers(packages, requirements)
    body = encode(build_install_pending(specs))
    target = resolve_server_url(url)
    mode = "run" if one else "broadcast"

    try:
        with make_client(url) as client:
            r = client.post(
                f"/compute/{pool}/executions",
                params={"mode": mode},
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None

    if r.status_code == 404:
        console.print(f"[red]Session {pool!r} not found[/red]")
        raise SystemExit(1)
    if r.status_code == 409:
        console.print("[red]session not ready[/red]")
        raise SystemExit(1)
    if r.status_code != 202:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    eid = r.json()["id"]
    exit_code = asyncio.run(_stream_run(target, pool, eid))
    if exit_code != 0:
        raise SystemExit(exit_code)
