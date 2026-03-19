"""sky config — inspect configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from . import config_app
from ._output import console, print_status


@config_app.command(name="path")
def config_path(
    *,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Show configuration file locations."""
    from skyward.config import GLOBAL_CONFIG_PATH, PROJECT_CONFIG_NAME

    global_path = GLOBAL_CONFIG_PATH
    project_path = Path.cwd() / PROJECT_CONFIG_NAME

    if json:
        import json as _json
        import sys

        sys.stdout.write(
            _json.dumps({"global": str(global_path), "project": str(project_path)}) + "\n"
        )
        return

    print_status("Global config", "ok" if global_path.is_file() else "-", str(global_path))
    print_status("Project config", "ok" if project_path.is_file() else "-", str(project_path))


@config_app.command(name="show")
def config_show(
    *,
    pool: Annotated[str | None, Parameter(name="--pool", help="Filter to a single named pool")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Show merged configuration (global + project)."""
    from skyward.config import load_config

    cfg = load_config()

    if pool:
        pools = cfg.get("pools", {})
        if pool not in pools:
            console.print(f"[red]Pool '{pool}' not found. Available: {', '.join(pools) or 'none'}[/red]")
            return
        cfg = {"pools": {pool: pools[pool]}}

    if json:
        import json as _json
        import sys

        sys.stdout.write(_json.dumps(cfg, indent=2, default=str) + "\n")
        return

    def _to_toml_lines(data: dict, indent: int = 0) -> list[str]:
        lines: list[str] = []
        prefix = "  " * indent
        for key, value in data.items():
            match value:
                case dict():
                    lines.append(f"{prefix}[{key}]")
                    lines.extend(_to_toml_lines(value, indent + 1))
                case _:
                    lines.append(f"{prefix}{key} = {value!r}")
        return lines

    for line in _to_toml_lines(cfg):
        console.print(line)


@config_app.command(name="validate")
def config_validate(
    *,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Validate configuration files and pool references."""
    from skyward.config import GLOBAL_CONFIG_PATH, PROJECT_CONFIG_NAME, _get_provider_map, load_config

    global_path = GLOBAL_CONFIG_PATH
    project_path = Path.cwd() / PROJECT_CONFIG_NAME

    print_status(
        "Global config",
        "ok" if global_path.is_file() else "-",
        str(global_path),
        as_json=json,
    )
    print_status(
        "Project config",
        "ok" if project_path.is_file() else "-",
        str(project_path),
        as_json=json,
    )

    try:
        cfg = load_config()
    except Exception as exc:
        print_status("Parse", "fail", str(exc)[:120], as_json=json)
        return

    provider_map = _get_provider_map()
    providers = cfg.get("providers", {})

    for name, raw in providers.items():
        ptype = raw.get("type", "")
        if ptype in provider_map:
            print_status(f"Provider '{name}'", "ok", f"type={ptype}", as_json=json)
        else:
            print_status(f"Provider '{name}'", "fail", f"unknown type '{ptype}'", as_json=json)

    pools = cfg.get("pools", {})
    for name, raw in pools.items():
        provider_ref = raw.get("provider", "")
        if provider_ref in providers:
            print_status(f"Pool '{name}'", "ok", f"provider '{provider_ref}' exists", as_json=json)
        else:
            print_status(f"Pool '{name}'", "fail", f"provider '{provider_ref}' not defined in [providers]", as_json=json)
