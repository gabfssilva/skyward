"""sky providers — check cloud provider status."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from . import providers_app
from ._output import print_status, print_table

type CredentialStatus = tuple[str, str, str]

_PROVIDER_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "aws": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    "gcp": ("GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_CONFIG"),
    "hyperstack": ("HYPERSTACK_API_KEY",),
    "jarvislabs": ("JL_API_KEY",),
    "runpod": ("RUNPOD_API_KEY",),
    "tensordock": ("TENSORDOCK_API_KEY",),
    "vastai": ("VAST_API_KEY",),
    "verda": ("VERDA_CLIENT_ID", "VERDA_CLIENT_SECRET"),
    "vultr": ("VULTR_API_KEY",),
}

_AUTH_LABELS: dict[str, str] = {
    "aws": "IAM credentials",
    "gcp": "Application Default Credentials",
    "hyperstack": "API key",
    "jarvislabs": "API key",
    "runpod": "API key",
    "tensordock": "API key",
    "vastai": "API key",
    "verda": "OAuth client credentials",
    "vultr": "API key",
}

_FILE_AUTH: dict[str, tuple[Path, str]] = {
    "aws": (Path.home() / ".aws" / "credentials", "~/.aws/credentials"),
    "gcp": (Path.home() / ".config" / "gcloud" / "application_default_credentials.json", "Application Default Credentials"),
    "vastai": (Path.home() / ".config" / "vastai" / "vast_api_key", "~/.config/vastai/vast_api_key"),
    "runpod": (Path.home() / ".runpod" / "config.toml", "~/.runpod/config.toml"),
}


def _check_credentials(provider_name: str) -> CredentialStatus:
    env_vars = _PROVIDER_CREDENTIALS.get(provider_name, ())
    if not env_vars:
        return ("-", "-", "Unknown provider")

    all_present = all(os.environ.get(v) for v in env_vars)
    if all_present:
        return ("ok", _AUTH_LABELS.get(provider_name, "Configured"), "")

    if file_auth := _FILE_AUTH.get(provider_name):
        path, label = file_auth
        if path.exists():
            return ("ok", label, "")

    found = [v for v in env_vars if os.environ.get(v)]
    if not found:
        return ("-", "-", "Not configured")
    return ("fail", _AUTH_LABELS.get(provider_name, "Partial"), f"Missing: {', '.join(v for v in env_vars if not os.environ.get(v))}")


@providers_app.command(name="list")
def list_providers(
    *,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """List all providers and their credential status."""
    columns = ["Provider", "Status", "Auth", "Detail"]
    rows = []
    for name in sorted(_PROVIDER_CREDENTIALS):
        status, auth, detail = _check_credentials(name)
        rows.append((name, status, auth, detail))

    print_table(columns, rows, as_json=json)


def _get_config_class(name: str) -> type | None:
    from skyward.config import _get_provider_map

    return _get_provider_map().get(name)


async def _deep_check(name: str) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []

    status, auth, detail = _check_credentials(name)
    if status != "ok":
        results.append(("Credentials", status, detail or "Not configured"))
        return results
    results.append(("Credentials", "ok", auth))

    config_cls = _get_config_class(name)
    if config_cls is None:
        results.append(("SDK", "-", "Unknown provider"))
        return results

    try:
        config = config_cls()
        await config.create_provider()
        results.append(("SDK auth", "ok", f"Provider created: {name}"))
    except Exception as exc:
        results.append(("SDK auth", "fail", str(exc)[:120]))
        return results

    return results


@providers_app.command(name="check")
def check_provider(
    name: Annotated[str | None, Parameter(help="Provider name to check")] = None,
    *,
    all: Annotated[bool, Parameter(name="--all", help="Check all configured providers")] = False,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Deep-check provider authentication and SDK access."""
    if not name and not all:
        from ._output import console

        console.print("[red]Specify a provider name or use --all[/red]")
        return

    targets = sorted(_PROVIDER_CREDENTIALS) if all else [name]  # type: ignore[list-item]

    for target in targets:
        if target not in _PROVIDER_CREDENTIALS:
            print_status(f"{target}", "fail", "Unknown provider", as_json=json)
            continue

        results = asyncio.run(_deep_check(target))
        if json:
            import json as _json
            import sys

            sys.stdout.write(_json.dumps({"provider": target, "checks": [{"check": c, "status": s, "detail": d} for c, s, d in results]}) + "\n")
        else:
            from ._output import console

            console.print(f"\n[bold]{target.upper()} Provider Check[/bold]")
            for check, status, detail in results:
                print_status(check, status, detail)
