"""sky config — what a command resolved before it dialled anything.

There is no configuration file. A command is a daemon URL and a database path,
both resolved from the environment at the moment of the call, so configuration
here is that resolution made visible: which daemon a call would reach, where an
embedded one would keep its state, and whether the daemon answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from . import config_app
from ._client import call, resolve
from ._output import EMPTY, Output, render

type Setting = tuple[str, str, str]


def _settings(url: str | None, database: Path | None) -> Setting:
    from skyward.persistence.db import DEFAULT_PATH

    target = resolve(url)
    source = "flag" if url else "environment" if target else "embedded"
    return target or EMPTY, source, str(database or DEFAULT_PATH)


@config_app.command(name="path")
def config_path(
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(name="--database", help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(name="--output", help="table or json")] = "table",
) -> None:
    """Show the daemon database path and the resolved daemon URL."""
    target, _, path = _settings(url, database)

    render(["setting", "value"], [["database", path], ["url", target]], output=output)


@config_app.command(name="show")
def config_show(
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(name="--database", help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(name="--output", help="table or json")] = "table",
) -> None:
    """Show the effective settings a command would run with."""
    target, source, path = _settings(url, database)

    render(
        ["setting", "value"],
        [
            ["url", target],
            ["source", source],
            ["database", path],
            ["database exists", str(Path(path).is_file()).lower()],
        ],
        output=output,
    )


@config_app.command(name="validate")
def config_validate(
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(name="--database", help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(name="--output", help="table or json")] = "table",
) -> None:
    """Check that the resolved daemon is reachable and ready."""
    target, source, _ = _settings(url, database)

    try:
        body = call(
            lambda client: client.call("GET", "/v1/health/ready", dict[str, bool]),
            url=url,
            database=database,
        )
    except Exception as exc:
        status, detail = "fail", str(exc)[:120] or exc.__class__.__name__
    else:
        status, detail = ("ok", "ready") if body.get("ready") else ("fail", "not ready")

    render(["check", "status", "detail"], [[f"daemon ({source})", status, detail], ["url", "-", target]], output=output)

    if status == "fail":
        raise SystemExit(1)


__all__ = ["config_path", "config_show", "config_validate"]
