"""``sky notebook`` — the Jupyter kernelspec for a compute.

Two commands that write and delete one file. What the kernel does lives in
:mod:`skyward.core.notebook`; this only records which compute it should attach to,
and through which daemon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from skyward.cli._client import resolve
from skyward.cli._output import Output, render

notebook_app = App(name="notebook", help="Manage the Skyward Jupyter kernel")

COLUMNS = ("kernel", "compute", "display")
MISSING = "the Skyward Jupyter kernel needs: pip install 'skyward[notebook]'"


@notebook_app.command(name="install")
def install_kernel(
    compute: Annotated[str, Parameter(help="Compute to attach the kernel to, by name or id")],
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Daemon URL")] = None,
    directory: Annotated[Path | None, Parameter(name="--directory", help="Write the spec here instead of installing it user-level")] = None,
    output: Annotated[Output, Parameter(name="--output", help="table or json")] = "table",
) -> None:
    """Install a Jupyter kernel that runs on a compute.

    Open Jupyter afterwards and pick ``Skyward (<compute>)``.
    """
    try:
        from skyward.core.notebook.kernelspec import install_kernelspec
    except ModuleNotFoundError as missing:
        raise SystemExit(MISSING) from missing

    name = install_kernelspec(compute, resolve(url) or "", directory)
    render(COLUMNS, [(name, compute, f"Skyward ({compute})")], output=output)


@notebook_app.command(name="remove")
def remove_kernel(
    compute: Annotated[str, Parameter(help="Compute whose kernel to remove")],
    *,
    directory: Annotated[Path | None, Parameter(name="--directory", help="Remove from this tree instead of the user-level location")] = None,
    output: Annotated[Output, Parameter(name="--output", help="table or json")] = "table",
) -> None:
    """Remove a compute's Jupyter kernel."""
    try:
        from skyward.core.notebook.kernelspec import remove_kernelspec
    except ModuleNotFoundError as missing:
        raise SystemExit(MISSING) from missing

    name = remove_kernelspec(compute, directory)
    render(COLUMNS[:2], [(name, compute)], output=output)


__all__ = ["install_kernel", "notebook_app", "remove_kernel"]
