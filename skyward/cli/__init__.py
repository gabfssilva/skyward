"""Skyward CLI — read-only commands for offers, providers, and configuration."""

from __future__ import annotations

import sys

from cyclopts import App

app = App(name="sky", help="Skyward CLI — GPU compute orchestration", version_flags=[])

offers_app = App(name="offers", help="Browse GPU offers and pricing")
providers_app = App(name="providers", help="Check cloud provider status")
config_app = App(name="config", help="Inspect configuration")

app.command(offers_app)
app.command(providers_app)
app.command(config_app)


@app.command
def version() -> None:
    """Print Skyward and Python version."""
    from skyward import __version__

    print(f"skyward {__version__}, python {sys.version.split()[0]}")


from skyward.cli import config as config  # noqa: F401, E402
from skyward.cli import offers as offers  # noqa: F401, E402
from skyward.cli import providers as providers  # noqa: F401, E402

__all__ = ["app", "offers_app", "providers_app", "config_app"]
