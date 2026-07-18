"""The ``sky`` command — a thin client over the daemon's HTTP API.

Nothing here decides anything. Every command turns a few flags into one HTTP
call and prints what came back, which is why the CLI holds no state and needs
no daemon of its own: :mod:`skyward.cli._client` opens a remote one or runs an
embedded one, and the command cannot tell which it got.

The CLI library is an extra. A node installs ``skyward`` to run somebody's
training loop and has no business acquiring an argument parser to do it.
"""

from __future__ import annotations

import sys

try:
    from cyclopts import App
except ModuleNotFoundError as missing:
    raise SystemExit("the sky CLI needs: pip install 'skyward[cli]'") from missing

app = App(name="sky", help="Skyward CLI — GPU compute orchestration", version_flags=[])

offers_app = App(name="offers", help="Browse GPU offers and pricing")
providers_app = App(name="providers", help="Check cloud provider status")
config_app = App(name="config", help="Inspect configuration")
server_app = App(name="server", help="Manage the Skyward daemon")
compute_app = App(name="compute", help="Manage computes and the work running on them")
log_app = App(name="log", help="Export a compute's execution history")

app.command(offers_app)
app.command(providers_app)
app.command(config_app)
app.command(server_app)
app.command(compute_app)
app.command(log_app)


@app.command
def version() -> None:
    """Print the Skyward and Python versions."""
    from skyward._version import __version__

    print(f"skyward {__version__}, python {sys.version.split()[0]}")


from skyward.cli import compute as compute  # noqa: E402, F401
from skyward.cli import config as config  # noqa: E402, F401
from skyward.cli import console as console  # noqa: E402, F401
from skyward.cli import log as log  # noqa: E402, F401
from skyward.cli import monitor as monitor  # noqa: E402, F401
from skyward.cli import offers as offers  # noqa: E402, F401
from skyward.cli import providers as providers  # noqa: E402, F401
from skyward.cli import server as server  # noqa: E402, F401
from skyward.cli import sessions as sessions  # noqa: E402, F401
from skyward.cli.compute import create_compute  # noqa: E402
from skyward.cli.notebook import notebook_app  # noqa: E402

app.command(notebook_app)
app.command(create_compute, name="new")

__all__ = [
    "app",
    "compute_app",
    "config_app",
    "log_app",
    "notebook_app",
    "offers_app",
    "providers_app",
    "server_app",
    "version",
]
