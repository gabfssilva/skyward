"""The daemon as a process: where its pid goes, and how one is started.

:mod:`skyward.server.http.app` is the application; this is the process around it.
It lives here rather than in the CLI because starting a daemon is not a CLI act —
a pool that finds nothing at the default address starts one too, and the two have
to agree on the pidfile, or ``sky server stop`` would not stop what a pool began.

Nothing here waits for the daemon to answer. Whoever started it holds a client
already, and asking is that client's job.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

RUNTIME_DIR = Path.home() / ".skyward"
PID_FILE = RUNTIME_DIR / "server.pid"
LOG_FILE = RUNTIME_DIR / "server.log"

TARGET = "skyward.server.http.app:daemon"
MISSING = "the daemon needs an ASGI server: pip install 'skyward[server]'"

GRACEFUL_SECONDS = 5
"""How long a stopping daemon waits for its open connections.

Uvicorn's default is forever, and this daemon's connections are the kind that
never end on their own — an event stream being watched, a result being
long-polled. A stop that waits for those outlives every ``sky server stop``
timeout; the clients know how to come back, so they are cut instead.
"""


def installed() -> bool:
    """Whether there is an ASGI server here to run the application."""
    return importlib.util.find_spec("uvicorn") is not None


def pid() -> int | None:
    """The recorded pid, or None when there is no readable pidfile."""
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def alive(process: int) -> bool:
    """Whether the process exists, signalling nothing to find out."""
    try:
        os.kill(process, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def record(process: int) -> None:
    """Write the pid down, once the daemon it names has answered.

    After rather than before, because the pidfile is what ``stop`` reads: a pid
    written by a start that then lost the port would name a dead process, and the
    daemon that won it would be the one nobody could stop.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(process))


def forget() -> None:
    """Drop the pidfile, whether or not there is one."""
    PID_FILE.unlink(missing_ok=True)


def environment(database: Path | None, log_level: str | None = None) -> dict[str, str]:
    """This process's environment, plus what the daemon is to open and how loudly.

    Both travel as variables rather than as arguments because the daemon is not
    called, it is spawned: uvicorn imports the factory, and the factory is the only
    thing in a position to read them.
    """
    return {
        **os.environ,
        **({"SKYWARD_DATABASE": str(database)} if database is not None else {}),
        **({"SKYWARD_LOG_LEVEL": log_level} if log_level is not None else {}),
    }


def serve(host: str, port: int, database: Path | None = None, log_level: str | None = None) -> None:
    """Run the daemon here, ending with whoever started it."""
    import uvicorn

    os.environ.update(environment(database, log_level))
    uvicorn.run(TARGET, host=host, port=port, factory=True, timeout_graceful_shutdown=GRACEFUL_SECONDS)


def spawn(host: str, port: int, database: Path | None = None, log_level: str | None = None) -> int:
    """Start a daemon in a session of its own and return its pid.

    Detached deliberately: a control plane that dies with the terminal — or with
    the script — that launched it is not a control plane. Its output goes to
    :data:`LOG_FILE`, which is the only account of a daemon that never answers.
    """
    if not installed():
        raise ImportError(MISSING)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_FILE.open("ab")  # noqa: SIM115
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        TARGET,
        "--factory",
        "--host",
        host,
        "--port",
        str(port),
        "--timeout-graceful-shutdown",
        str(GRACEFUL_SECONDS),
    ]
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        env=environment(database, log_level),
    )
    return process.pid


__all__ = ["LOG_FILE", "MISSING", "PID_FILE", "RUNTIME_DIR", "TARGET", "alive", "forget", "installed", "pid", "record", "serve", "spawn"]
