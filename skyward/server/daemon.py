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
import ipaddress
import os
import resource
import socket
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from skyward.shared.observability import logger

logger = logger.bind(component="daemon")

RUNTIME_DIR = Path.home() / ".skyward"
PID_FILE = RUNTIME_DIR / "server.pid"
LOG_FILE = RUNTIME_DIR / "server.log"
INTERFACE_FILE = RUNTIME_DIR / "server.interface"
"""The network interface ``sky server interface set`` recorded — see :func:`listening`."""
WILDCARD = "0.0.0.0"

MISSING = "the daemon needs an ASGI server: pip install 'skyward[server]'"
STARTUP_FAILURE = 3
"""The exit status of a daemon that never came up — uvicorn's own, for a server that failed to start."""

GRACEFUL_SECONDS = 5
"""How long a stopping daemon waits for its open connections.

Uvicorn's default is forever, and this daemon's connections are the kind that
never end on their own — an event stream being watched, a result being
long-polled. A stop that waits for those outlives every ``sky server stop``
timeout; the clients know how to come back, so they are cut instead.
"""

DESCRIPTORS = 8192
"""How many descriptors a daemon asks for, whatever the shell that started it had.

A daemon's descriptors are its work and not a leak: a node is an SSH connection, a
listening tunnel and the connections through it, and every task in flight is a
client connection held open for its result. Thirty-two nodes running four tasks
each is past the 256 a macOS shell hands down, and the first thing to fail there
is whatever opens a file next — the database, which then answers every write
with a 500.
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


def interface() -> str | None:
    """The interface a daemon started here listens on beside its host, or None when none was recorded."""
    try:
        return INTERFACE_FILE.read_text().strip() or None
    except OSError:
        return None


def choose(name: str) -> None:
    """Record ``name`` as the interface every daemon started here from now on listens on."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    INTERFACE_FILE.write_text(name)


def interfaces() -> dict[str, tuple[str, ...]]:
    """Every network interface of this machine, with the IPv4 addresses it has now."""
    import psutil

    return {name: tuple(held.address for held in addresses if held.family == socket.AF_INET) for name, addresses in psutil.net_if_addrs().items()}


def addresses(name: str) -> tuple[str, ...]:
    """What ``name`` stands for: itself when it is an address, else the IPv4 addresses the interface has now."""
    try:
        return (str(ipaddress.IPv4Address(name)),)
    except ValueError:
        return interfaces().get(name, ())


def listening(host: str) -> tuple[str, ...]:
    """Where a daemon started here on ``host`` listens: ``host``, then the recorded interface.

    ``host`` is where this machine's pools and commands look for the daemon, so the
    interface is listened on beside it and never in its place. A name is read for
    its addresses when the daemon starts rather than when it was recorded, because
    the address is what moves — a DHCP lease, a VPN that came back on another. The
    wildcard already covers every other address, so it is listened on alone.
    """
    beside = addresses(name) if (name := interface()) else ()
    wanted = tuple(dict.fromkeys((host, *beside)))
    return (WILDCARD,) if WILDCARD in wanted else wanted


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


def descriptors(wanted: int = DESCRIPTORS) -> None:
    """Raise this process's descriptor limit to ``wanted``, as far as the hard limit lets it."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    ceiling = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
    if soft < ceiling:
        resource.setrlimit(resource.RLIMIT_NOFILE, (ceiling, hard))


def serve(host: str, port: int, database: Path | None = None, log_level: str | None = None, access_log: bool = True) -> None:
    """Run the daemon here, ending with whoever started it.

    A daemon being stopped hears it before its server starts cutting connections, so
    the results being long-polled are answered — no outcome yet, ask again — instead
    of being cut with a 500 their callers would take for the task's verdict. See
    :meth:`skyward.server.persistence.tasks.TaskStore.close`.

    What it cannot bind of the recorded interface is logged and left out rather
    than taking the daemon down with it: the host is where this machine's own work
    reaches it, and a VPN that is down is no reason for that to stop.
    """
    import uvicorn

    from skyward.server.http.app import daemon

    os.environ.update(environment(database, log_level))
    descriptors()
    standalone = daemon()

    first, *beside = listening(host)
    try:
        sockets = [_bound(first, port)]
    except OSError as refused:
        raise SystemExit(f"cannot listen on {first}:{port}: {refused.strerror}") from None
    for address in beside:
        try:
            sockets.append(_bound(address, port))
        except OSError as refused:
            logger.warning("not listening on {}:{}: {}", address, port, refused.strerror)

    class Server(uvicorn.Server):
        async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
            standalone.closing()
            await super().shutdown(sockets)

    server = Server(uvicorn.Config(standalone.app, host=host, port=port, timeout_graceful_shutdown=GRACEFUL_SECONDS, access_log=access_log))
    with suppress(KeyboardInterrupt):
        server.run(sockets)
    if not server.started:
        raise SystemExit(STARTUP_FAILURE)


def spawn(host: str, port: int, database: Path | None = None, log_level: str | None = None) -> int:
    """Start a daemon in a session of its own and return its pid.

    Detached deliberately: a control plane that dies with the terminal — or with
    the script — that launched it is not a control plane. Its output goes to
    :data:`LOG_FILE`, which is the only account of a daemon that never answers.
    Uvicorn's access log is left out of it: a line per request is noise in a file
    nothing rotates, and the daemon's own rotating log already records what matters.
    """
    if not installed():
        raise ImportError(MISSING)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_FILE.open("ab")  # noqa: SIM115
    process = subprocess.Popen(
        [sys.executable, "-m", "skyward.server.daemon", host, str(port)],
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        env=environment(database, log_level),
    )
    return process.pid


def _bound(address: str, port: int) -> socket.socket:
    """A socket bound to ``address`` as uvicorn binds one for its host: not yet listening, so refused until the app is up."""
    bound = socket.socket(socket.AF_INET6 if ":" in address else socket.AF_INET)
    bound.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        bound.bind((address, port))
    except OSError:
        bound.close()
        raise
    return bound


__all__ = [
    "DESCRIPTORS",
    "INTERFACE_FILE",
    "LOG_FILE",
    "MISSING",
    "PID_FILE",
    "RUNTIME_DIR",
    "WILDCARD",
    "addresses",
    "alive",
    "choose",
    "descriptors",
    "forget",
    "installed",
    "interface",
    "interfaces",
    "listening",
    "pid",
    "record",
    "serve",
    "spawn",
]


if __name__ == "__main__":
    serve(sys.argv[1], int(sys.argv[2]), access_log=False)
