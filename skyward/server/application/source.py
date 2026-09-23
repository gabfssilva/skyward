"""Where the node gets skyward from.

Three answers, and the interesting one is ``local``: the daemon builds a wheel
out of the checkout it is itself running from, and ships that. Without it nothing
unpublished could ever run on a real machine — which, while the code is being
written, is all of it. The casty it runs goes with it when that is unpublished too.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
import threading
from functools import cache
from importlib.metadata import distribution
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

import msgspec
from msgspec import Struct

from skyward.shared.schemas import SkywardSource
from skyward.worker.journal import SKYWARD_DIR

REPO = "git+https://github.com/gabfssilva/skyward.git#subdirectory=v2"
PROJECT = Path(__file__).resolve().parents[3]


class Wheel(Struct, frozen=True):
    name: str
    data: bytes


class Source(Struct, frozen=True):
    """What the bootstrap installs, and what has to be there before it can.

    Attributes
    ----------
    arguments : tuple[str, ...]
        What follows ``uv pip install`` — package names, git URLs, or the paths
        of wheels on the machine, and the options that say where else to look.
    wheels : tuple[Wheel, ...]
        The wheels themselves, when there are wheels to upload. They are carried
        as bytes rather than paths because they are built once for the whole
        compute and every node needs them.
    """

    arguments: tuple[str, ...]
    wheels: tuple[Wheel, ...] = ()

    @property
    def argument(self) -> str:
        """The arguments as one install clause."""
        return " ".join(self.arguments)


class DirInfo(Struct):
    editable: bool = False


class DirectUrl(Struct):
    url: str = ""
    dir_info: DirInfo | None = None


async def resolve(mode: SkywardSource) -> Source:
    """Turn the requested mode into something a node can install."""
    match await detect() if mode == "auto" else mode:
        case "pypi":
            return Source(arguments=("skyward",))
        case "github":
            return Source(arguments=(REPO,))
        case "local":
            wheels = await asyncio.to_thread(build)
            casty = await asyncio.to_thread(checkout)
            shipped = ("--find-links", SKYWARD_DIR, "--reinstall-package", "casty") if casty else ()
            return Source(arguments=(*shipped, *(f"{SKYWARD_DIR}/{wheel.name}" for wheel in wheels)), wheels=(*wheels, *casty))


async def detect() -> Literal["local", "pypi"]:
    """Where the daemon's own skyward came from.

    Reading it off the running installation rather than asking is deliberate: what
    the node runs should be what the daemon is, and a flag the user has to keep in
    step with their venv is a flag that will disagree with it.

    Read once per process, and never on the event loop: ``packages_distributions``
    reads the file list of every distribution installed, a tenth of a second or more
    of blocking I/O, and the installation this process is running from does not
    change while it runs.
    """

    def once() -> Literal["local", "pypi"]:
        with _lock:
            return _installation()

    return await asyncio.to_thread(once)


def build() -> tuple[Wheel, ...]:
    """Build a wheel from the checkout, into a directory nobody else is using.

    The directory is gone by the time this returns: the wheels leave as bytes, and
    nothing needs the files after they are read.
    """
    with tempfile.TemporaryDirectory(prefix="skyward-wheel-") as out:
        result = subprocess.run(
            ["uv", "build", "--wheel", "-o", out],
            cwd=PROJECT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"could not build the skyward wheel: {result.stderr}")

        return tuple(Wheel(name=path.name, data=path.read_bytes()) for path in Path(out).glob("*.whl"))


def checkout() -> tuple[Wheel, ...]:
    """The casty this daemon runs, as the wheels a node installs, when it is a checkout and not a release.

    A casty installed from an index is one the node installs from that index as well,
    and nothing is shipped. One installed from a directory is unpublished, like the
    skyward beside it, and a node has nowhere to get it from but here: the manylinux
    wheels of the same version in that directory's ``dist``, one per architecture,
    which the node picks from with ``--find-links``. Reinstalled, because a rebuilt
    wheel keeps its version and a machine that has one would keep the old.

    A checkout with no such wheel is refused here, by name, rather than leaving the
    node to install whatever casty the index has — which is not the one this daemon
    speaks, and would be refused by every worker's handshake after a whole bootstrap.
    """
    installed = distribution("casty")
    origin = installed.read_text("direct_url.json")
    if origin is None:
        return ()
    direct = msgspec.json.decode(origin.encode(), type=DirectUrl)
    if direct.dir_info is None or not direct.url.startswith("file://"):
        return ()

    dist = Path(unquote(urlparse(direct.url).path)) / "dist"
    wheels = sorted(dist.glob(f"casty-{installed.version}-*-manylinux*.whl"))
    if not wheels:
        raise RuntimeError(
            f"casty {installed.version} is installed from {dist.parent}, and {dist} has no manylinux wheel of it for a node to install: "
            "build them there with `uvx maturin build --release --zig --target <x86_64|aarch64>-unknown-linux-gnu --compatibility manylinux2014 --out dist`"
        )
    return tuple(Wheel(name=path.name, data=path.read_bytes()) for path in wheels)


_lock = threading.Lock()
"""Held around the cached read: ``functools.cache`` alone lets two threads that miss at once both do the work."""


@cache
def _installation() -> Literal["local", "pypi"]:
    from importlib.metadata import packages_distributions

    installed = packages_distributions().get("skyward")
    if not installed:
        return "local"

    url = distribution(installed[0]).read_text("direct_url.json")
    info = None if url is None else msgspec.json.decode(url.encode(), type=DirectUrl).dir_info
    return "local" if info is not None and info.editable else "pypi"
