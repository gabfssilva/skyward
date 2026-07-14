"""Where the node gets skyward from.

Three answers, and the interesting one is ``local``: the daemon builds a wheel
out of the checkout it is itself running from, and ships that. Without it nothing
unpublished could ever run on a real machine — which, while the code is being
written, is all of it.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import msgspec
from msgspec import Struct, field

from skyward.protocol.schemas import SkywardSource
from skyward.runtime.journal import SKYWARD_DIR

REPO = "git+https://github.com/gabfssilva/skyward.git#subdirectory=v2"
PROJECT = Path(__file__).resolve().parent.parent.parent


class Source(Struct, frozen=True):
    """What the bootstrap installs, and what has to be there before it can.

    Attributes
    ----------
    argument : str
        What follows ``uv pip install`` — a package name, a git URL, or the path
        of a wheel on the machine.
    wheel : bytes | None
        The wheel itself, when there is one to upload. It is carried as bytes
        rather than a path because it is built once for the whole compute and
        every node needs it.
    """

    argument: str
    wheel: bytes | None = None


class DirInfo(Struct):
    editable: bool = False


class DirectUrl(Struct):
    dir_info: DirInfo = field(default_factory=DirInfo)


async def resolve(mode: SkywardSource) -> Source:
    """Turn the requested mode into something a node can install."""
    match detect() if mode == "auto" else mode:
        case "pypi":
            return Source(argument="skyward")
        case "github":
            return Source(argument=REPO)
        case "local":
            wheel = await asyncio.to_thread(build)
            return Source(argument=f"{SKYWARD_DIR}/{wheel.name}", wheel=wheel.read_bytes())


def detect() -> Literal["local", "pypi"]:
    """Where the daemon's own skyward came from.

    Reading it off the running installation rather than asking is deliberate: what
    the node runs should be what the daemon is, and a flag the user has to keep in
    step with their venv is a flag that will disagree with it.
    """
    from importlib.metadata import distribution, packages_distributions

    installed = packages_distributions().get("skyward")
    if not installed:
        return "local"

    url = distribution(installed[0]).read_text("direct_url.json")
    editable = url is not None and msgspec.json.decode(url.encode(), type=DirectUrl).dir_info.editable
    return "local" if editable else "pypi"


def build() -> Path:
    """Build a wheel from the checkout, into a directory nobody else is using."""
    out = tempfile.mkdtemp(prefix="skyward-wheel-")
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

    return next(Path(out).glob("*.whl"))
