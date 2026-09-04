"""What the feature tests run against: containers on this machine, and nothing else.

The daemon is a real one — uvicorn in a process of its own, one per xdist worker —
because that is the deployment where everything can reach it: the pools in this
process, and the ``sky`` command run as a shell would run it. An embedded daemon
would tie the compute's SSH connections to this process, and a CLI subprocess
could never reach them.

The pool is the expensive thing — a container, an sshd, a bootstrap and a wheel per
node — so :func:`pool` builds two of them once for the whole session and every
feature that fits two CPU nodes shares them. The files that share it are pinned to
one xdist worker (``xdist_group("pool")``), because a session fixture is per worker
and an unpinned group would buy a pool per core.

A feature that is *about* the pool — an image, a plugin, an elastic range, an
attachment — asks :func:`compute` for its own and pays for it.
"""

import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import msgspec
import pytest

import skyward as sky
from skyward.server.application.mock import SPEC
from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.shared.schemas import Compute, ComputeCreate, PluginRef

PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"
"""The node's interpreter is the one running the tests.

A task crosses the wire as cloudpickle, which is bytecode, and bytecode does not
survive a change of minor version. A 3.12 client against a 3.13 node does not
fail on the payload — the worker dies unpickling it, and the client sees the
connection drop. Any test that builds an image of its own pins this too.
"""

IMAGE = sky.Image(python=PYTHON, skyward="local")
"""The wheel comes from this checkout, so a test runs against the code under it."""

DEFAULT_EXECUTOR = sky.Executor()
DEFAULT_OPTIONS = sky.Options()


SKY = Path(sys.executable).parent / "sky"


@dataclass(frozen=True, slots=True)
class Ran:
    """What the shell would have seen."""

    code: int
    out: str
    err: str


def cli(*tokens: str) -> Ran:
    """Run the ``sky`` command in a process of its own, as a shell does."""
    done = subprocess.run([str(SKY), *tokens], capture_output=True, text=True, timeout=180, check=False)
    return Ran(done.returncode, done.stdout, done.stderr)


def rows(*tokens: str) -> Any:
    """The same, asking for json so the answer can be read rather than matched."""
    ran = cli(*tokens, "--output", "json")
    assert ran.code == 0, f"sky {' '.join(tokens)} left with {ran.code}: {ran.err.strip()}"
    return json.loads(ran.out)


class Build(Protocol):
    """A pool built to order — the knobs a feature test is allowed to turn."""

    def __call__(
        self,
        *,
        nodes: int | tuple[int, int] | sky.Nodes = 1,
        image: sky.Image = ...,
        plugins: Sequence[sky.plugins.Plugin] = (),
        executor: sky.Executor = ...,
        options: sky.Options = ...,
        ports: Sequence[sky.Port] = (),
        name: str | None = None,
        delete_on_exit: bool = True,
        callbacks: Sequence[sky.EventCallback] = (),
    ) -> sky.Compute: ...


@pytest.fixture(scope="session")
def shared_database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where this worker's daemon keeps its state."""
    return tmp_path_factory.mktemp("shared") / "skyward.sqlite"


@contextmanager
def serving(database: Path, log: Path) -> Iterator[str]:
    """A daemon on a free port, for as long as the block lasts.

    The CLI has no embedded transport, so every command in a test needs one of
    these — there is no database a test can hand ``sky`` instead of a daemon.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    url = f"http://127.0.0.1:{port}"
    sink = log.open("wb")
    process = subprocess.Popen(
        [str(SKY), "server", "start", "--foreground", "--port", str(port), "--database", str(database)],
        stdout=sink,
        stderr=sink,
    )

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/v1/health/ready", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        process.terminate()
        raise RuntimeError("the daemon never answered /health")

    try:
        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.fixture(scope="session")
def daemon(shared_database: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A real daemon on a port of its own, serving this worker's computes."""
    with serving(shared_database, tmp_path_factory.mktemp("daemon") / "daemon.log") as url:
        yield url


@pytest.fixture(scope="session")
def pool(daemon: str) -> Iterator[sky.Compute]:
    with sky.Compute(
        provider=sky.Container(),
        nodes=2,
        cpus=1,
        memory_gb=1,
        image=IMAGE,
        name="shared",
        url=daemon,
    ) as pool:
        yield pool


@pytest.fixture
def alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A daemon of this test's own, never one somebody else started."""
    monkeypatch.delenv("SKYWARD_URL", raising=False)
    with serving(tmp_path / "skyward.sqlite", tmp_path / "daemon.log") as url:
        yield url


@pytest.fixture
def compute(daemon: str) -> Build:
    """Built-to-order pools live on the same daemon the session pool does."""

    def build(
        *,
        nodes: int | tuple[int, int] | sky.Nodes = 1,
        image: sky.Image = IMAGE,
        plugins: Sequence[sky.plugins.Plugin] = (),
        executor: sky.Executor = DEFAULT_EXECUTOR,
        options: sky.Options = DEFAULT_OPTIONS,
        ports: Sequence[sky.Port] = (),
        name: str | None = None,
        delete_on_exit: bool = True,
        callbacks: Sequence[sky.EventCallback] = (),
    ) -> sky.Compute:
        return sky.Compute(
            provider=sky.Container(),
            nodes=nodes,
            cpus=1,
            memory_gb=1,
            image=image,
            plugins=list(plugins),
            executor=executor,
            options=options,
            ports=list(ports),
            name=name,
            delete_on_exit=delete_on_exit,
            callbacks=callbacks,
            url=daemon,
        )

    return build


async def given(database: Path, *plugins: str, events: EventStore | None = None) -> tuple[ComputeStore, Compute]:
    """A compute in a database of this test's own, carrying the plugins named."""
    await connect(database)
    store = ComputeStore(events or EventStore())
    spec = msgspec.structs.replace(SPEC, plugins=tuple(PluginRef(kind=kind) for kind in plugins))
    compute, _ = await store.create(ComputeCreate(spec=spec), idempotency_key="given")
    return store, compute
