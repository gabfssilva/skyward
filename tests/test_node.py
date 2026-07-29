"""The node's includes step, exercised through the ``Ssh`` protocol — no machine.

The protocol is there precisely so this can be tested without a container: a fake
that records what it was told to put and run is enough to pin the one thing that
is easy to get wrong — where the tarball lands, and that a failure to unpack it
fails the node rather than passing silently.
"""

from __future__ import annotations

import pytest

from skyward.application.provider import Machine
from skyward.protocol.schemas import Image, Options
from skyward.runtime import bootstrap
from skyward.runtime.node import BootstrapFailedError, Node
from skyward.runtime.source import Source
from skyward.runtime.ssh import Result

pytestmark = pytest.mark.unit

PURELIB = "/opt/skyward/.venv/lib/python3.13/site-packages"


class FakeSsh:
    def __init__(self, *, extract_code: int = 0, purelib: str = PURELIB) -> None:
        self.puts: list[tuple[str, bytes]] = []
        self.runs: list[str] = []
        self._extract_code = extract_code
        self._purelib = purelib

    async def put(self, path: str, content: bytes) -> None:
        self.puts.append((path, content))

    async def run(self, command: str, *, timeout: float | None = None) -> Result:
        self.runs.append(command)
        if "sysconfig" in command:
            return Result(exit_code=0, stdout=self._purelib, stderr="")
        if "tar xzf" in command:
            return Result(exit_code=self._extract_code, stdout="", stderr="boom" if self._extract_code else "")
        return Result(exit_code=0, stdout="", stderr="")

    async def forward(self, port: int) -> int:
        return port


def _node(
    user_code: bytes | None,
    *,
    user: str = "root",
    ssh: FakeSsh | None = None,
    options: Options | None = None,
    instance_timeout: int | None = None,
) -> tuple[Node, FakeSsh]:
    node = Node(
        Machine(id="mch", state="running", host="127.0.0.1", user=user),
        compute="cmp",
        private_key="",
        image=Image(),
        source=Source(argument="skyward"),
        listener=lambda *_: None,
        output=lambda *_: None,
        sample=lambda *_: None,
        phase=lambda *_: None,
        user_code=user_code,
        options=options or Options(),
        instance_timeout=instance_timeout,
    )
    fake = ssh or FakeSsh()
    node._ssh = fake  # type: ignore[assignment]
    return node, fake


async def test_without_user_code_nothing_is_sent():
    node, fake = _node(None)
    await node._sync_user_code()
    assert fake.puts == [] and fake.runs == []


async def test_the_tarball_is_put_and_unpacked_into_site_packages():
    node, fake = _node(b"tarbytes")
    await node._sync_user_code()

    assert fake.puts == [("/tmp/_user_code.tar.gz", b"tarbytes")]
    assert any(f"tar xzf /tmp/_user_code.tar.gz -C {PURELIB}" in command for command in fake.runs)


async def test_a_non_root_user_gets_sudo_for_the_extract():
    node, fake = _node(b"tarbytes", user="ubuntu")
    await node._sync_user_code()
    assert any(command.startswith("sudo tar xzf") for command in fake.runs)


async def test_a_failed_extract_fails_the_node():
    node, _ = _node(b"tarbytes", ssh=FakeSsh(extract_code=2))
    with pytest.raises(BootstrapFailedError, match="user code"):
        await node._sync_user_code()


async def test_an_undiscoverable_site_packages_fails_the_node():
    node, _ = _node(b"tarbytes", ssh=FakeSsh(purelib=""))
    with pytest.raises(BootstrapFailedError, match="site-packages"):
        await node._sync_user_code()


def test_the_extract_uses_the_bootstrap_python():
    """A guard that the query targets the venv the worker actually runs from."""
    assert bootstrap.PYTHON.endswith("/.venv/bin/python")


async def test_the_remote_health_predicate_is_uploaded_with_its_limits() -> None:
    node, fake = _node(
        None,
        options=Options(
            health_function=b"predicate",
            health_interval=4.0,
            health_timeout=1.5,
            health_failures=5,
            health_initial_delay=2.0,
        ),
    )

    environment = await node._health_environment()

    assert fake.puts == [("/opt/skyward/health.bin", b"predicate")]
    assert environment == {
        "SKYWARD_HEALTH": "/opt/skyward/health.bin",
        "SKYWARD_HEALTH_INTERVAL": "4.0",
        "SKYWARD_HEALTH_TIMEOUT": "1.5",
        "SKYWARD_HEALTH_FAILURES": "5",
        "SKYWARD_HEALTH_INITIAL_DELAY": "2.0",
    }


@pytest.mark.parametrize(("cluster", "value"), [(True, "1"), (False, "0")])
async def test_cluster_mode_reaches_the_worker_environment(cluster: bool, value: str) -> None:
    node, fake = _node(None, options=Options(cluster=cluster))
    node._settle("worker", None)

    await node._launch()

    assert any(f"SKYWARD_CLUSTER={value}" in command for command in fake.runs)


async def test_provider_instance_timeout_arms_remote_shutdown() -> None:
    node, fake = _node(None, user="ubuntu", instance_timeout=45)

    await node._arm_timeout()

    assert fake.runs == ["nohup sh -c 'sleep 45; sudo shutdown -h now' > /dev/null 2>&1 &"]
