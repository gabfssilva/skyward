"""Tests for SSH reconnection after transient connection loss.

The container uses a respawning sshd entrypoint so that killing sshd
doesn't kill the container.  When sshd is killed, all SSH connections
break.  The node actor should detect the connection error on the next
task, call ``transport.reconnect()``, and resume — all without replacing
the node.  The monitor should also reconnect its event stream.
"""

from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace

import pytest

import skyward as sky
from skyward.api.model import Cluster, Instance, Offer
from skyward.api.spec import PoolSpec
from skyward.providers.container.provider import ContainerProvider, ContainerSpecific
from skyward.providers.ssh_keys import get_local_ssh_key

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.integration,
    pytest.mark.timeout(180),
    pytest.mark.xdist_group("pool"),
]

_PREFIX = "skyward-ssh-reconnect"


def _respawning_entrypoint(ttl: int) -> str:
    """Entrypoint that wraps sshd in a respawn loop.

    PID 1 is the shell loop, not sshd.  Killing sshd causes it to restart
    after 1 second without killing the container.
    """
    ttl_cmd = f"(sleep {ttl} && kill 1) & " if ttl else ""
    return (
        f"{ttl_cmd}"
        'echo "$SSH_PUB_KEY" > /root/.ssh/authorized_keys && '
        "chmod 600 /root/.ssh/authorized_keys && "
        "while true; do /usr/sbin/sshd -D; sleep 1; done"
    )


class _SSHDisruptor:
    """Kills sshd inside the container, breaking all SSH connections.

    Because the container uses a respawning entrypoint, sshd restarts
    automatically after ~1 second.
    """

    def __init__(self, binary: str, container_prefix: str) -> None:
        self._bin = binary
        self._prefix = container_prefix
        self._lock = threading.Lock()
        self._instance_ids: list[str] = []

    def register(self, instance_id: str) -> None:
        with self._lock:
            self._instance_ids.append(instance_id)

    def kill_sshd(self, index: int = 0) -> None:
        with self._lock:
            iid = self._instance_ids[index]
        name = f"{self._prefix}-{iid}"
        subprocess.run(
            [self._bin, "exec", name, "pkill", "-x", "sshd"],
            check=False,
            capture_output=True,
        )


class _SSHDisruptorProvider:
    """Provider wrapper that uses a respawning sshd entrypoint."""

    def __init__(self, inner: ContainerProvider, disruptor: _SSHDisruptor) -> None:
        self._inner = inner
        self._disruptor = disruptor

    @property
    def name(self) -> str:
        return self._inner.name

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        async for offer in self._inner.offers(spec):
            yield offer

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[ContainerSpecific]:
        return await self._inner.prepare(spec, offer)

    async def provision(
        self, cluster: Cluster[ContainerSpecific], count: int,
    ) -> tuple[Cluster[ContainerSpecific], Sequence[Instance]]:
        _, pub_key = get_local_ssh_key()
        ttl = cluster.spec.ttl or 0
        entrypoint = _respawning_entrypoint(ttl)

        coros = [
            self._inner._launch_instance(entrypoint, pub_key, cluster)
            for _ in range(count)
        ]
        instances = sorted(await asyncio.gather(*coros), key=lambda i: i.id)
        for inst in instances:
            self._disruptor.register(inst.id)
        return replace(cluster, instances=tuple(instances)), instances

    async def get_instance(
        self, cluster: Cluster[ContainerSpecific], instance_id: str,
    ) -> tuple[Cluster[ContainerSpecific], Instance | None]:
        return await self._inner.get_instance(cluster, instance_id)

    async def terminate(
        self, cluster: Cluster[ContainerSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[ContainerSpecific]:
        return await self._inner.terminate(cluster, instance_ids)

    async def teardown(
        self, cluster: Cluster[ContainerSpecific],
    ) -> Cluster[ContainerSpecific]:
        return await self._inner.teardown(cluster)


@dataclass(frozen=True, slots=True)
class _SSHDisruptorContainer(sky.Container):
    disruptor: _SSHDisruptor | None = field(default=None, hash=False)

    async def create_provider(self) -> _SSHDisruptorProvider:  # type: ignore[override]
        inner = await ContainerProvider.create(self)
        d = self.disruptor or _SSHDisruptor(
            self.binary, self.container_prefix or "skyward",
        )
        return _SSHDisruptorProvider(inner, d)


class TestSSHReconnect:
    def test_pool_recovers_after_sshd_restart(self) -> None:
        """Kill sshd, verify node reconnects transport without replacement."""
        disruptor = _SSHDisruptor(binary="docker", container_prefix=_PREFIX)

        with sky.App(console=False), sky.ComputePool(
            provider=_SSHDisruptorContainer(
                network="skyward",
                container_prefix=_PREFIX,
                disruptor=disruptor,
            ),
            nodes=1,
            vcpus=0.5,
            memory_gb=0.5,
            worker=sky.Worker(concurrency=2),
            default_compute_timeout=30,
        ) as pool:

            @sky.function
            def add(a: int, b: int) -> int:
                return a + b

            # 1) Pool works normally
            assert add(1, 2) >> pool == 3

            # 2) Kill sshd — breaks all SSH connections.
            #    Respawning entrypoint restarts sshd in ~1s.
            disruptor.kill_sshd(0)

            # 3) Wait for sshd to restart, then retry until the node
            #    detects the connection error and reconnects transport.
            time.sleep(3)

            deadline = time.monotonic() + 60
            result = None
            while time.monotonic() < deadline:
                try:
                    result = add(10, 20) >> pool
                    break
                except (RuntimeError, TimeoutError):
                    time.sleep(2)
            else:
                pytest.fail("Pool did not recover after sshd restart")

            assert result == 30
