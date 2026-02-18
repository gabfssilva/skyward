from __future__ import annotations

import asyncio
import hashlib
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance
from skyward.observability.logger import logger
from skyward.providers.container.cli import run, run_json
from skyward.providers.container.config import Container
from skyward.providers.provider import CloudProvider
from skyward.providers.ssh_keys import get_local_ssh_key, get_ssh_key_path

log = logger.bind(provider="container")

_NETWORK_PREFIX = "skyward"
_CLUSTER_LABEL = "skyward.cluster"
_DOCKERFILE = (
    "FROM {base}\n"
    "RUN apt-get update -qq && "
    "apt-get install -y -qq openssh-server > /dev/null 2>&1 && "
    "mkdir -p /run/sshd /root/.ssh && "
    "chmod 700 /root/.ssh && "
    "rm -rf /var/lib/apt/lists/*\n"
    "EXPOSE 22\n"
)

_CONTAINER_PREFIX = "skyward"


def _make_entrypoint(ttl: int) -> str:
    ttl_cmd = f"(sleep {ttl} && kill 1) & " if ttl else ""
    return (
        f"{ttl_cmd}"
        "echo \"$SSH_PUB_KEY\" > /root/.ssh/authorized_keys && "
        "chmod 600 /root/.ssh/authorized_keys && "
        "/usr/sbin/sshd -D"
    )


def _is_apple_container(binary: str) -> bool:
    return binary == "container"

@dataclass(frozen=True, slots=True)
class ContainerSpecific:
    network: str
    image: str
    context: str


class ContainerProvider(CloudProvider[Container, ContainerSpecific]):

    def __init__(self, config: Container) -> None:
        self._config = config
        self._bin = config.binary
        self._image: str = ""

    @classmethod
    async def create(cls, config: Container) -> ContainerProvider:
        return cls(config)

    async def _ensure_image(self) -> str:
        tag = f"skyward-ssh:{hashlib.md5(self._config.image.encode()).hexdigest()[:12]}"
        try:
            await run(self._bin, "image", "inspect", tag)
            log.debug("Image {tag} already exists", tag=tag)
            return tag
        except RuntimeError:
            pass

        log.info("Building SSH image from {base}", base=self._config.image)
        dockerfile = _DOCKERFILE.format(base=self._config.image)

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "Dockerfile").write_text(dockerfile)
            await run(self._bin, "build", "-t", tag, tmpdir)

        log.info("Image {tag} built", tag=tag)
        return tag

    async def prepare(self, spec: PoolSpec) -> Cluster[ContainerSpecific]:
        cluster_id = f"skyward-{uuid.uuid4().hex[:8]}"
        ssh_key_path = get_ssh_key_path()

        tag = await self._ensure_image()

        network_name = f"{_NETWORK_PREFIX}-{cluster_id}"
        await run(self._bin, "network", "create", network_name)

        log.info("Cluster {id} network created: {net}", id=cluster_id, net=network_name)

        return Cluster(
            id=cluster_id,
            status="provisioning",
            spec=spec,
            ssh_key_path=ssh_key_path,
            ssh_user=self._config.ssh_user,
            use_sudo=False,
            shutdown_command="kill 1",
            specific=ContainerSpecific(
                network=network_name,
                image=tag,
                context=await run(self._config.binary, 'context', 'show'),
            ),
        )

    async def provision(
        self, cluster: Cluster[ContainerSpecific], count: int,
    ) -> Sequence[Instance]:
        _, pub_key = get_local_ssh_key()
        ttl = cluster.spec.ttl or 0
        entrypoint = _make_entrypoint(ttl)

        coros = [
            self._launch_instance(entrypoint, pub_key, cluster)
            for _ in range(count)
        ]
        return await asyncio.gather(*coros)

    async def _launch_instance(
        self, entrypoint: str, pub_key: str,
        cluster: Cluster[ContainerSpecific],
    ) -> Instance:
        short_id = uuid.uuid4().hex[:12]
        container_name = f"{_CONTAINER_PREFIX}-{short_id}"

        cmd: list[str] = [
            self._bin, "run", "-d",
            "--name", container_name,
            "-e", f"SSH_PUB_KEY={pub_key}",
            "-p", "0:22",
            "--network", cluster.specific.network,
            "-l", f"{_CLUSTER_LABEL}={cluster.id}",
        ]

        vcpus = cluster.spec.vcpus or 0.5
        memory_gb = cluster.spec.memory_gb or 0.5
        cmd.extend(["--cpus", str(vcpus)])
        if _is_apple_container(self._bin):
            cmd.extend(["--memory", f"{int(memory_gb * 1024)}M"])
        else:
            cmd.extend(["--memory", f"{memory_gb}g"])

        cmd.extend([cluster.specific.image, "sh", "-c", entrypoint])

        await run(*cmd)

        log.info("Container {id} launched", id=container_name)

        return Instance(
            id=short_id,
            status="provisioning",
            instance_type=cluster.specific.context,
            vcpus=vcpus,
            memory_gb=memory_gb,
        )

    async def get_instance(
        self, cluster: Cluster[ContainerSpecific], instance_id: str,
    ) -> Instance | None:
        container_name = f"{_CONTAINER_PREFIX}-{instance_id}"
        try:
            info = await run_json(self._bin, "inspect", container_name)
        except RuntimeError:
            return None

        data = info[0] if isinstance(info, list) else info

        running, private_ip, ssh_port = _parse_inspect(data, self._bin)
        if not running:
            return None

        return Instance(
            id=instance_id,
            status="provisioned",
            ip="127.0.0.1",
            private_ip=private_ip,
            ssh_port=ssh_port,
            instance_type="container",
        )

    async def terminate(self, instance_ids: tuple[str, ...]) -> None:
        import asyncio

        async def _kill(iid: str) -> None:
            container_name = f"{_CONTAINER_PREFIX}-{iid}"
            await _stop_and_remove(self._bin, container_name)
            log.info("Container {id} terminated", id=iid)

        await asyncio.gather(*(_kill(iid) for iid in instance_ids))

    async def teardown(self, cluster: Cluster[ContainerSpecific]) -> None:
        import asyncio

        ids = await _list_cluster_containers(self._bin, cluster.id)
        await asyncio.gather(*(_stop_and_remove(self._bin, cid) for cid in ids))

        try:
            await run(self._bin, "network", "rm", cluster.specific.network)
        except RuntimeError as e:
            log.warning("Failed to remove network {net}: {err}", net=cluster.specific, err=e)

        log.info("Cluster {id} torn down", id=cluster.id)


def _parse_inspect(data: dict, binary: str) -> tuple[bool, str | None, int]:
    if _is_apple_container(binary):
        running = data.get("status") == "running"
        net_list = data.get("networks", [])
        private_ip = net_list[0]["ipv4Address"].split("/")[0] if net_list else None
        published = data.get("configuration", {}).get("publishedPorts", [])
        ssh_port = next(
            (p["hostPort"] for p in published if p.get("containerPort") == 22),
            22,
        )
    else:
        running = data.get("State", {}).get("Running", False)
        networks = data.get("NetworkSettings", {}).get("Networks", {})
        private_ip = next(iter(networks.values()))["IPAddress"] if networks else None
        ports = data.get("NetworkSettings", {}).get("Ports", {}).get("22/tcp")
        ssh_port = int(ports[0]["HostPort"]) if ports else 22
    return running, private_ip, ssh_port


async def _stop_and_remove(binary: str, container_id: str) -> None:
    if _is_apple_container(binary):
        try:
            await run(binary, "kill", container_id)
        except RuntimeError as e:
            log.warning("Failed to kill container {id}: {err}", id=container_id, err=e)
        try:
            await run(binary, "rm", "-f", container_id)
        except RuntimeError as e:
            log.warning("Failed to remove container {id}: {err}", id=container_id, err=e)
    else:
        try:
            await run(binary, "rm", "-f", container_id)
        except RuntimeError as e:
            log.warning("Failed to remove container {id}: {err}", id=container_id, err=e)


async def _list_cluster_containers(binary: str, cluster_id: str) -> list[str]:
    if _is_apple_container(binary):
        try:
            raw = await run(
                binary, "ps", "-a",
                "--filter", f"label={_CLUSTER_LABEL}={cluster_id}",
                "-q",
            )
            return [c.strip() for c in raw.splitlines() if c.strip()]
        except RuntimeError:
            return []

    try:
        containers = await run_json(binary, "list", "--all", "--format", "json")
    except RuntimeError:
        return []

    if not isinstance(containers, list):
        return []

    return [
        c.get("configuration", {}).get("id", c.get("id", ""))
        for c in containers
        if c.get("configuration", {}).get("labels", {}).get(_CLUSTER_LABEL) == cluster_id
    ]
