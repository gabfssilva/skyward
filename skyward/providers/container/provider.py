from __future__ import annotations

import asyncio
import hashlib
import tempfile
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.container.cli import run, run_json
from skyward.providers.container.config import Container
from skyward.providers.provider import WarmableProvider
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
    base_image: str
    context: str


class ContainerProvider(WarmableProvider[Container, ContainerSpecific]):

    def __init__(self, config: Container) -> None:
        self._config = config
        self._bin = config.binary
        self._container_prefix = config.container_prefix or _CONTAINER_PREFIX
        self._shared_network = config.network

    def _resolve_image(self, python_version: str) -> str:
        return self._config.image.format(python_version=python_version)

    @classmethod
    async def create(cls, config: Container) -> ContainerProvider:
        return cls(config)

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        it = InstanceType(
            name="container",
            accelerator=None,
            vcpus=spec.vcpus or 0.5,
            memory_gb=spec.memory_gb or 0.5,
            architecture="x86_64",
            specific=None,
        )
        yield Offer(
            id="container-local",
            instance_type=it,
            spot_price=None,
            on_demand_price=0.0,
            billing_unit="second",
            specific=None,
        )

    async def _is_prebaked(self, pool: PoolSpec) -> bool:
        try:
            await run(self._bin, "image", "inspect", self._image_name(pool))
            log.debug("Image {tag} already exists", tag=self._image_name(pool))
            return True
        except RuntimeError:
            return False

    def _image_name(self, pool: PoolSpec) -> str:
        image = self._resolve_image(pool.image.python)
        return f"{image}-{pool.image.content_hash()}"

    async def _ensure_base_image(self, image: str) -> str:
        if image.startswith("ghcr.io/gabfssilva/skyward:"):
            try:
                await run(self._bin, "image", "inspect", image)
            except RuntimeError:
                log.info("Pulling {image}", image=image)
                await run(self._bin, "pull", image)
            return image

        tag = f"skyward-ssh:{hashlib.md5(image.encode()).hexdigest()[:12]}"
        try:
            await run(self._bin, "image", "inspect", tag)
            log.debug("Image {tag} already exists", tag=tag)
            return tag
        except RuntimeError:
            pass

        log.info("Building SSH image from {base}", base=image)
        dockerfile = _DOCKERFILE.format(base=image)

        with tempfile.TemporaryDirectory() as tmpdir:
            await asyncio.to_thread(Path(tmpdir, "Dockerfile").write_text, dockerfile)
            await run(self._bin, "build", "-t", tag, tmpdir)

        log.info("Image {tag} built", tag=tag)
        return tag

    async def _ensure_network(self, cluster_id: str) -> str:
        if self._shared_network:
            try:
                await run(
                    self._bin, "network", "inspect", self._shared_network,
                )
            except RuntimeError:
                try:
                    await run(
                        self._bin, "network", "create", self._shared_network,
                    )
                    log.info("Shared network created: {net}", net=self._shared_network)
                except RuntimeError as exc:
                    if "already exists" not in str(exc):
                        raise
            return self._shared_network

        network_name = f"{_NETWORK_PREFIX}-{cluster_id}"
        await run(self._bin, "network", "create", network_name)
        log.info(
            "Cluster {id} network created: {net}",
            id=cluster_id, net=network_name,
        )
        return network_name

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[ContainerSpecific]:
        cluster_id = f"skyward-{uuid.uuid4().hex[:8]}"
        ssh_key_path = get_ssh_key_path()

        image = self._resolve_image(spec.image.python)
        base_image = await self._ensure_base_image(image)
        network_name = await self._ensure_network(cluster_id)
        prebaked = await self._is_prebaked(spec)

        if prebaked:
            log.info("Image is prebaked with tag={tag}", tag=self._image_name(spec))

        return Cluster(
            id=cluster_id,
            status="provisioning",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user=self._config.ssh_user,
            use_sudo=False,
            shutdown_command="kill 1",
            specific=ContainerSpecific(
                network=network_name,
                base_image=base_image,
                context=await run(self._config.binary, 'context', 'show'),
            ),
            prebaked=prebaked,
        )

    async def provision(
        self, cluster: Cluster[ContainerSpecific], count: int,
    ) -> tuple[Cluster[ContainerSpecific], Sequence[Instance]]:
        _, pub_key = get_local_ssh_key()
        ttl = cluster.spec.ttl or 0
        entrypoint = _make_entrypoint(ttl)

        coros = [
            self._launch_instance(entrypoint, pub_key, cluster)
            for _ in range(count)
        ]
        instances = sorted(await asyncio.gather(*coros), key=lambda i: i.id)
        return replace(cluster, instances=tuple(instances)), instances

    async def _launch_instance(
        self, entrypoint: str, pub_key: str,
        cluster: Cluster[ContainerSpecific],
    ) -> Instance:
        short_id = uuid.uuid4().hex[:12]
        container_name = f"{self._container_prefix}-{short_id}"

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

        image = self._image_name(cluster.spec) if cluster.prebaked else cluster.specific.base_image
        cmd.extend([image, "sh", "-c", entrypoint])

        await run(*cmd)

        log.info("Container {id} launched", id=container_name)

        return Instance(
            id=short_id,
            status="provisioning",
            offer=cluster.offer,
        )

    async def get_instance(
        self, cluster: Cluster[ContainerSpecific], instance_id: str,
    ) -> tuple[Cluster[ContainerSpecific], Instance | None]:
        container_name = f"{self._container_prefix}-{instance_id}"
        try:
            info = await run_json(self._bin, "inspect", container_name)
        except RuntimeError:
            return cluster, None

        data = info[0] if isinstance(info, list) else info

        running, private_ip, ssh_port = _parse_inspect(data, self._bin)
        if not running:
            return cluster, None

        return cluster, Instance(
            id=instance_id,
            status="provisioned",
            offer=cluster.offer,
            ip="127.0.0.1",
            private_ip=private_ip,
            ssh_port=ssh_port,
        )

    async def terminate(
        self, cluster: Cluster[ContainerSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[ContainerSpecific]:
        async def _kill(iid: str) -> None:
            container_name = f"{self._container_prefix}-{iid}"
            await _stop_and_remove(self._bin, container_name)
            log.info("Container {id} terminated", id=iid)

        await asyncio.gather(*(_kill(iid) for iid in instance_ids))
        return cluster

    async def teardown(self, cluster: Cluster[ContainerSpecific]) -> Cluster[ContainerSpecific]:
        ids = await _list_cluster_containers(self._bin, cluster.id)
        await asyncio.gather(*(_stop_and_remove(self._bin, cid) for cid in ids))

        if not self._shared_network:
            try:
                await run(self._bin, "network", "rm", cluster.specific.network)
            except RuntimeError as e:
                log.warning(
                    "Failed to remove network {net}: {err}",
                    net=cluster.specific.network, err=e,
                )

        log.info("Cluster {id} torn down", id=cluster.id)
        return cluster

    async def save(self, cluster: Cluster[ContainerSpecific]) -> Cluster[ContainerSpecific]:
        if _is_apple_container(self._bin):
            return cluster

        tag = self._image_name(cluster.spec)

        try:
            await run(self._bin, "image", "inspect", tag)
            log.debug("Prebaked image {tag} already exists", tag=tag)
            return cluster
        except RuntimeError:
            pass

        if not cluster.instances:
            log.warning("No instances in cluster {id}, skipping save", id=cluster.id)
            return cluster

        container_name = f"{self._container_prefix}-{cluster.instances[0].id}"
        log.info("Saving prebaked image as {tag} from {cid}", tag=tag, cid=container_name)
        await run(self._bin, "commit", container_name, tag)
        log.info("Prebaked image {tag} saved", tag=tag)
        return cluster

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
    try:
        raw = await run(
            binary, "ps", "-a",
            "--filter", f"label={_CLUSTER_LABEL}={cluster_id}",
            "-q",
        )
        return [c.strip() for c in raw.splitlines() if c.strip()]
    except RuntimeError:
        return []
