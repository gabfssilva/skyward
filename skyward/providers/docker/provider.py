from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
from collections.abc import Sequence

import aiodocker

from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance
from skyward.observability.logger import logger
from skyward.providers.docker.config import Docker
from skyward.providers.provider import CloudProvider
from skyward.providers.ssh_keys import get_local_ssh_key, get_ssh_key_path

log = logger.bind(provider="docker")

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


def _make_entrypoint(ttl: int) -> str:
    ttl_cmd = f"(sleep {ttl} && kill 1) & " if ttl else ""
    return (
        f"{ttl_cmd}"
        "echo \"$SSH_PUB_KEY\" > /root/.ssh/authorized_keys && "
        "chmod 600 /root/.ssh/authorized_keys && "
        "/usr/sbin/sshd -D"
    )


class DockerCloudProvider(CloudProvider[Docker, str]):

    def __init__(self, config: Docker, client: aiodocker.Docker) -> None:
        self._config = config
        self._client = client

    @classmethod
    async def create(cls, config: Docker) -> DockerCloudProvider:
        return cls(config, aiodocker.Docker())

    async def _ensure_image(self) -> None:
        tag = f"skyward-ssh:{hashlib.md5(self._config.image.encode()).hexdigest()[:12]}"
        try:
            await self._client.images.inspect(tag)
            log.debug("Image {tag} already exists", tag=tag)
            self._image = tag
            return
        except aiodocker.exceptions.DockerError:
            pass

        log.info("Building SSH image from {base}", base=self._config.image)
        dockerfile = _DOCKERFILE.format(base=self._config.image)
        context = _build_tar_context(dockerfile)
        await self._client.images.build(
            fileobj=context,
            tag=tag,
            encoding="gzip",
        )
        log.info("Image {tag} built", tag=tag)
        self._image = tag

    async def prepare(self, spec: PoolSpec) -> Cluster[str]:
        cluster_id = f"skyward-{uuid.uuid4().hex[:8]}"
        ssh_key_path = get_ssh_key_path()

        await self._ensure_image()

        network = await self._client.networks.create({
            "Name": f"{_NETWORK_PREFIX}-{cluster_id}",
            "Driver": "bridge",
        })

        log.info("Cluster {id} network created: {net}", id=cluster_id, net=network.id)

        return Cluster(
            id=cluster_id,
            status="provisioning",
            spec=spec,
            ssh_key_path=ssh_key_path,
            ssh_user=self._config.ssh_user,
            use_sudo=False,
            shutdown_command="kill 1",
            specific=network.id,
        )

    async def provision(self, cluster: Cluster[str], count: int) -> Sequence[Instance]:
        _, pub_key = get_local_ssh_key()
        network_name = f"{_NETWORK_PREFIX}-{cluster.id}"
        ttl = cluster.spec.ttl or 0
        entrypoint = _make_entrypoint(ttl)

        coros = [
            self._launch_instance(entrypoint, pub_key, cluster, network_name)
            for _ in range(count)
        ]
        return await asyncio.gather(*coros)

    async def _launch_instance(
        self, entrypoint: str, pub_key: str,
        cluster: Cluster[str], network_name: str,
    ) -> Instance:
        container = await self._client.containers.run(
            config={
                "Image": self._image,
                "Cmd": ["sh", "-c", entrypoint],
                "Env": [f"SSH_PUB_KEY={pub_key}"],
                "ExposedPorts": {"22/tcp": {}},
                "HostConfig": _host_config(cluster.spec, network_name),
                "Labels": {_CLUSTER_LABEL: cluster.id},
            },
        )

        short_id = container.id[:12]

        log.info("Container {id} launched", id=short_id)

        return Instance(
            id=short_id,
            status="provisioning",
            instance_type="docker",
            vcpus=cluster.spec.vcpus or 1,
            memory_gb=cluster.spec.memory_gb or 1,
        )


    async def get_instance(self, cluster: Cluster[str], instance_id: str) -> Instance | None:
        try:
            container = self._client.containers.container(instance_id)
            info = await container.show()
        except aiodocker.exceptions.DockerError:
            return None

        if not info["State"]["Running"]:
            return None

        networks = info["NetworkSettings"]["Networks"]
        private_ip = next(iter(networks.values()))["IPAddress"] if networks else None

        ports = info["NetworkSettings"]["Ports"].get("22/tcp")
        ssh_port = int(ports[0]["HostPort"]) if ports else 22

        return Instance(
            id=instance_id,
            status="provisioned",
            ip="127.0.0.1",
            private_ip=private_ip,
            ssh_port=ssh_port,
            instance_type="docker",
        )

    async def terminate(self, instance_ids: tuple[str, ...]) -> None:
        for iid in instance_ids:
            container = self._client.containers.container(iid)
            with _IgnoreNotFound():
                await container.kill()
            with _IgnoreNotFound():
                await container.delete(force=True)
            log.info("Container {id} terminated", id=iid)

    async def teardown(self, cluster: Cluster[str]) -> None:
        containers = await self._client.containers.list(
            filters={"label": [f"{_CLUSTER_LABEL}={cluster.id}"]},
        )
        for c in containers:
            with _IgnoreNotFound():
                await c.kill()
            with _IgnoreNotFound():
                await c.delete(force=True)

        with _IgnoreNotFound():
            network = await self._client.networks.get(cluster.specific)
            await network.delete()

        log.info("Cluster {id} torn down", id=cluster.id)


def _build_tar_context(dockerfile: str) -> io.BytesIO:
    import gzip
    import tarfile

    buf = io.BytesIO()
    with gzip.open(buf, "wb") as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        data = dockerfile.encode()
        info = tarfile.TarInfo(name="Dockerfile")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def _host_config(spec: PoolSpec, network_name: str) -> dict:
    config: dict = {
        "PublishAllPorts": True,
        "NetworkMode": network_name,
    }
    if spec.vcpus:
        config["NanoCpus"] = int(spec.vcpus * 1e9)
    if spec.memory_gb:
        config["Memory"] = int(spec.memory_gb * 1024 * 1024 * 1024)
    return config


class _IgnoreNotFound:
    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: object) -> bool:
        return bool(
            exc_type is not None
            and issubclass(exc_type, aiodocker.exceptions.DockerError)
        )
