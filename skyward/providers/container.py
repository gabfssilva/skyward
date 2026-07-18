import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

from skyward.application.errors import CapabilityMismatchError
from skyward.application.provider import Binding, Machine
from skyward.protocol.schemas import ComputeSpec, Market, Offer

COMPUTE_LABEL = "skyward.compute"
BASE_IMAGE = "ghcr.io/gabfssilva/skyward:py{python}"
WARM_IMAGE = "skyward-warm:{tag}"
APPLE = "container"
DEFAULT_PYTHON = "3.13"
CPUS = (1, 2, 4, 8, 16)
MEMORY_GB = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

ENTRYPOINT = (
    'echo "$SSH_PUB_KEY" > /root/.ssh/authorized_keys && '
    "chmod 600 /root/.ssh/authorized_keys && "
    "/usr/sbin/sshd -D"
)


class ContainerProvider:
    """The local machine, pretending to be a cloud, one container per node.

    The only provider whose machines cost nothing, which is what makes it the one
    place the provisioning half of the contract can be exercised in a test. It is
    not a simulation: the containers run sshd, the control plane connects to them
    over SSH and bootstraps them exactly as it would an EC2 instance.

    It offers CPU only, at price zero. That price never distorts a comparison,
    because a compute names the providers it may draw offers from — the container
    competes only when it was asked for.
    """

    kind: ClassVar[str] = "container"
    credential_fields: ClassVar[tuple[str, ...]] = ()
    offers_ttl: ClassVar[timedelta] = timedelta(days=1)

    def __init__(self, provider_id: str, name: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._binary = str(config.get("binary", "docker"))

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        return cls(provider_id, name, config)

    async def offers(self) -> AsyncIterator[Offer]:
        now = datetime.now(UTC)
        for cpus in CPUS:
            for memory_gb in MEMORY_GB:
                yield Offer(
                    id=f"container-{cpus}c-{memory_gb:g}g",
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    billing_unit="second",
                    instance_type=f"container.{cpus}c{memory_gb:g}g",
                    accelerator_count=0,
                    cpus=cpus,
                    memory_gb=memory_gb,
                    on_demand_price=0.0,
                    fetched_at=now,
                    expires_at=now + self.offers_ttl,
                )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        image = BASE_IMAGE.format(python=spec.image.python or DEFAULT_PYTHON)
        network = f"skyward-{compute_id}"

        await self._pull(image)
        await self._network(network)

        return {
            "compute_id": compute_id,
            "network": network,
            "image": image,
            "public_key": public_key,
            "cpus": offer.cpus,
            "memory_gb": offer.memory_gb,
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        async with asyncio.TaskGroup() as group:
            started = [group.create_task(self._start(binding)) for _ in range(count)]

        return binding, tuple(Machine(id=task.result(), state="pending") for task in started)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        listed = await self._cli(
            "ps", "-a", "--filter", f"label={COMPUTE_LABEL}={binding['compute_id']}", "--format", "{{.ID}}",
        )
        if not (ids := listed.split()):
            return {}

        inspected = json.loads(await self._cli("inspect", *ids))
        machines = (_machine(data) for data in inspected)
        return {machine.id: machine for machine in machines if machine is not None}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        async with asyncio.TaskGroup() as group:
            for machine_id in machine_ids:
                group.create_task(self._try("rm", "-f", machine_id))

    async def release(self, binding: Binding) -> None:
        await self._try("network", "rm", binding["network"])

    async def bake(self, binding: Binding, machine_id: str, tag: str) -> str:
        """Commit the container as it stands, bootstrapped venv and heavy wheels and all.

        Apple's ``container`` has no commit verb, so on it there is nothing to keep and
        saying so is the only honest answer.
        """
        if self._binary == APPLE:
            raise CapabilityMismatchError("the apple container runtime cannot commit a running container", provider=self._name)

        warm = WARM_IMAGE.format(tag=tag)
        await self._cli("commit", machine_id, warm)
        return warm

    async def baked(self, binding: Binding, tag: str) -> str | None:
        warm = WARM_IMAGE.format(tag=tag)
        return warm if await self._try("image", "inspect", warm) is not None else None

    async def _start(self, binding: Binding) -> str:
        created = await self._cli(
            "run", "-d",
            "--name", f"skyward-{uuid.uuid4().hex[:12]}",
            "--label", f"{COMPUTE_LABEL}={binding['compute_id']}",
            "--network", binding["network"],
            "--env", f"SSH_PUB_KEY={binding['public_key']}",
            "--publish", "0:22",
            "--cpus", str(binding["cpus"]),
            "--memory", f"{binding['memory_gb']:g}g",
            binding["image"],
            "sh", "-c", ENTRYPOINT,
        )
        return created[:12]

    async def _pull(self, image: str) -> None:
        if await self._try("image", "inspect", image) is None:
            await self._cli("pull", image)

    async def _network(self, name: str) -> None:
        if await self._try("network", "inspect", name) is None:
            await self._try("network", "create", name)

    async def _cli(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            self._binary, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"{self._binary} {' '.join(args)}: {stderr.decode().strip()}")
        return stdout.decode().strip()

    async def _try(self, *args: str) -> str | None:
        """Run the binary, and read a failure as an answer rather than as an error.

        "Already exists" and "already gone" are the two failures this adapter
        expects, and both of them are successes: every call has to survive being
        made a second time by a daemon that restarted mid-reconcile.
        """
        try:
            return await self._cli(*args)
        except RuntimeError:
            return None


def _machine(data: dict[str, Any]) -> Machine | None:
    if not data["State"]["Running"]:
        return None

    network = next(iter(data["NetworkSettings"]["Networks"].values()), {})
    published = data["NetworkSettings"]["Ports"].get("22/tcp")

    return Machine(
        id=data["Id"][:12],
        state="running",
        host="127.0.0.1",
        port=int(published[0]["HostPort"]) if published else 22,
        private_host=network.get("IPAddress") or None,
    )
