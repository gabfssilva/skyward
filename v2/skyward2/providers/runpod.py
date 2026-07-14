import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.application.provider import Binding, Machine
from skyward2.protocol.schemas import ComputeSpec, Market, Offer

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"

GPU_TYPES_QUERY = """
query GpuTypes($secureCloud: Boolean) {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    maxGpuCount
    maxGpuCountSecureCloud
    maxGpuCountCommunityCloud
    lowestPrice(input: {gpuCount: 1, secureCloud: $secureCloud}) {
      minimumBidPrice
      uninterruptablePrice
      stockStatus
      totalCount
      rentedCount
      minVcpu
      minMemory
    }
  }
}
"""

CLOUDS: tuple[tuple[str, bool], ...] = (("SECURE", True), ("COMMUNITY", False))

DEPLOY_ON_DEMAND = """
mutation DeployOnDemand($input: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $input) { id }
}
"""

DEPLOY_SPOT = """
mutation DeploySpot($input: PodRentInterruptableInput!) {
  podRentInterruptable(input: $input) { id }
}
"""

DEFAULT_IMAGE = "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04"
DEFAULT_DISK_GB = 50
PORTS = "22/tcp"

ENTRYPOINT = (
    "{ [ -x /usr/sbin/sshd ] || ("
    "apt-get -o DPkg::Lock::Timeout=-1 update && "
    "DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=-1 "
    "install -y --no-install-recommends openssh-server"
    "); "
    "mkdir -p /run/sshd ~/.ssh; "
    'echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys; '
    "chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys; "
    "ssh-keygen -A; "
    'sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config; '
    "/usr/sbin/sshd; }; sleep infinity"
)


class RunPodProvider:
    """RunPod — a two-tier GPU cloud: SECURE (datacenter fleet) and COMMUNITY (peer hosts).

    Its catalog is a small fixed list of GPU types, but ``lowestPrice`` tracks live
    host supply and interruptible bids, so the TTL is minutes rather than hours.
    """

    kind: ClassVar[str] = "runpod"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=10)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("runpod requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _graphql(
        self, client: httpx.AsyncClient, query: str, variables: Mapping[str, Any],
    ) -> dict[str, Any]:
        response = await client.post(
            GRAPHQL_URL,
            json={"query": query, "variables": dict(variables)},
            headers=self._headers,
        )
        response.raise_for_status()
        payload = response.json()
        if errors := payload.get("errors"):
            raise RuntimeError(f"runpod graphql error: {errors}")
        return payload.get("data") or {}

    async def _gpu_types(self, client: httpx.AsyncClient, secure: bool) -> list[dict[str, Any]]:
        data = await self._graphql(client, GPU_TYPES_QUERY, {"secureCloud": secure})
        return data.get("gpuTypes") or []

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(timeout=30) as client:
            catalogs = await asyncio.gather(*(self._gpu_types(client, secure) for _, secure in CLOUDS))

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for (cloud, secure), gpu_types in zip(CLOUDS, catalogs, strict=True):
            for gpu in gpu_types:
                if not gpu.get("secureCloud" if secure else "communityCloud"):
                    continue

                price = gpu.get("lowestPrice") or {}
                bid = price.get("minimumBidPrice")
                on_demand = price.get("uninterruptablePrice")
                if bid is None and on_demand is None:
                    continue

                gpu_id = str(gpu["id"])
                display = str(gpu.get("displayName") or gpu_id)
                vram_gb = int(gpu.get("memoryInGb") or 0)
                vcpus = int(price.get("minVcpu") or 0)
                memory_gb = float(price.get("minMemory") or 0)
                total = price.get("totalCount")
                rented = int(price.get("rentedCount") or 0)
                available = max(int(total) - rented, 0) if total else None

                max_count = int(
                    gpu.get("maxGpuCountSecureCloud" if secure else "maxGpuCountCommunityCloud")
                    or gpu.get("maxGpuCount")
                    or 1
                )

                for count in range(1, max_count + 1):
                    yield Offer(
                        id=f"{gpu_id}:{cloud}:{count}x",
                        provider_id=self._id,
                        provider_name=self._name,
                        kind=self.kind,
                        instance_type=f"{gpu_id}:{cloud}",
                        accelerator=display,
                        accelerator_count=count,
                        vram=float(vram_gb) or None,
                        cpus=vcpus * count,
                        memory_gb=memory_gb * count,
                        spot_price=round(bid * count, 4) if bid is not None else None,
                        on_demand_price=round(on_demand * count, 4) if on_demand is not None else None,
                        available=available,
                        fetched_at=now,
                        expires_at=expires_at,
                        specific={
                            "gpu_type_id": gpu_id,
                            "cloud_type": cloud,
                            "gpu_display_name": display,
                            "gpu_memory_gb": vram_gb,
                            "stock_status": price.get("stockStatus"),
                        },
                    )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """Resolve the pod recipe. RunPod has nothing to create up front.

        There is no keypair resource, no network and no security group: the public
        key travels as a ``PUBLIC_KEY`` env var and the entrypoint appends it to
        ``authorized_keys`` on first boot. Everything here is therefore a decision
        rather than an allocation, which is what makes it trivially idempotent —
        and what makes :meth:`release` a no-op.
        """
        gpu_count = offer.accelerator_count
        if market == "spot" and not offer.spot_price:
            raise CapabilityMismatchError(
                f"{offer.id} was picked on the spot market and carries no bid price", provider=self._name,
            )

        data_center_id = self._config.get("data_center_id") or next(
            (s.region for s in spec.specs if s.provider.kind == self.kind and s.region), None,
        )

        return {
            "compute_id": compute_id,
            "prefix": f"skyward-{compute_id}-",
            "image": spec.image.base or DEFAULT_IMAGE,
            "public_key": public_key,
            "gpu_type_id": offer.specific["gpu_type_id"],
            "gpu_count": gpu_count,
            "cloud_type": offer.specific["cloud_type"],
            "data_center_id": str(data_center_id) if data_center_id else None,
            "container_disk_gb": int(self._config.get("container_disk_gb", DEFAULT_DISK_GB)),
            "market": market,
            "bid_per_gpu": round(offer.spot_price / gpu_count, 4) if market == "spot" and offer.spot_price else None,
        }

    async def launch(self, binding: Binding, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        async with httpx.AsyncClient(timeout=120) as client, asyncio.TaskGroup() as group:
            attempts = [group.create_task(self._deploy(client, binding)) for _ in range(count)]

        results = [task.result() for task in attempts]
        machines = tuple(result for result in results if isinstance(result, Machine))
        if len(machines) < min_count:
            raise ExceptionGroup(
                f"runpod deployed {len(machines)} pods, {min_count} were required",
                [result for result in results if isinstance(result, Exception)],
            )
        return binding, machines

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """Find the compute's pods by name.

        RunPod has no tags or labels on a pod: the name is the only field an
        adapter controls and can filter an account-wide listing on, so the
        compute id is carried in it.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{REST_URL}/pods", headers=self._headers)
            response.raise_for_status()
            pods = response.json() or []

        found = (
            _machine(pod)
            for pod in pods
            if str(pod.get("name") or "").startswith(binding["prefix"])
        )
        return {machine.id: machine for machine in found if machine is not None}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        if not machine_ids:
            return

        async with httpx.AsyncClient(timeout=60) as client, asyncio.TaskGroup() as group:
            for machine_id in machine_ids:
                group.create_task(self._destroy(client, machine_id))

    async def release(self, binding: Binding) -> None:
        return None

    async def _deploy(self, client: httpx.AsyncClient, binding: Binding) -> Machine | Exception:
        """Deploy one pod, and hand a failure back rather than raise it.

        A launch of ``count`` pods is allowed to come back with fewer, so one host
        refusing the bid must not cancel its siblings — which is what raising
        inside the task group would do.
        """
        deploy: dict[str, Any] = {
            "name": f"{binding['prefix']}{uuid.uuid4().hex[:8]}",
            "imageName": binding["image"],
            "gpuTypeId": binding["gpu_type_id"],
            "gpuCount": binding["gpu_count"],
            "cloudType": binding["cloud_type"],
            "containerDiskInGb": binding["container_disk_gb"],
            "volumeInGb": 0,
            "ports": PORTS,
            "supportPublicIp": True,
            "dockerArgs": f"bash -c '{ENTRYPOINT}'",
            "env": [{"key": "PUBLIC_KEY", "value": binding["public_key"]}],
        }
        if data_center_id := binding["data_center_id"]:
            deploy["dataCenterId"] = data_center_id

        match binding["market"]:
            case "spot":
                mutation, key = DEPLOY_SPOT, "podRentInterruptable"
                deploy["bidPerGpu"] = binding["bid_per_gpu"]
            case _:
                mutation, key = DEPLOY_ON_DEMAND, "podFindAndDeployOnDemand"

        try:
            data = await self._graphql(client, mutation, {"input": deploy})
        except (httpx.HTTPError, RuntimeError) as error:
            return error

        match data.get(key):
            case {"id": str(pod_id)}:
                return Machine(id=pod_id, state="pending")
            case _:
                return RuntimeError(f"runpod {key} returned no pod: {data}")

    async def _destroy(self, client: httpx.AsyncClient, machine_id: str) -> None:
        response = await client.delete(f"{REST_URL}/pods/{machine_id}", headers=self._headers)
        if response.status_code != 404:
            response.raise_for_status()


def _machine(pod: Mapping[str, Any]) -> Machine | None:
    """A pod that has stopped is reported as gone.

    A stopped pod cannot be made to run again by anything the control plane does,
    and it keeps billing its disk for as long as it exists. Reporting it absent is
    what turns it into a lost node, and a lost node is terminated.
    """
    match pod.get("desiredStatus"):
        case "EXITED" | "TERMINATED":
            return None
        case _:
            host = pod.get("publicIp") or None
            ports = pod.get("portMappings") or {}
            return Machine(
                id=pod["id"],
                state="running" if host else "pending",
                host=host,
                port=int(ports.get("22", 22)),
            )
