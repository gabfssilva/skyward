"""RunPod Provider Handler - event-driven with Event Pipeline.

Uses intermediate events (InstanceLaunched, InstanceRunning) for
decoupled instance lifecycle management.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from dataclasses import field
from typing import TYPE_CHECKING, Any

from loguru import logger

from skyward.app import component, on
from skyward.bus import AsyncEventBus
from skyward.events import (
    BootstrapFailed,
    BootstrapPhase,
    BootstrapRequested,
    ClusterDestroyed,
    ClusterProvisioned,
    ClusterRequested,
    InstanceBootstrapped,
    InstanceLaunched,
    InstanceRequested,
    InstanceRunning,
    ShutdownRequested,
)
from skyward.monitors import SSHCredentialsRegistry
from skyward.providers.ssh_keys import get_local_ssh_key, get_ssh_key_path
from skyward.providers.wait import wait_for_ready

from .client import RunPodClient, RunPodError, get_api_key
from .config import CloudType, RunPod
from .state import RunPodClusterState
from .types import (
    ClusterCreateParams,
    CpuPodCreateParams,
    PodCreateParams,
    PodResponse,
    get_gpu_model,
    get_ssh_port,
)

if TYPE_CHECKING:
    from skyward.spec import PoolSpec


@component
class RunPodHandler:
    """Event-driven RunPod provider using Event Pipeline.

    Flow:
        ClusterRequested -> setup -> ClusterProvisioned
        InstanceRequested -> create_pod -> InstanceLaunched
        InstanceLaunched -> poll running -> InstanceRunning
        BootstrapRequested -> wait bootstrap -> InstanceBootstrapped
        ShutdownRequested -> cleanup -> ClusterDestroyed

    The InstanceOrchestrator handles:
        InstanceRunning -> InstanceProvisioned + BootstrapRequested
    """

    bus: AsyncEventBus
    config: RunPod
    ssh_credentials: SSHCredentialsRegistry

    _clusters: dict[str, RunPodClusterState] = field(default_factory=dict)
    _bootstrap_waiters: dict[str, asyncio.Future[bool]] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Cluster Lifecycle
    # -------------------------------------------------------------------------

    @on(ClusterRequested, match=lambda self, e: e.provider == "runpod")
    async def handle_cluster_requested(self, _: Any, event: ClusterRequested) -> None:
        """Provision RunPod infrastructure for a new cluster.

        For multi-node (nodes >= 2), creates an Instant Cluster with
        high-speed networking (1600-3200 Gbps).
        For single-node, uses individual pod creation.
        """
        logger.info(f"RunPod: Provisioning cluster for {event.spec.nodes} nodes")

        cluster_id = f"runpod-{uuid.uuid4().hex[:8]}"

        initial_region = (
            self.config.data_center_ids[0]
            if self.config.data_center_ids != "global"
            else "global"
        )
        state = RunPodClusterState(
            cluster_id=cluster_id,
            spec=event.spec,
            cloud_type=self.config.cloud_type.value,
            data_center_ids=self.config.data_center_ids,
            region=initial_region,
        )
        self._clusters[cluster_id] = state

        # Register SSH credentials for EventStreamer
        ssh_key_path = get_ssh_key_path()
        _, ssh_public_key = get_local_ssh_key()
        state.ssh_key_path = ssh_key_path
        state.ssh_public_key = ssh_public_key
        self.ssh_credentials.register(cluster_id, state.username, ssh_key_path)

        # Ensure SSH key is registered on RunPod account
        api_key = get_api_key(self.config.api_key)
        async with RunPodClient(api_key) as client:
            await client.ensure_ssh_key(ssh_public_key)
            logger.debug("RunPod: SSH key registered on account")

        # Resolve GPU type
        gpu_type_id = await self._resolve_gpu_type(event.spec)
        state.gpu_type_id = gpu_type_id

        # Multi-node: create Instant Cluster for high-speed networking
        if event.spec.nodes >= 2:
            await self._create_instant_cluster(state, event)

        self.bus.emit(
            ClusterProvisioned(
                request_id=event.request_id,
                cluster_id=cluster_id,
                provider="runpod",
            )
        )

    @on(ShutdownRequested)
    async def handle_shutdown_requested(self, _: Any, event: ShutdownRequested) -> None:
        """Terminate all pods in a cluster."""
        cluster = self._clusters.pop(event.cluster_id, None)
        if not cluster:
            return

        logger.info(f"RunPod: Shutting down cluster {event.cluster_id}")

        api_key = get_api_key(self.config.api_key)
        async with RunPodClient(api_key) as client:
            if cluster.is_instant_cluster:
                # Delete the entire Instant Cluster
                with suppress(Exception):
                    await client.delete_cluster(cluster.runpod_cluster_id)  # type: ignore[arg-type]
            else:
                # Single-node: terminate individual pods
                for pod_id in cluster.pod_ids.values():
                    with suppress(Exception):
                        await client.terminate_pod(pod_id)

        self.bus.emit(ClusterDestroyed(cluster_id=event.cluster_id))

    # -------------------------------------------------------------------------
    # Instance Lifecycle - Event Pipeline
    # -------------------------------------------------------------------------

    @on(InstanceRequested, match=lambda self, e: e.provider == "runpod")
    async def handle_instance_requested(self, _: Any, event: InstanceRequested) -> None:
        """Launch RunPod pod and emit InstanceLaunched.

        For Instant Clusters, the pods are already created - just emit InstanceLaunched.
        For single-node, creates a new pod.
        """
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        # For Instant Clusters, pods are already created
        if cluster.is_instant_cluster:
            pod_id = cluster.pod_ids.get(event.node_id)
            if pod_id:
                logger.info(f"RunPod: Instant Cluster pod {pod_id} for node {event.node_id}")
                cluster.pending_nodes.add(event.node_id)
                self.bus.emit(
                    InstanceLaunched(
                        request_id=event.request_id,
                        cluster_id=event.cluster_id,
                        node_id=event.node_id,
                        provider="runpod",
                        instance_id=pod_id,
                    )
                )
            return

        # Single-node: create individual pod
        logger.info(f"RunPod: Launching pod for node {event.node_id}")

        api_key = get_api_key(self.config.api_key)

        try:
            async with RunPodClient(api_key) as client:
                if cluster.gpu_type_id:
                    pod = await self._create_gpu_pod(client, cluster, event)
                else:
                    pod = await self._create_cpu_pod(client, cluster, event)
        except RunPodError as e:
            logger.error(f"RunPod: Failed to create pod: {e}")
            return

        pod_id = pod["id"]
        logger.debug(f"RunPod: Pod created with id {pod_id}")
        cluster.pod_ids[event.node_id] = pod_id
        cluster.pending_nodes.add(event.node_id)
        machine = pod.get("machine") or {}
        cluster.region = machine.get("dataCenterId") or machine.get("location") or ""

        # Emit intermediate event - pod created, waiting for running
        logger.debug(f"RunPod: Emitting InstanceLaunched for pod {pod_id}")
        self.bus.emit(
            InstanceLaunched(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="runpod",
                instance_id=pod_id,
            )
        )
        logger.debug(f"RunPod: InstanceLaunched emitted for pod {pod_id}")

    @on(InstanceLaunched, match=lambda self, e: e.provider == "runpod")
    async def handle_instance_launched(self, _: Any, event: InstanceLaunched) -> None:
        """Wait for pod to be running and emit InstanceRunning."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        logger.info(f"RunPod: Waiting for pod {event.instance_id} to be running...")

        api_key = get_api_key(self.config.api_key)

        try:
            async with RunPodClient(api_key) as client:
                pod = await wait_for_ready(
                    poll_fn=lambda: client.get_pod(event.instance_id),
                    ready_check=lambda p: (
                        p is not None
                        and p.get("desiredStatus") == "RUNNING"
                        and bool(p.get("publicIp"))
                    ),
                    terminal_check=lambda p: (
                        p is not None and p.get("desiredStatus") == "TERMINATED"
                    ),
                    timeout=self.config.provision_timeout,
                    interval=5.0,
                    description=f"RunPod pod {event.instance_id}",
                )
        except TimeoutError:
            logger.error(f"RunPod: Pod {event.instance_id} did not become ready")
            return

        if not pod:
            logger.error(f"RunPod: Pod {event.instance_id} not found")
            return

        # Extract info from pod response
        ip = pod.get("publicIp") or ""
        ssh_port = get_ssh_port(pod)
        hourly_rate = pod.get("costPerHr", 0.0)
        adjusted_rate = pod.get("adjustedCostPerHr", hourly_rate)

        # For Instant Clusters, get the internal cluster IP for inter-node communication
        # This enables high-speed networking (1600-3200 Gbps) between nodes
        private_ip: str | None = None
        if cluster.is_instant_cluster:
            private_ip = cluster.cluster_ips.get(event.node_id)
            if private_ip:
                logger.debug(f"RunPod: Using cluster IP {private_ip} for node {event.node_id}")

        # Update cluster state with pricing
        cluster.hourly_rate = adjusted_rate
        cluster.on_demand_rate = hourly_rate
        cluster.gpu_count = pod.get("gpuCount", 0)
        cluster.gpu_model = cluster.spec.accelerator_name or get_gpu_model(pod)
        cluster.vcpus = pod.get("vcpuCount", 0)
        cluster.memory_gb = pod.get("memoryInGb", 0.0)

        # Emit InstanceRunning - InstanceOrchestrator will handle the rest
        self.bus.emit(
            InstanceRunning(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="runpod",
                instance_id=event.instance_id,
                ip=ip,
                private_ip=private_ip,  # Cluster IP for Instant Clusters
                ssh_port=ssh_port,
                spot=pod.get("interruptible", False),
                # Pricing info
                hourly_rate=adjusted_rate,
                on_demand_rate=hourly_rate,
                billing_increment=1,  # RunPod bills per-minute
                # Instance details
                instance_type=cluster.gpu_type_id or "",
                gpu_count=cluster.gpu_count,
                gpu_model=cluster.gpu_model,
                # Hardware specs
                vcpus=cluster.vcpus,
                memory_gb=cluster.memory_gb,
                gpu_vram_gb=cluster.gpu_vram_gb,
                region=cluster.region,
            )
        )

    @on(BootstrapRequested, match=lambda self, e: e.instance.provider == "runpod")
    async def handle_bootstrap_requested(self, _: Any, event: BootstrapRequested) -> None:
        """Run bootstrap via SSH and wait for completion."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        info = event.instance
        ttl = cluster.spec.ttl or self.config.instance_timeout
        bootstrap_script = cluster.spec.image.generate_bootstrap(
            ttl=ttl,
            shutdown_command="eval $(cat /proc/1/environ | tr '\\0' '\\n' | grep RUNPOD_ | sed 's/^/export /'); runpodctl remove pod $RUNPOD_POD_ID",
        )

        logger.info(f"RunPod: Connecting to {info.ip}:{info.ssh_port} to run bootstrap...")
        from skyward.providers.bootstrap import run_bootstrap_via_ssh, wait_for_ssh

        transport = await wait_for_ssh(
            host=info.ip,
            user=cluster.username,
            key_path=cluster.ssh_key_path,
            port=info.ssh_port,
            timeout=60.0,
        )

        try:
            await transport.run("mkdir -p /opt/skyward")
            await run_bootstrap_via_ssh(
                transport=transport,
                info=info,
                bootstrap_script=bootstrap_script,
                log_prefix="RunPod: ",
            )
        finally:
            await transport.close()

        logger.debug(f"RunPod: Waiting for bootstrap completion on {info.id}")

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[bool] = loop.create_future()
        self._bootstrap_waiters[info.id] = waiter

        try:
            success = await asyncio.wait_for(waiter, timeout=self.config.bootstrap_timeout)
            if success:
                logger.info(f"RunPod: Bootstrap completed, skyward_source={cluster.spec.image.skyward_source}")
                if cluster.spec.image and cluster.spec.image.skyward_source == "local":
                    logger.info(f"RunPod: Installing local skyward wheel on {info.id}...")
                    await self._install_local_skyward(info, cluster)
                    logger.info(f"RunPod: Local skyward installed on {info.id}")
                cluster.add_instance(info)
                self.bus.emit(InstanceBootstrapped(instance=info))
            else:
                logger.error(f"RunPod: Bootstrap failed on {info.id}")
        except asyncio.TimeoutError:
            logger.error(f"RunPod: Bootstrap timed out on {info.id}")
        finally:
            self._bootstrap_waiters.pop(info.id, None)

    @on(BootstrapPhase, match=lambda self, e: e.instance.provider == "runpod", audit=False)
    async def handle_bootstrap_phase(self, _: Any, event: BootstrapPhase) -> None:
        """Handle bootstrap phase events from EventStreamer."""
        # Only care about bootstrap phase completion/failure
        if event.phase != "bootstrap" or event.event not in ("completed", "failed"):
            return

        instance_id = event.instance.id
        waiter = self._bootstrap_waiters.get(instance_id)
        if waiter and not waiter.done():
            waiter.set_result(event.event == "completed")

    @on(BootstrapFailed, match=lambda self, e: e.instance.provider == "runpod", audit=False)
    async def handle_bootstrap_failed(self, _: Any, event: BootstrapFailed) -> None:
        """Handle bootstrap failure from EventStreamer."""
        instance_id = event.instance.id
        waiter = self._bootstrap_waiters.get(instance_id)
        if waiter and not waiter.done():
            waiter.set_result(False)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    async def _create_gpu_pod(
        self,
        client: RunPodClient,
        cluster: RunPodClusterState,
        event: InstanceRequested,
    ) -> PodResponse:
        """Create a GPU pod via REST API."""
        use_spot = cluster.spec.allocation in ("spot", "spot-if-available")

        params: PodCreateParams = {
            "name": f"skyward-{cluster.cluster_id}-{event.node_id}",
            "imageName": self._get_image_name(cluster.spec),
            "gpuTypeIds": [cluster.gpu_type_id or ""],
            "gpuCount": cluster.spec.accelerator_count or 1,
            "cloudType": self.config.cloud_type.value.upper(),
            "containerDiskInGb": self.config.container_disk_gb,
            "volumeInGb": self.config.volume_gb,
            "volumeMountPath": self.config.volume_mount_path,
            "ports": list(self.config.ports),
            "interruptible": use_spot,
        }

        if self.config.data_center_ids != "global":
            params["dataCenterIds"] = list(self.config.data_center_ids)

        return await client.create_pod(params)

    async def _create_cpu_pod(
        self,
        client: RunPodClient,
        cluster: RunPodClusterState,
        _event: InstanceRequested,
    ) -> PodResponse:
        """Create a CPU-only pod via GraphQL deployCpuPod mutation."""
        vcpus = cluster.spec.vcpus or 4
        memory_gb = cluster.spec.memory_gb or 8
        disk_gb = min(self.config.container_disk_gb, 20)
        instance_id = f"cpu{self.config.cpu_clock}-{vcpus}-{memory_gb}"

        params: CpuPodCreateParams = {
            "instanceId": instance_id,
            "cloudType": self.config.cloud_type.value.upper(),
            "containerDiskInGb": disk_gb,
            "startSsh": True,
            "templateId": "runpod-ubuntu",
            "ports": ",".join(self.config.ports),
            "deployCost": cluster.spec.max_hourly_cost or 0.50,
        }

        if self.config.data_center_ids != "global":
            params["dataCenterId"] = self.config.data_center_ids[0]

        logger.info(f"RunPod: Creating CPU pod with instance {instance_id}")
        return await client.create_cpu_pod(params)

    async def _create_instant_cluster(
        self,
        state: RunPodClusterState,
        event: ClusterRequested,
    ) -> None:
        """Create an Instant Cluster for multi-node deployments.

        Instant Clusters provide high-speed networking (1600-3200 Gbps)
        between pods via InfiniBand/RoCE.

        Args:
            state: Cluster state to update with pod IDs and cluster IPs.
            event: The cluster request event with spec.
        """
        logger.info(f"RunPod: Creating Instant Cluster with {event.spec.nodes} nodes")

        api_key = get_api_key(self.config.api_key)

        # Determine cluster type based on allocation
        # TRAINING is the appropriate type for ML distributed training
        cluster_type = "TRAINING"

        # deployCost is required - use max_hourly_cost from spec or default
        # RunPod will charge the actual market rate, not the full bid
        deploy_cost = event.spec.max_hourly_cost or 10.0

        params: ClusterCreateParams = {
            "clusterName": f"skyward-{state.cluster_id}",
            "gpuTypeId": state.gpu_type_id or "",
            "podCount": event.spec.nodes,
            "gpuCountPerPod": event.spec.accelerator_count or 1,
            "type": cluster_type,
            "imageName": self._get_image_name(event.spec),
            "startSsh": True,
            "containerDiskInGb": self.config.container_disk_gb,
            "volumeInGb": self.config.volume_gb,
            "volumeMountPath": self.config.volume_mount_path,
            "ports": ",".join(self.config.ports),
            "deployCost": deploy_cost,
        }

        if self.config.data_center_ids != "global":
            params["dataCenterId"] = self.config.data_center_ids[0]

        try:
            async with RunPodClient(api_key) as client:
                cluster = await client.create_cluster(params)
        except RunPodError as e:
            logger.error(f"RunPod: Failed to create Instant Cluster: {e}")
            raise

        state.runpod_cluster_id = cluster["id"]
        logger.info(f"RunPod: Instant Cluster created with id {state.runpod_cluster_id}")

        # Map pods to nodes by their cluster index
        for pod in cluster["pods"]:
            pod_id = pod["id"]
            # Use clusterIdx if available, otherwise use order
            node_id = pod.get("clusterIdx", len(state.pod_ids))
            state.pod_ids[node_id] = pod_id

            # Store cluster IP for inter-node communication
            cluster_ip = pod.get("clusterIp")
            if cluster_ip:
                state.cluster_ips[node_id] = cluster_ip
                logger.debug(f"RunPod: Node {node_id} -> pod {pod_id}, cluster IP {cluster_ip}")
            else:
                logger.debug(f"RunPod: Node {node_id} -> pod {pod_id} (no cluster IP yet)")

    async def _resolve_gpu_type(self, spec: PoolSpec) -> str | None:
        """Resolve GPU type ID from spec accelerator name."""
        if not spec.accelerator_name:
            return None  # CPU-only pod

        api_key = get_api_key(self.config.api_key)
        async with RunPodClient(api_key) as client:
            gpu_types = await client.get_gpu_types()

        # Filter by cloud type
        is_secure = self.config.cloud_type == CloudType.SECURE
        available = [
            g for g in gpu_types
            if (is_secure and g.get("secureCloud")) or (not is_secure and g.get("communityCloud"))
        ]

        # Match by name (case-insensitive, partial match)
        requested = spec.accelerator_name.upper()
        for gpu in available:
            display_name = gpu.get("displayName", "").upper()
            gpu_id = gpu.get("id", "").upper()
            if requested in display_name or requested in gpu_id:
                logger.info(f"RunPod: Selected GPU type {gpu['id']} ({gpu.get('displayName')})")
                return gpu["id"]

        available_names = [g.get("displayName", g["id"]) for g in available]
        raise RuntimeError(
            f"No GPU type matches '{spec.accelerator_name}'. "
            f"Available: {', '.join(available_names)}"
        )

    def _get_image_name(self, spec: PoolSpec) -> str:
        """Get container image name from spec."""
        # Default to PyTorch CUDA image
        if spec.image and hasattr(spec.image, "container_image"):
            return getattr(spec.image, "container_image")
        return "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

    async def _install_local_skyward(
        self,
        info: Any,
        cluster: RunPodClusterState,
    ) -> None:
        """Install local skyward wheel."""
        from skyward.providers.bootstrap import install_local_skyward, wait_for_ssh

        transport = await wait_for_ssh(
            host=info.ip,
            user=cluster.username,
            key_path=cluster.ssh_key_path,
            port=info.ssh_port,
            timeout=60.0,
            log_prefix="RunPod: ",
        )

        try:
            await install_local_skyward(
                transport=transport,
                info=info,
                log_prefix="RunPod: ",
                use_sudo=False,
            )
        finally:
            await transport.close()


__all__ = ["RunPodHandler"]
