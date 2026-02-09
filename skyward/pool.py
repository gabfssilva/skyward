"""ComputePool component - cluster orchestration.

ComputePool is the main entry point for users. It:
- Requests cluster infrastructure from provider
- Creates and manages Node instances
- Tracks cluster readiness
- Provides task execution API
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from skyward.cluster.info import InstanceInfo

from .app import component, on
from .bus import AsyncEventBus
from .events import (
    ClusterId,
    ClusterProvisioned,
    ClusterReady,
    ClusterRequested,
    InstanceMetadata,
    NodeId,
    NodeReady,
    ProviderName,
    ShutdownRequested,
)
from .executor import HTTP_PORT, Executor
from .monitors import EventStreamer
from .node import Node
from .spec import PoolSpec


# =============================================================================
# Pool State
# =============================================================================


class PoolState:
    """Pool lifecycle states."""

    INIT = "init"
    REQUESTING = "requesting"
    PROVISIONING = "provisioning"
    READY = "ready"
    SHUTTING_DOWN = "shutting_down"
    DESTROYED = "destroyed"


# =============================================================================
# ComputePool Component
# =============================================================================


@component
class ComputePool:
    """
    ComputePool - cluster orchestration component.

    Main user-facing API for distributed compute. Manages the full lifecycle
    of a cluster from provisioning through execution to shutdown.

    Lifecycle:
    1. User creates ComputePool with spec
    2. pool.start() emits ClusterRequested
    3. Provider creates infra, emits ClusterProvisioned
    4. Pool creates Nodes, each requests their instance
    5. As nodes become ready, pool tracks progress
    6. When all nodes ready, pool emits ClusterReady
    7. User can execute tasks via pool.run()
    8. pool.stop() emits ShutdownRequested

    Usage:
        async with app_context(AppModule()) as app:
            pool = app.get(ComputePool)
            await pool.start()  # Wait for cluster to be ready
            result = await pool.run(my_function, args)
            await pool.stop()

    Attributes:
        bus: Event bus for communication
        spec: Pool specification
        provider: Provider name (from spec or injected)
    """

    # Required - injected or passed
    bus: AsyncEventBus
    spec: PoolSpec

    # EventStreamer - injected to ensure it's created before cluster starts
    # This component handles continuous event streaming from instances
    event_streamer: EventStreamer

    # Computed from spec or injected
    provider: ProviderName = "aws"  # Default, overridden by spec.provider

    # Internal state
    _state: str = PoolState.INIT
    _cluster_id: ClusterId = ""
    _request_id: str = ""
    _nodes: dict[NodeId, Node] = {}  # type: ignore[assignment]
    _ready_nodes: set[NodeId] = set()  # type: ignore[assignment]
    _ready_event: asyncio.Event | None = None
    _executor: Executor | None = None
    _executor_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _network_interfaces: dict[NodeId, str] = {}  # type: ignore[assignment]
    _next_node: int = 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def start(self, timeout: float = 600.0) -> None:
        """Start cluster and wait for all nodes to be ready.

        Args:
            timeout: Maximum time to wait for cluster readiness.

        Raises:
            asyncio.TimeoutError: If cluster doesn't become ready in time.
            RuntimeError: If pool is not in INIT state.
        """
        if self._state != PoolState.INIT:
            raise RuntimeError(f"Cannot start pool in state {self._state}")

        # Initialize state containers
        self._nodes = {}
        self._ready_nodes = set()
        self._ready_event = asyncio.Event()
        self._network_interfaces = {}

        # Determine provider from spec or use default
        if self.spec.provider:
            self.provider = self.spec.provider

        # Generate request ID and cluster ID
        self._request_id = f"pool-{uuid.uuid4().hex[:8]}"
        self._state = PoolState.REQUESTING

        # Request cluster
        self.bus.emit(
            ClusterRequested(
                request_id=self._request_id,
                provider=self.provider,
                spec=self.spec,
            )
        )

        # Wait for cluster to be ready
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._state = PoolState.DESTROYED
            raise

    async def stop(self) -> None:
        """Shutdown cluster gracefully."""
        if self._state not in (PoolState.READY, PoolState.PROVISIONING):
            return

        self._state = PoolState.SHUTTING_DOWN

        from loguru import logger

        if self._executor is not None:
            logger.debug("Pool: Disconnecting executor...")
            await self._executor.disconnect()
            self._executor = None
            logger.debug("Pool: Executor disconnected")

        # Emit shutdown and wait for provider to terminate instances
        logger.debug(f"Pool: Emitting ShutdownRequested for cluster_id={self._cluster_id}")
        await self.bus.emit_await(
            ShutdownRequested(cluster_id=self._cluster_id)
        )
        logger.debug("Pool: ShutdownRequested handled")

        self._state = PoolState.DESTROYED

    async def run[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        node: int | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute function on a node.

        Args:
            fn: Function to execute remotely.
            *args: Positional arguments.
            node: Specific node index, or None for any available.
            **kwargs: Keyword arguments.

        Returns:
            Result of function execution.

        Raises:
            RuntimeError: If pool is not ready.
        """
        if self._state != PoolState.READY:
            raise RuntimeError(f"Pool not ready, state: {self._state}")

        # Validate node if specified
        if node is not None:
            target_node = self._nodes.get(node)
            if not target_node or not target_node.is_ready:
                raise ValueError(f"Node {node} not available")
            node_id = node
        else:
            # Round-robin across ready nodes
            ready = [n for n in self._nodes.values() if n.is_ready]
            if not ready:
                raise RuntimeError("No ready nodes available")
            idx = self._next_node % len(ready)
            self._next_node = idx + 1
            target_node = ready[idx]
            node_id = target_node.id

        # Ensure executor is connected
        executor = await self._get_or_create_executor()
        return await executor.execute(fn, *args, node_id=node_id, **kwargs)

    async def broadcast[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> list[T]:
        """Execute function on all nodes.

        Args:
            fn: Function to execute on each node.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            List of results from each node, ordered by node_id.
        """
        if self._state != PoolState.READY:
            raise RuntimeError(f"Pool not ready, state: {self._state}")

        # Ensure executor is connected
        executor = await self._get_or_create_executor()
        return await executor.broadcast(fn, *args, **kwargs)

    async def _get_or_create_executor(self) -> Executor:
        if self._executor is not None:
            return self._executor

        async with self._executor_lock:
            if self._executor is not None:
                return self._executor

            return await self._create_executor()

    async def _create_executor(self) -> Executor:
        from skyward.providers.ssh_keys import get_ssh_key_path

        head_node = self._nodes.get(0)
        if head_node is None or head_node.info is None:
            raise RuntimeError("Head node (node 0) not available")

        head_info = head_node.info
        head_addr = head_info.private_ip or head_info.ip
        ssh_user = self._get_ssh_user()
        ssh_key = get_ssh_key_path()

        await self._start_casty_cluster(head_addr, ssh_user, ssh_key)

        pool_infos: list[str] = []
        for node_id in range(self.spec.nodes):
            node = self._nodes.get(node_id)
            if node is None or node.info is None:
                pool_infos.append("")
                continue
            pool_info = self._build_pool_info(node.info, head_addr)
            pool_infos.append(pool_info.model_dump_json())

        image_env = dict(self.spec.image.env) if self.spec.image and self.spec.image.env else {}
        self._executor = Executor(
            head_ip=head_info.ip,
            ssh_port=head_info.ssh_port,
            user=ssh_user,
            key_path=ssh_key,
            num_nodes=self.spec.nodes,
            http_port=HTTP_PORT,
            env_vars=image_env,
            pool_infos=pool_infos,
        )
        await self._executor.connect()

        await self._executor.setup_cluster(env_vars=image_env)
        logger.debug("Executor connected and ready")

        return self._executor

    async def _start_casty_cluster(
        self,
        head_addr: str,
        ssh_user: str,
        ssh_key: str,
    ) -> None:
        from skyward.transport.ssh import SSHTransport
        from skyward.constants import VENV_DIR
        from skyward.bootstrap import EMIT_SH_PATH
        from skyward.providers.common import detect_network_interface

        python_bin = f"{VENV_DIR}/bin/python"
        casty_port = 25520
        http_port = HTTP_PORT
        logger.debug("Starting Casty cluster via SSH...")

        head_node = self._nodes.get(0)
        if head_node is None or head_node.info is None:
            raise RuntimeError("Head node not available")

        use_sudo = ssh_user != "root"

        async def _start_node(
            transport: SSHTransport,
            node_id: int,
            host: str,
            seeds: str = "",
        ) -> str:
            seeds_arg = f"--seeds {seeds} " if seeds else ""
            casty_cmd = (
                f"nohup {python_bin} -m skyward.casty_worker "
                f"--node-id {node_id} --port {casty_port} --http-port {http_port} "
                f"--num-nodes {self.spec.nodes} --host {host} "
                f"--workers-per-node {self.spec.concurrency} "
                f"{seeds_arg}"
                f"> /var/log/casty.log 2>&1 & echo $!"
            )
            tail_inner = (
                f"source {EMIT_SH_PATH} && "
                f"tail -f /var/log/casty.log 2>/dev/null | while IFS= read -r line; do "
                f'emit_console "[casty] $line"; done'
            )
            tail_cmd = f"nohup bash -c '{tail_inner}' </dev/null >/dev/null 2>&1 &"

            if use_sudo:
                casty_cmd = f"sudo bash -c '{casty_cmd}'"
                tail_cmd = f"sudo {tail_cmd}"

            exit_code, stdout, stderr = await transport.run(casty_cmd, timeout=60.0)
            if exit_code != 0:
                raise RuntimeError(f"Failed to start Casty node {node_id}: {stderr}")
            await transport.run(tail_cmd, timeout=10.0)
            return stdout.strip()

        async with SSHTransport(
            host=head_node.info.ip,
            user=ssh_user,
            key_path=ssh_key,
            port=head_node.info.ssh_port,
        ) as transport:
            logger.debug("Starting Casty head on node 0...")
            pid = await _start_node(transport, 0, host=head_addr)
            logger.debug(f"Casty head PID: {pid}")
            self._network_interfaces[0] = await detect_network_interface(transport)

        async def _start_worker(node_id: int, node: Node) -> None:
            assert node.info is not None
            async with SSHTransport(
                host=node.info.ip,
                user=ssh_user,
                key_path=ssh_key,
                port=node.info.ssh_port,
            ) as transport:
                worker_addr = node.info.private_ip or node.info.ip
                logger.debug(f"Starting Casty worker on node {node_id}...")
                await _start_node(transport, node_id, host=worker_addr, seeds=f"{head_addr}:{casty_port}")
                self._network_interfaces[node_id] = await detect_network_interface(transport)

        await asyncio.gather(*(
            _start_worker(node_id, node)
            for node_id, node in self._nodes.items()
            if node_id != 0 and node.info is not None
        ))

        logger.debug("Casty cluster started")

    def _build_pool_info(
        self,
        info: InstanceMetadata,
        head_addr: str,
    ) -> "InstanceInfo":
        """Build COMPUTE_POOL info for an instance.

        Uses v1's build_pool_info to create compatible cluster info.
        """
        from skyward.providers.pool_info import build_pool_info

        # Collect peers info from all nodes
        peers = [
            {
                "node": n.id,
                "private_ip": n.info.private_ip or n.info.ip if n.info else "",
            }
            for n in self._nodes.values()
            if n.info is not None
        ]

        accelerator_count = info.gpu_count or 1
        total_accelerators = sum(
            (n.info.gpu_count or 1) for n in self._nodes.values() if n.info is not None
        )

        return build_pool_info(
            node=info.node,
            total_nodes=self.spec.nodes,
            accelerator_count=accelerator_count,
            total_accelerators=total_accelerators,
            head_addr=head_addr,
            head_port=29500,
            job_id=self._cluster_id,
            peers=peers,
            accelerator_type=self.spec.accelerator_name,
            placement_group=self._network_interfaces.get(info.node) or info.network_interface or None,
            worker=0,
            workers_per_node=1,
        )

    def _get_ssh_user(self) -> str:
        """Get SSH username based on provider."""
        provider = self.spec.provider or self.provider
        match provider:
            case "verda" | "vastai" | "runpod":
                return "root"
            case _:
                return "ubuntu"


    @property
    def cluster_id(self) -> ClusterId:
        """Cluster ID once provisioned."""
        return self._cluster_id

    @property
    def nodes(self) -> dict[NodeId, Node]:
        """Map of node ID to Node instances."""
        return self._nodes

    @property
    def is_ready(self) -> bool:
        """Whether all nodes are ready."""
        return self._state == PoolState.READY

    @property
    def state(self) -> str:
        """Current pool state."""
        return self._state

    @property
    def ready_count(self) -> int:
        """Number of ready nodes."""
        return len(self._ready_nodes)

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> ComputePool:
        """Start pool on context entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop pool on context exit."""
        await self.stop()

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    @on(ClusterProvisioned, match=lambda self, e: e.request_id == self._request_id)
    async def _on_cluster_provisioned(
        self,
        _sender: Any,
        event: ClusterProvisioned,
    ) -> None:
        """Handle cluster infrastructure ready - create nodes."""
        self._cluster_id = event.cluster_id
        self._state = PoolState.PROVISIONING

        # Create nodes (auto-wired via @component)
        for i in range(self.spec.nodes):
            node = Node(
                id=i,
                bus=self.bus,
                provider=self.provider,
                cluster_id=self._cluster_id,
            )
            self._nodes[i] = node

            # Start provisioning each node
            await node.provision()

    @on(NodeReady)
    async def _on_node_ready(self, _sender: Any, event: NodeReady) -> None:
        """Handle node ready - track progress and emit ClusterReady when done."""
        # Only track nodes for our cluster
        node = self._nodes.get(event.node_id)
        if not node:
            return

        self._ready_nodes.add(event.node_id)

        # Check if all nodes are ready
        if len(self._ready_nodes) == self.spec.nodes:
            self._state = PoolState.READY

            # Collect instance info
            instances = tuple(
                node.info
                for node in self._nodes.values()
                if node.info is not None
            )

            # Emit cluster ready
            self.bus.emit(
                ClusterReady(
                    cluster_id=self._cluster_id,
                    nodes=instances,
                )
            )

            # Signal waiters
            if self._ready_event:
                self._ready_event.set()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ComputePool",
    "PoolState",
]
