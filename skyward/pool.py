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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skyward.cluster.info import InstanceInfo as ClusterInstanceInfo

from .app import component, on
from .bus import AsyncEventBus
from .events import (
    ClusterId,
    ClusterProvisioned,
    ClusterReady,
    ClusterRequested,
    InstanceInfo,
    NodeId,
    NodeReady,
    ProviderName,
    ShutdownRequested,
)
from .executor import AsyncRPyCExecutor
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
    _executors: dict[str, AsyncRPyCExecutor] = {}  # type: ignore[assignment]

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

        # Close all executors
        from loguru import logger
        logger.debug(f"Pool: Closing {len(self._executors)} executors...")
        for executor in self._executors.values():
            await executor.close()
        self._executors = {}
        logger.debug("Pool: All executors closed")

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
        timeout: float | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> T:
        """Execute function on a node.

        Args:
            fn: Function to execute remotely.
            *args: Positional arguments.
            node: Specific node index, or None for any available.
            timeout: Execution timeout.
            **kwargs: Keyword arguments.

        Returns:
            Result of function execution.

        Raises:
            RuntimeError: If pool is not ready.
            TimeoutError: If execution times out.
        """
        if self._state != PoolState.READY:
            raise RuntimeError(f"Pool not ready, state: {self._state}")

        # Select node
        if node is not None:
            target_node = self._nodes.get(node)
            if not target_node or not target_node.is_ready:
                raise ValueError(f"Node {node} not available")
        else:
            # Pick first ready node
            target_node = next(
                (n for n in self._nodes.values() if n.is_ready),
                None,
            )
            if not target_node:
                raise RuntimeError("No ready nodes available")

        # Execute via RPyC
        info = target_node.info
        if info is None:
            raise RuntimeError("Node has no instance info")

        executor = await self._get_or_create_executor(info)
        return await executor.execute(fn, *args, **kwargs)

    async def broadcast[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        timeout: float | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> list[T]:
        """Execute function on all nodes.

        Args:
            fn: Function to execute on each node.
            *args: Positional arguments.
            timeout: Execution timeout per node.
            **kwargs: Keyword arguments.

        Returns:
            List of results from each node.
        """
        if self._state != PoolState.READY:
            raise RuntimeError(f"Pool not ready, state: {self._state}")

        # Execute on all ready nodes in parallel
        ready_nodes = [n for n in self._nodes.values() if n.is_ready and n.info]

        async def execute_on_node(node: Node) -> T:
            executor = await self._get_or_create_executor(node.info)  # type: ignore
            return await executor.execute(fn, *args, **kwargs)

        results = await asyncio.gather(
            *[execute_on_node(n) for n in ready_nodes]
        )
        return list(results)

    async def _get_or_create_executor(self, info: InstanceInfo) -> AsyncRPyCExecutor:
        """Get or create executor for an instance."""
        from .executor import AsyncRPyCExecutor

        if info.id not in self._executors:
            from skyward.providers.ssh_keys import get_ssh_key_path

            executor = AsyncRPyCExecutor(
                host=info.ip,
                ssh_port=info.ssh_port,
                user=self._get_ssh_user(),
                key_path=get_ssh_key_path(),
            )
            await executor.connect()

            # Get head node PRIVATE IP for distributed coordination (inter-node comm)
            head_node = self._nodes.get(0)
            # Use private_ip for inter-node communication within the VPC
            head_addr = (
                head_node.info.private_ip or head_node.info.ip
                if head_node and head_node.info
                else "127.0.0.1"
            )

            # Build COMPUTE_POOL JSON like v1 does
            pool_info = self._build_pool_info(info, head_addr)

            # Setup cluster environment (COMPUTE_POOL + extra env vars)
            await executor.setup_cluster(
                pool_info_json=pool_info.model_dump_json(),
                env_vars={
                    "SKYWARD_NODE_ID": str(info.node),
                    "SKYWARD_TOTAL_NODES": str(self.spec.nodes),
                    "SKYWARD_HEAD_ADDR": head_addr,
                    "SKYWARD_HEAD_PORT": "29500",
                },
            )

            # Note: Log streaming is handled by EventStreamer component,
            # which streams events from instances throughout their lifecycle.

            self._executors[info.id] = executor

        return self._executors[info.id]

    def _build_pool_info(
        self,
        info: InstanceInfo,
        head_addr: str,
    ) -> "ClusterInstanceInfo":
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

        # Default to 1 GPU per node (v2 doesn't track accelerator count yet)
        accelerator_count = 1
        total_accelerators = accelerator_count * self.spec.nodes

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
            placement_group=info.network_interface or None,  # NCCL interface (e.g., eth1)
            worker=0,
            workers_per_node=1,
        )

    def _get_ssh_user(self) -> str:
        """Get SSH username based on provider."""
        # Different providers use different default SSH users
        provider = self.spec.provider or self.provider
        if provider == "verda":
            return "root"
        elif provider == "vastai":
            return "root"
        # AWS EC2 typically uses ubuntu for Ubuntu AMIs
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
