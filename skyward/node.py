"""Node component - instance lifecycle management.

Each Node manages a single instance slot in the cluster. It handles:
- Initial provisioning request
- Tracking instance state
- Automatic replacement on preemption
- Signaling readiness to pool
"""

from __future__ import annotations

import uuid
from typing import Any

from .app import component, on
from .bus import AsyncEventBus
from .events import (
    ClusterId,
    InstanceBootstrapped,
    InstanceId,
    InstanceInfo,
    InstancePreempted,
    InstanceProvisioned,
    InstanceReplaced,
    InstanceRequested,
    NodeId,
    NodeReady,
    ProviderName,
)


# =============================================================================
# Node State
# =============================================================================


class NodeState:
    """Internal state tracking for node."""

    INIT = "init"
    PROVISIONING = "provisioning"
    BOOTSTRAPPING = "bootstrapping"
    READY = "ready"
    REPLACING = "replacing"
    DESTROYED = "destroyed"


# =============================================================================
# Node Component
# =============================================================================


@component
class Node:
    """
    Node component - manages a single instance slot.

    A Node is responsible for one "slot" in the cluster. It requests
    instances from the provider and handles replacement on preemption.

    Lifecycle:
    1. Pool creates Node with id, provider, cluster_id
    2. Node.provision() emits InstanceRequested
    3. Provider handles request, emits InstanceProvisioned, InstanceBootstrapped
    4. Node receives InstanceBootstrapped, emits NodeReady
    5. On preemption: Node receives InstancePreempted, emits InstanceRequested(replacing=...)
    6. Provider launches replacement, Node receives InstanceReplaced, emits NodeReady

    Attributes:
        id: Node slot number (0, 1, 2, ...)
        bus: Event bus for communication
        provider: Provider name for this node
        cluster_id: Cluster this node belongs to
    """

    # Required - injected or passed
    id: NodeId
    bus: AsyncEventBus
    provider: ProviderName
    cluster_id: ClusterId

    # Internal state
    _state: str = NodeState.INIT
    _instance_id: InstanceId = ""
    _info: InstanceInfo | None = None
    _pending_request_id: str = ""

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def provision(self) -> None:
        """Request initial instance for this node.

        Emits InstanceRequested and transitions to PROVISIONING state.
        """
        if self._state != NodeState.INIT:
            raise RuntimeError(f"Cannot provision node in state {self._state}")

        request_id = self._make_request_id()
        self._pending_request_id = request_id
        self._state = NodeState.PROVISIONING

        self.bus.emit(
            InstanceRequested(
                request_id=request_id,
                provider=self.provider,
                cluster_id=self.cluster_id,
                node_id=self.id,
                replacing=None,
            )
        )

    async def replace(self, reason: str) -> None:  # noqa: ARG002
        """Request replacement instance after preemption.

        Args:
            reason: Why the instance was preempted (for logging).

        Emits InstanceRequested with replacing= set to old instance ID.
        """
        if self._state not in (NodeState.READY, NodeState.BOOTSTRAPPING):
            return  # Ignore if not in a replaceable state

        old_id = self._instance_id
        request_id = self._make_request_id()
        self._pending_request_id = request_id
        self._state = NodeState.REPLACING

        self.bus.emit(
            InstanceRequested(
                request_id=request_id,
                provider=self.provider,
                cluster_id=self.cluster_id,
                node_id=self.id,
                replacing=old_id,
            )
        )

    @property
    def info(self) -> InstanceInfo | None:
        """Current instance info, or None if not provisioned."""
        return self._info

    @property
    def is_ready(self) -> bool:
        """Whether node has a ready instance."""
        return self._state == NodeState.READY

    @property
    def state(self) -> str:
        """Current node state."""
        return self._state

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    @on(InstanceProvisioned, match=lambda self, e: e.instance.node == self.id)
    async def _on_provisioned(self, _sender: Any, event: InstanceProvisioned) -> None:
        """Handle instance provisioned (created but not yet bootstrapped)."""
        self._instance_id = event.instance.id
        self._info = event.instance
        self._state = NodeState.BOOTSTRAPPING

    @on(InstanceBootstrapped, match=lambda self, e: e.instance.node == self.id)
    async def _on_bootstrapped(self, _sender: Any, event: InstanceBootstrapped) -> None:
        """Handle instance bootstrapped (ready for work)."""
        self._info = event.instance
        self._state = NodeState.READY

        # Signal to pool that this node is ready
        self.bus.emit(
            NodeReady(
                node_id=self.id,
                instance=event.instance,
            )
        )

    @on(InstancePreempted, match=lambda self, e: e.instance.node == self.id)
    async def _on_preempted(self, _sender: Any, event: InstancePreempted) -> None:
        """Handle instance preemption - request replacement."""
        await self.replace(event.reason)

    @on(InstanceReplaced, match=lambda self, e: e.new.node == self.id)
    async def _on_replaced(self, _sender: Any, event: InstanceReplaced) -> None:
        """Handle successful replacement after preemption."""
        self._instance_id = event.new.id
        self._info = event.new
        self._state = NodeState.READY

        # Signal to pool that this node is ready again
        self.bus.emit(
            NodeReady(
                node_id=self.id,
                instance=event.new,
            )
        )

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _make_request_id(self) -> str:
        """Generate unique request ID."""
        return f"node-{self.id}-{uuid.uuid4().hex[:8]}"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Node",
    "NodeState",
]
