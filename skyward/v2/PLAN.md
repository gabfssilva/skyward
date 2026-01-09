# Skyward v2 - Architecture Plan

## Overview

Event-driven architecture with:
- **100% event communication** between components
- **asyncio** native (no threads)
- **blinker** for signals
- **injector** for DI
- **`@component`** decorator that replaces `@dataclass` + auto-wires handlers

## Core Concepts

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER CODE                                   │
│                                                                          │
│  async with app_context(AppModule()) as app:                            │
│      pool = app.get(ComputePool)                                        │
│      await pool.start()                                                 │
│      result = await pool.run(train, data)                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           AsyncEventBus                                  │
│                                                                          │
│  - Routes events between components                                      │
│  - Supports request/response via correlation IDs                        │
│  - Fire-and-forget or await patterns                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │    Pool     │     │    Node     │     │  Provider   │
   │             │     │             │     │   Handler   │
   │ @component  │     │ @component  │     │ @component  │
   │ @on events  │     │ @on events  │     │ @on events  │
   └─────────────┘     └─────────────┘     └─────────────┘
```

## Event Flow

```
STARTUP:
=========
Pool.start()
    │
    ├──► emit ClusterRequested(provider="aws", spec=...)
    │                                    │
    │         ┌──────────────────────────┘
    │         ▼
    │    AWSHandler @on(ClusterRequested)
    │         │
    │         ├──► create VPC, SG, etc.
    │         │
    │         └──► emit ClusterProvisioned(cluster_id=...)
    │                                    │
    │         ┌──────────────────────────┘
    │         ▼
    ├──◄ Pool @on(ClusterProvisioned)
    │         │
    │         └──► create Nodes, call node.provision()
    │                        │
    │                        ▼
    │              Node.provision()
    │                        │
    │                        └──► emit InstanceRequested(node=0, ...)
    │                                              │
    │                   ┌──────────────────────────┘
    │                   ▼
    │              AWSHandler @on(InstanceRequested)
    │                   │
    │                   ├──► launch EC2
    │                   ├──► emit InstanceProvisioned(...)
    │                   ├──► wait bootstrap
    │                   └──► emit InstanceBootstrapped(...)
    │                                              │
    │                   ┌──────────────────────────┘
    │                   ▼
    │              Node @on(InstanceBootstrapped)
    │                   │
    │                   └──► emit NodeReady(node=0, ...)
    │                                    │
    │         ┌──────────────────────────┘
    │         ▼
    └──◄ Pool @on(NodeReady)
              │
              └──► when all nodes ready: emit ClusterReady(...)


PREEMPTION:
===========
Monitor detects preemption
    │
    └──► emit InstancePreempted(node=2, reason="spot")
                        │
         ┌──────────────┘
         ▼
    Node @on(InstancePreempted)
         │
         └──► emit InstanceRequested(node=2, replacing="i-xxx")
                              │
         ┌────────────────────┘
         ▼
    AWSHandler @on(InstanceRequested)
         │
         ├──► terminate old instance
         ├──► launch new instance
         └──► emit InstanceReplaced(old=..., new=...)
                              │
         ┌────────────────────┘
         ▼
    Node @on(InstanceReplaced)
         │
         └──► emit NodeReady(node=2, ...)
```

## File Structure

```
skyward/v2/
│
├── __init__.py              # Public API exports
│
│   # ═══════════════════════════════════════════════════════════════════
│   # CORE INFRASTRUCTURE
│   # ═══════════════════════════════════════════════════════════════════
│
├── events.py                # All events (Requests + Facts)
├── bus.py                   # AsyncEventBus
├── app.py                   # @component, @on, @monitor, create_app
│
│   # ═══════════════════════════════════════════════════════════════════
│   # DOMAIN MODEL
│   # ═══════════════════════════════════════════════════════════════════
│
├── spec.py                  # PoolSpec, ImageSpec (frozen dataclasses)
├── pool.py                  # ComputePool (@component)
├── node.py                  # Node (@component)
├── protocols.py             # Instance, Transport protocols
│
│   # ═══════════════════════════════════════════════════════════════════
│   # PROVIDERS
│   # ═══════════════════════════════════════════════════════════════════
│
├── providers/
│   ├── __init__.py          # Exports: AWS, VastAI, DigitalOcean, Verda
│   ├── base.py              # Shared utilities
│   │
│   ├── aws/                 # AWS (complex, multiple files)
│   │   ├── __init__.py      # Exports: AWS
│   │   ├── config.py        # AWS config dataclass
│   │   ├── handler.py       # AWSHandler (@component, @on)
│   │   ├── cluster.py       # AWS cluster state
│   │   ├── instance.py      # AWS instance wrapper
│   │   └── infra.py         # VPC, SG, Fleet logic
│   │
│   ├── vastai.py            # VastAI (simpler, one file)
│   ├── digitalocean.py      # DigitalOcean (one file)
│   └── verda.py             # Verda (one file)
│
│   # ═══════════════════════════════════════════════════════════════════
│   # MONITORS
│   # ═══════════════════════════════════════════════════════════════════
│
├── monitors.py              # @monitor functions for preemption, health, metrics
│
│   # ═══════════════════════════════════════════════════════════════════
│   # TRANSPORT & EXECUTION
│   # ═══════════════════════════════════════════════════════════════════
│
├── transport/
│   ├── __init__.py
│   ├── ssh.py               # AsyncSSH transport
│   └── rpyc.py              # RPyC over SSH
│
│   # ═══════════════════════════════════════════════════════════════════
│   # BOOTSTRAP
│   # ═══════════════════════════════════════════════════════════════════
│
└── bootstrap/
    ├── __init__.py
    ├── script.py            # Script composition
    └── ops.py               # apt, pip, systemd operations
```

## Detailed File Specifications

### `events.py`

```python
"""All events - the language of the system."""

# Type aliases
type RequestId = str
type ClusterId = str
type InstanceId = str
type NodeId = int
type ProviderName = Literal["aws", "digitalocean", "vastai", "verda"]

# Value objects
@dataclass(frozen=True, slots=True)
class InstanceInfo:
    id: InstanceId
    node: NodeId
    provider: ProviderName
    ip: str
    spot: bool = False

# ── Requests (commands) ──

@dataclass(frozen=True, slots=True)
class ClusterRequested:
    """Pool requests a cluster."""
    request_id: RequestId
    provider: ProviderName
    spec: PoolSpec

@dataclass(frozen=True, slots=True)
class InstanceRequested:
    """Node requests an instance."""
    request_id: RequestId
    provider: ProviderName
    cluster_id: ClusterId
    node_id: NodeId
    replacing: InstanceId | None = None

@dataclass(frozen=True, slots=True)
class ShutdownRequested:
    """Pool requests shutdown."""
    cluster_id: ClusterId

# ── Facts (what happened) ──

@dataclass(frozen=True, slots=True)
class ClusterProvisioned:
    request_id: RequestId
    cluster_id: ClusterId
    provider: ProviderName

@dataclass(frozen=True, slots=True)
class InstanceProvisioned:
    request_id: RequestId
    instance: InstanceInfo

@dataclass(frozen=True, slots=True)
class InstanceBootstrapped:
    instance: InstanceInfo

@dataclass(frozen=True, slots=True)
class InstancePreempted:
    instance: InstanceInfo
    reason: str

@dataclass(frozen=True, slots=True)
class InstanceReplaced:
    request_id: RequestId
    old_id: InstanceId
    new: InstanceInfo

@dataclass(frozen=True, slots=True)
class InstanceDestroyed:
    instance_id: InstanceId

@dataclass(frozen=True, slots=True)
class NodeReady:
    node_id: NodeId
    instance: InstanceInfo

@dataclass(frozen=True, slots=True)
class ClusterReady:
    cluster_id: ClusterId
    nodes: tuple[InstanceInfo, ...]

@dataclass(frozen=True, slots=True)
class ClusterDestroyed:
    cluster_id: ClusterId

# Type unions
type Request = ClusterRequested | InstanceRequested | ShutdownRequested
type Fact = ClusterProvisioned | InstanceProvisioned | ...
type Event = Request | Fact
```

### `bus.py`

```python
"""AsyncEventBus - async event routing with blinker."""

class AsyncEventBus:
    """
    Features:
    - emit(event) - fire and forget
    - emit_await(event) - wait for handlers
    - request(command) - emit and wait for correlated response
    - connect(event_type, handler) - register handler
    """

    def __init__(self) -> None:
        self._signals: dict[type, Signal] = {}
        self._pending: set[asyncio.Task] = set()
        self._waiters: dict[RequestId, asyncio.Future] = {}

    def connect(self, event_type: type, handler: Callable) -> None: ...
    def emit(self, event: Event) -> None: ...
    async def emit_await(self, event: Event) -> None: ...
    async def request[T](self, command: Event, timeout: float = 300) -> T: ...
    async def drain(self) -> None: ...
```

### `app.py`

```python
"""Application infrastructure: @component, @on, @monitor."""

# ── @on decorator ──

def on(event_type: type) -> Callable:
    """Mark method as event handler."""
    def decorator(method):
        method.__event_handlers__ = getattr(method, '__event_handlers__', [])
        method.__event_handlers__.append(event_type)
        return method
    return decorator

# ── @component decorator ──

def component(cls: type) -> type:
    """
    Transform class into a component:
    1. Generate __init__ from type hints (like @dataclass)
    2. Apply @inject for DI
    3. Auto-wire @on handlers to bus after init

    Usage:
        @component
        class Node:
            id: NodeId
            bus: AsyncEventBus
            provider: ProviderName

            # Optional defaults
            _count: int = 0

            @on(InstancePreempted)
            async def handle(self, sender, event): ...
    """

    # 1. Get type hints for fields
    hints = get_type_hints(cls)

    # 2. Separate required vs optional (has default)
    required = []
    optional = []
    for name, type_hint in hints.items():
        if hasattr(cls, name):
            optional.append((name, type_hint, getattr(cls, name)))
        else:
            required.append((name, type_hint))

    # 3. Generate __init__
    def __init__(self, **kwargs):
        for name, _ in required:
            setattr(self, name, kwargs[name])
        for name, _, default in optional:
            setattr(self, name, kwargs.get(name, default))
        _wire_handlers(self)

    # 4. Apply @inject
    cls.__init__ = inject(__init__)

    # 5. Register for discovery
    _COMPONENT_REGISTRY.append(cls)

    return cls

# ── @monitor decorator ──

def monitor(interval: float = 5.0, name: str | None = None):
    """Transform async function into background loop."""
    def decorator(fn):
        fn.__monitor__ = {"interval": interval, "name": name or fn.__name__}
        return fn
    return decorator

# ── Bootstrap ──

async def create_app(*modules: Module) -> tuple[Injector, MonitorManager]:
    """Create app, wire components, start monitors."""
    ...

@asynccontextmanager
async def app_context(*modules: Module) -> AsyncIterator[Injector]:
    """Full lifecycle context manager."""
    ...
```

### `pool.py`

```python
"""ComputePool - cluster orchestration."""

@component
class ComputePool:
    # Required (injected or passed)
    bus: AsyncEventBus
    provider: ProviderName
    spec: PoolSpec

    # Internal state (defaults)
    cluster_id: str = ""
    _nodes: dict[int, Node] = field(default_factory=dict)
    _ready: asyncio.Event = field(default_factory=asyncio.Event)

    async def start(self) -> None:
        """Request cluster and wait for ready."""
        self.bus.emit(ClusterRequested(...))
        await self._ready.wait()

    async def stop(self) -> None:
        """Shutdown cluster."""
        self.bus.emit(ShutdownRequested(cluster_id=self.cluster_id))

    async def run[T](self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Execute on available node."""
        ...

    @on(ClusterProvisioned)
    async def _on_cluster_provisioned(self, _, event):
        if event.request_id != self._request_id:
            return
        self.cluster_id = event.cluster_id
        # Create nodes...

    @on(NodeReady)
    async def _on_node_ready(self, _, event):
        # Track ready nodes, emit ClusterReady when all done
        ...
```

### `node.py`

```python
"""Node - instance lifecycle management."""

@component
class Node:
    # Required
    id: NodeId
    bus: AsyncEventBus
    provider: ProviderName
    cluster_id: ClusterId

    # State
    instance_id: str = ""
    info: InstanceInfo | None = None

    async def provision(self) -> None:
        """Request initial instance."""
        self.bus.emit(InstanceRequested(...))

    async def replace(self, reason: str) -> None:
        """Request replacement after preemption."""
        self.bus.emit(InstanceRequested(..., replacing=self.instance_id))

    @on(InstanceProvisioned)
    async def _on_provisioned(self, _, event):
        if event.instance.node != self.id:
            return
        self.instance_id = event.instance.id

    @on(InstanceBootstrapped)
    async def _on_bootstrapped(self, _, event):
        if event.instance.node != self.id:
            return
        self.bus.emit(NodeReady(node_id=self.id, instance=event.instance))

    @on(InstancePreempted)
    async def _on_preempted(self, _, event):
        if event.instance.node != self.id:
            return
        await self.replace(event.reason)
```

### `providers/aws/handler.py`

```python
"""AWS command handler."""

@component
class AWSHandler:
    bus: AsyncEventBus
    config: AWS

    _clusters: dict[str, AWSClusterState] = {}

    @on(ClusterRequested)
    async def handle_cluster(self, _, event):
        if event.provider != "aws":
            return
        # Create infra, emit ClusterProvisioned

    @on(InstanceRequested)
    async def handle_instance(self, _, event):
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return
        # Launch EC2, emit InstanceProvisioned, InstanceBootstrapped

    @on(ShutdownRequested)
    async def handle_shutdown(self, _, event):
        cluster = self._clusters.pop(event.cluster_id, None)
        if not cluster:
            return
        # Destroy all, emit ClusterDestroyed
```

### `monitors.py`

```python
"""Background monitors."""

class MonitorModule(Module):

    @singleton
    @provider
    def provide_instance_registry(self) -> InstanceRegistry:
        return InstanceRegistry()

    @monitor(interval=5.0)
    async def check_preemption(
        self,
        registry: InstanceRegistry,
        checker: PreemptionChecker,
        bus: AsyncEventBus,
    ):
        """Check for spot preemptions."""
        for instance, reason in await checker.check(registry.instances):
            bus.emit(InstancePreempted(instance=instance, reason=reason))

    @monitor(interval=10.0)
    async def collect_metrics(
        self,
        registry: InstanceRegistry,
        collector: MetricsCollector,
        bus: AsyncEventBus,
    ):
        """Collect metrics from instances."""
        for instance in registry.instances:
            for name, value in await collector.collect(instance):
                bus.emit(Metric(instance=instance, name=name, value=value))
```

## Usage Example

```python
from skyward.v2 import (
    ComputePool, AWS, PoolSpec, ImageSpec,
    app_context, ClusterReady, InstancePreempted,
)
from injector import Module, provider, singleton

class AppModule(Module):
    @singleton
    @provider
    def provide_aws(self) -> AWS:
        return AWS(region="us-east-1")

    @singleton
    @provider
    def provide_spec(self) -> PoolSpec:
        return PoolSpec(
            nodes=4,
            accelerator="H100",
            image=ImageSpec(pip=["torch", "transformers"]),
        )

# Custom event handler
@component
class MyEventLogger:
    bus: AsyncEventBus

    @on(ClusterReady)
    async def log_ready(self, _, event: ClusterReady):
        print(f"Cluster {event.cluster_id} ready with {len(event.nodes)} nodes")

    @on(InstancePreempted)
    async def log_preemption(self, _, event: InstancePreempted):
        print(f"Node {event.instance.node} preempted: {event.reason}")

async def main():
    async with app_context(AppModule(), MonitorModule()) as app:
        pool = app.get(ComputePool)

        async with pool:
            # All nodes ready, monitors running
            result = await pool.run(train, dataset)

            # Broadcast to all nodes
            await pool.broadcast(load_checkpoint, "/data/model.pt")

asyncio.run(main())
```

## Implementation Order

1. **Phase 1: Core Infrastructure**
   - [ ] `events.py` - All event definitions
   - [ ] `bus.py` - AsyncEventBus
   - [ ] `app.py` - @component, @on, @monitor, create_app

2. **Phase 2: Domain Model**
   - [ ] `spec.py` - PoolSpec, ImageSpec
   - [ ] `protocols.py` - Instance, Transport protocols
   - [ ] `node.py` - Node component
   - [ ] `pool.py` - ComputePool component

3. **Phase 3: AWS Provider**
   - [ ] `providers/aws/config.py` - AWS config
   - [ ] `providers/aws/handler.py` - AWSHandler
   - [ ] `providers/aws/infra.py` - VPC, SG, Fleet
   - [ ] `providers/aws/instance.py` - Instance wrapper

4. **Phase 4: Transport**
   - [ ] `transport/ssh.py` - AsyncSSH
   - [ ] `transport/rpyc.py` - RPyC connection

5. **Phase 5: Bootstrap**
   - [ ] `bootstrap/script.py` - Script composition
   - [ ] `bootstrap/ops.py` - Operations

6. **Phase 6: Monitors**
   - [ ] `monitors.py` - Preemption, health, metrics

7. **Phase 7: Other Providers**
   - [ ] `providers/vastai.py`
   - [ ] `providers/digitalocean.py`
   - [ ] `providers/verda.py`

## Key Design Decisions

1. **@component replaces @dataclass** - Auto-generates `__init__`, applies DI, wires handlers
2. **Events are the only communication** - No direct method calls between components
3. **Node is autonomous** - Manages its own instance lifecycle, including replacement
4. **Pool only coordinates** - Creates nodes, tracks readiness, doesn't manage instances directly
5. **Providers are handlers** - React to requests, emit facts, don't know about Pool/Node
6. **Monitors are functions** - Simple @monitor decorated functions in modules
