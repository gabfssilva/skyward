# Architecture

Deep dive into Skyward's internal design.

## Design Philosophy: Ephemeral Compute for ML

Skyward is built on a core principle: **GPU resources should not outlive your training job**.

### Why ML Needs Ephemeral Infrastructure

Traditional infrastructure models were designed for web services — always-on servers handling unpredictable traffic. ML workloads are fundamentally different:

| Web Services | ML Training |
|--------------|-------------|
| Run 24/7 | Run hours to days |
| Unpredictable load | Predictable workload |
| Scale on demand | Fixed resources per job |
| Stateless requests | Stateful training loop |
| CPU-bound | GPU-bound |

Applying "always-on" infrastructure patterns to ML creates waste:

```
Training job: ████████░░░░░░░░░░░░░░░░ 8 hours
Server life:  ████████████████████████ 168 hours (1 week)
Utilization:  4.7%
```

Ephemeral compute aligns infrastructure with workload:

```
Training job: ████████ 8 hours
Server life:  ████████ 8 hours
Utilization:  100%
```

### The Ephemeral ML Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                    EPHEMERAL ML LIFECYCLE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   with ComputePool(accelerator="A100", nodes=4) as pool:         │
│        │                                                         │
│        ▼                                                         │
│   ┌──────────────┐                                               │
│   │   PROVISION  │  Launch 4x p4d.24xlarge instances            │
│   │              │  ~60 seconds                                  │
│   └──────┬───────┘                                               │
│          ▼                                                       │
│   ┌──────────────┐                                               │
│   │   BOOTSTRAP  │  Install PyTorch, CUDA, your deps            │
│   │              │  Configure NCCL for multi-node                │
│   │              │  ~2-5 minutes (cached after first run)        │
│   └──────┬───────┘                                               │
│          ▼                                                       │
│   ┌──────────────┐                                               │
│   │    CONNECT   │  Establish tunnels to all nodes              │
│   │              │  Initialize distributed process group         │
│   └──────┬───────┘                                               │
│          ▼                                                       │
│   ┌──────────────┐                                               │
│   │    TRAIN     │  @torch()                                    │
│   │              │  def train(): ...                             │
│   │              │  train() @ pool  # Runs on all 4 nodes       │
│   │              │  Hours of training...                         │
│   └──────┬───────┘                                               │
│          ▼                                                       │
│   ┌──────────────┐                                               │
│   │  TERMINATE   │  Close connections                           │
│   │              │  Terminate all instances                      │
│   │              │  Report: "4x A100, 6.2 hours, $786.32"       │
│   └──────────────┘                                               │
│                                                                  │
│   # Instances gone. Checkpoints saved to S3. Nothing lingers.    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Safety Nets for Expensive GPUs

Even with context managers, things go wrong. Skyward includes multiple safety nets:

**1. Timeout Auto-Termination**

```python
ComputePool(
    accelerator="H100",
    timeout=14400,  # 4 hours max, then auto-terminate
)
```

The instance itself has a self-destruct timer. Even if your laptop dies, your network drops, or Python crashes — the GPU will terminate.

**2. Spot Instance Handling**

```python
ComputePool(
    accelerator="A100",
    allocation="always-spot",  # 60-90% cheaper
)
```

Spot instances can be reclaimed by AWS. Skyward handles interruptions gracefully — your code gets a chance to save checkpoints before termination.

**3. Cost Tracking**

Every pool emits real-time cost updates:

```
[cost] Running: $12.34/hr (4x A100 spot @ $3.08/hr each)
[cost] Elapsed: 2.5 hours, Total: $30.85
[cost] Final: 6.2 hours, $76.23 (saved $298.41 vs on-demand)
```

No surprise bills. You know exactly what you're spending.

### What Persists, What Doesn't

Ephemeral infrastructure doesn't mean ephemeral results:

| Ephemeral (gone after job) | Persistent (you manage) |
|---------------------------|-------------------------|
| EC2 instances | S3 checkpoints |
| Installed packages | Saved models |
| /tmp files | Training logs |
| Process state | Metrics (W&B, MLflow) |
| Network config | Your code |

Your infrastructure is disposable. Your artifacts are not.

```python
@compute
def train():
    model = train_model()

    # These persist beyond the ephemeral infrastructure:
    torch.save(model, "/mnt/s3/checkpoints/model.pt")  # S3 volume
    wandb.log(metrics)                                  # External service

    return metrics  # Returned to your local machine
```

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Local Machine                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   @compute              ComputePool              Provider            │
│   decorator   ──────►   context mgr   ──────►   (AWS/DO/Verda)      │
│       │                     │                       │                │
│       ▼                     │                       ▼                │
│   PendingCompute           │                  provision()            │
│       │                     │                  setup()               │
│       │     >> or @         │                  shutdown()            │
│       └─────────────────────┘                       │                │
│                                                     │                │
└─────────────────────────────────────────────────────│────────────────┘
                                                      │
                              ┌───────────────────────┘
                              │ SSH/SSM
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Cloud Instance(s)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Bootstrap Script        RPyC Server           Worker Process       │
│   (cloud-init)   ──────►  (systemd)   ──────►   (isolated)          │
│       │                       │                     │                │
│       ▼                       ▼                     ▼                │
│   Install deps           Recv fn bytes         Execute fn           │
│   Start server           Deserialize           Return result        │
│   Mount volumes          Route to worker       stdout → events      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Execution Flow

### Phase 1: Provision

1. **ComputePool.__enter__()** called
2. Provider selects instance type matching accelerator/cpu/memory
3. Instance(s) launched via provider API (EC2, DO, etc.)
4. Events emitted: `InfraCreating`, `InstanceLaunching`, `InstanceProvisioned`

### Phase 2: Bootstrap

1. User-data script generated from `Image`
2. Cloud-init executes on instance boot
3. Dependencies installed via `uv` (fast, cached)
4. Skyward RPC server started via systemd
5. Events: `BootstrapStarting`, `BootstrapProgress`, `BootstrapCompleted`

### Phase 3: Connect

1. Local machine creates SSH tunnel (or SSM tunnel for AWS)
2. RPyC connection established through tunnel
3. TaskPool tracks connections per instance
4. Cluster info sent to each worker on first use

### Phase 4: Execute

1. `pending >> pool` triggers execution
2. Function serialized with cloudpickle
3. Sent to remote worker via RPyC
4. Worker deserializes and executes
5. Result serialized and returned
6. Events: `LogLine` (stdout), `Metrics` (CPU/GPU)

### Phase 5: Shutdown

1. `ComputePool.__exit__()` called
2. RPC connections closed
3. Instances terminated
4. Events: `InstanceStopping`, `PoolStopping`, `CostFinal`

## Component Architecture

### Lazy Computation Layer

```
skyward/pending.py
├── @compute decorator
├── PendingCompute[R]      # Deferred computation
├── PendingBatch           # Grouped computations
├── PendingBatch2-8        # Typed batches for & operator
├── gather()               # Group with type safety
└── ComputeFunction        # Wrapper for decorated fn
```

**Key Design:**
- Immutable dataclasses (`frozen=True`)
- Generic type parameters preserved through pipeline
- Overloaded operators for ergonomic API

### Pool Management

```
skyward/pool.py
├── ComputePool
│   ├── __enter__()        # Provision resources
│   ├── __exit__()         # Release resources
│   ├── run()              # Single execution
│   ├── run_batch()        # Parallel execution
│   └── broadcast()        # All-worker execution
└── _PoolCompute           # Internal compute spec
```

**Key Design:**
- Context manager pattern for resource lifecycle
- TaskPool manages RPyC connections
- Round-robin distribution for batches

### Provider Abstraction

```
skyward/providers/
├── base/
│   ├── capabilities.py    # Provider capability detection
│   ├── mixins.py          # Shared functionality
│   ├── ssh_keys.py        # SSH key management
│   └── transport.py       # SSH/SFTP transport
├── aws/
│   ├── provider.py        # AWS implementation
│   ├── discovery.py       # Instance type discovery
│   ├── fleet.py           # EC2 Fleet management
│   └── ssm.py             # Session Manager support
└── verda/
    └── provider.py
```

**Provider Protocol:**

```python
class Provider(Protocol):
    def provision(compute: ComputeSpec) -> tuple[Instance, ...]
    def setup(instances: tuple[Instance, ...], compute: ComputeSpec) -> None
    def shutdown(instances: tuple[Instance, ...], compute: ComputeSpec) -> tuple[ExitedInstance, ...]
    def create_tunnel(instance: Instance, remote_port: int) -> tuple[int, Popen]
    def run_command(instance: Instance, command: str) -> str
    def available_instances() -> tuple[InstanceSpec, ...]
```

### Bootstrap System

```
skyward/bootstrap/
├── compose.py             # Script composition
├── ops.py                 # Core operations (apt, pip, uv)
├── control.py             # Control flow (when, unless)
├── worker.py              # Worker setup ops
└── unified.py             # Unified generator
```

**Declarative DSL:**

```python
script = bootstrap(
    instance_timeout(3600),
    env_export(CUDA_VISIBLE_DEVICES="0"),
    install_uv(),
    apt("python3", "curl"),
    uv_add("torch", "transformers"),
    checkpoint(".ready"),
)
```

**Key Design:**
- Functional composition of operations
- Idempotent scripts (safe to re-run)
- Content-addressable hashing for AMI caching

### Worker Isolation

```
skyward/worker/
├── config.py              # WorkerConfig, ResourceLimits
├── partition.py           # GPU partitioning strategies
└── pool.py                # TaskPool, connection management
```

**Isolation Mechanisms:**
- **cgroups v2**: CPU/memory limits per worker
- **MIG partitioning**: GPU isolation
- **CUDA_VISIBLE_DEVICES**: Device assignment
- **Separate RPyC servers**: Process isolation

### Event System

```
skyward/events.py          # Event definitions (27+ types)
skyward/bus.py             # AsyncEventBus (blinker-based)
skyward/app.py             # @component, @on, @monitor decorators
skyward/orchestrator.py    # InstanceOrchestrator for event pipeline
```

**Event-Driven Architecture:**

Skyward uses a fully async event system built on **blinker**. All component communication happens through events, never direct method calls.

**Event Categories:**

| Category | Events |
|----------|--------|
| Requests | ClusterRequested, InstanceRequested, ShutdownRequested, BootstrapRequested |
| Cluster | ClusterProvisioned, ClusterReady, ClusterDestroyed |
| Pipeline | InstanceLaunched, InstanceRunning |
| Lifecycle | InstanceProvisioned, InstanceBootstrapped, InstancePreempted, InstanceReplaced, InstanceDestroyed |
| Nodes | NodeReady |
| Execution | TaskStarted, TaskCompleted |
| Bootstrap | BootstrapConsole, BootstrapPhase, BootstrapCommand, BootstrapFailed |
| Observability | Metric, Log, Error |

**Event Pipeline Pattern:**

A 3-stage decoupled pipeline for instance provisioning:

```
PROVIDER LAYER              ORCHESTRATOR LAYER         POOL/NODE LAYER
─────────────────          ──────────────────         ─────────────────
InstanceLaunched ──→ InstanceOrchestrator ──→ InstanceProvisioned
(launch API call)       (transforms event)       (Node tracks state)

InstanceRunning ──→ InstanceOrchestrator ──→ BootstrapRequested
(IP available)        (builds InstanceInfo)    (provider handles)
```

Benefits:
- Providers don't duplicate bootstrap logic - orchestrator is generic
- Clear separation between provider-specific and lifecycle events
- Enables consistent observability across all providers

**AsyncEventBus Emission Patterns:**

```python
# Fire-and-forget (most common)
bus.emit(InstancePreempted(...))

# Await all handlers
await bus.emit_await(ShutdownRequested(...))

# Request/Response correlation
response = await bus.request(
    command=ClusterRequested(request_id="abc"),
    response_type=ClusterProvisioned,
    match=lambda r: r.request_id == "abc"
)
```

**Handler Pattern with @on:**

```python
from skyward import component, on

@component
class Node:
    bus: AsyncEventBus  # Injected

    @on(InstanceProvisioned)
    async def handle(self, sender, event):
        ...

    # With match filter
    @on(InstancePreempted, match=lambda self, e: e.instance.node == self.id)
    async def handle_preempted(self, sender, event):
        ...

    # Disable audit for noisy handlers
    @on(Metric, audit=False)
    async def handle_metric(self, sender, event):
        ...
```

**MonitorManager:**

```python
from skyward import MonitorManager, monitor

@monitor(interval=5.0, name="preemption")
async def check_preemption(registry: InstanceRegistry, bus: AsyncEventBus):
    for instance in registry.spot_instances:
        if await is_preempted(instance):
            bus.emit(InstancePreempted(...))

manager = MonitorManager()
manager.start("preemption", check_preemption, interval=5.0, injector=inj)
manager.stop_all()
```

## Data Flow

### Serialization

```
Local                           Remote
┌─────────────┐                ┌─────────────┐
│ fn          │                │ fn          │
│ args        │  cloudpickle   │ args        │
│ kwargs      │ ─────────────► │ kwargs      │
└─────────────┘                └─────────────┘
                                     │
                                     ▼
                               ┌─────────────┐
                               │  Execute    │
                               └─────────────┘
                                     │
                                     ▼
┌─────────────┐                ┌─────────────┐
│ result      │  cloudpickle   │ result      │
└─────────────┘ ◄───────────── └─────────────┘
```

### Transport

```
Local                           Remote
┌─────────────┐                ┌─────────────┐
│ ComputePool │                │ RPyC Server │
│     │       │                │     │       │
│     ▼       │  SSH Tunnel    │     ▼       │
│ TaskPool    │ ─────────────► │ Worker 0    │
│     │       │                │ Worker 1    │
│     ▼       │                │ Worker N    │
│ RPyC Client │                └─────────────┘
└─────────────┘
```

## Design Patterns

### 1. Protocol-Based Providers

No inheritance hierarchy—pure duck typing:

```python
class Provider(Protocol):
    name: str
    def provision(...) -> tuple[Instance, ...]: ...
    def setup(...) -> None: ...
    def shutdown(...) -> tuple[ExitedInstance, ...]: ...
```

### 2. Functional Bootstrap

Operations compose without side effects:

```python
script = bootstrap(
    op1(),
    op2(),
    op3(),
)
# No execution until bootstrap() renders
```

### 3. Algebraic Data Types for Events

Pattern matching for type-safe handling:

```python
match event:
    case Metrics(cpu_percent=cpu, gpu_utilization=gpu):
        handle_metrics(cpu, gpu)
    case Error(message=msg):
        handle_error(msg)
```

### 4. Lazy Evaluation with Types

Full type inference through the pipeline:

```python
@compute
def add(a: int, b: int) -> int: ...

pending: PendingCompute[int] = add(1, 2)
result: int = pending >> pool
```

### 5. Context Managers for Resources

Automatic cleanup on exit:

```python
with ComputePool(...) as pool:
    # Resources allocated
    result = fn() >> pool
# Resources released (even on exception)
```

## Directory Structure

```
skyward/
├── __init__.py            # Public API exports
├── pending.py             # Lazy computation primitives
├── pool.py                # ComputePool
├── image.py               # Image specification
├── accelerator.py         # GPU types and MIG
├── integrations/          # Framework integrations (keras, torch, etc.)
├── types.py               # Core types (Instance, Provider, etc.)
├── events.py              # Event ADT
├── callback.py            # Callback system
├── cluster.py             # instance_info()
├── data.py                # shard(), DistributedSampler
├── metrics.py             # Metrics polling
├── volume.py              # S3Volume
├── spec.py                # Spot strategies
├── serialization.py       # cloudpickle wrappers
├── exceptions.py          # Custom exceptions
├── conc.py                # Concurrency utilities
│
├── bootstrap/             # Bootstrap script generation
├── callbacks/             # Built-in callbacks
├── worker/                # Worker isolation
├── providers/             # Cloud providers
│   ├── base/              # Shared abstractions
│   ├── aws/
│   └── verda/
└── rpc/                   # RPC server
```

## Key Files

| File | Purpose |
|------|---------|
| `pending.py` | Core lazy API (@compute, gather, operators) |
| `pool.py` | Resource management (ComputePool) |
| `types.py` | Instance, Provider protocol, accelerators |
| `events.py` | Event definitions (ADT) |
| `integrations/` | Framework integration decorators |
| `bootstrap/compose.py` | Script generation DSL |
| `task/pool.py` | TaskPool connection management |
| `providers/common.py` | Shared provider utilities (tunnels, bootstrap) |

## Extension Points

### Adding a Provider

1. Create `skyward/providers/myprovider/`
2. Implement `Provider` protocol
3. Export in `skyward/providers/__init__.py`
4. Add to `skyward/__init__.py` exports

### Adding an Event

1. Define dataclass in `skyward/events.py`
2. Add to `SkywardEvent` union type
3. Emit with `emit(MyEvent(...))`

### Adding a Framework Decorator

1. Create new file in `skyward/integrations/` (e.g., `myframework.py`)
2. Implement env var builder and initializer
3. Export in `skyward/integrations/__init__.py`

---

## Related Topics

- [Getting Started](getting-started.md) — First steps with Skyward
- [Core Concepts](concepts.md) — Understanding the programming model
- [Troubleshooting](troubleshooting.md) — Common issues and solutions