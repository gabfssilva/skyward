# Architecture

Deep dive into Skyward's internal design.

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
3. WorkerPool tracks connections per worker
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
- WorkerPool manages RPyC connections
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
├── digitalocean/
│   └── provider.py
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
└── pool.py                # WorkerPool, connection management
```

**Isolation Mechanisms:**
- **cgroups v2**: CPU/memory limits per worker
- **MIG partitioning**: GPU isolation
- **CUDA_VISIBLE_DEVICES**: Device assignment
- **Separate RPyC servers**: Process isolation

### Event System

```
skyward/events.py          # ADT event definitions
skyward/callback.py        # Callback management
skyward/callbacks/
├── log.py                 # Logging callback
├── cost.py                # Cost tracking
└── spinner.py             # Progress spinner
```

**Event Categories:**

| Phase | Events |
|-------|--------|
| Provision | InfraCreating, InstanceLaunching, InstanceProvisioned |
| Setup | BootstrapStarting, BootstrapProgress, BootstrapCompleted |
| Execute | LogLine, Metrics |
| Shutdown | InstanceStopping, PoolStopping |
| Cost | CostUpdate, CostFinal |
| Errors | Error |

**Callback Pattern:**

```python
type Callback = Callable[[SkywardEvent], SkywardEvent | None]

def my_callback(event: SkywardEvent) -> SkywardEvent | None:
    match event:
        case Metrics(cpu_percent=cpu):
            print(f"CPU: {cpu}%")
    return None  # or return derived event
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
│ WorkerPool  │ ─────────────► │ Worker 0    │
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
├── distributed.py         # Framework decorators
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
│   ├── digitalocean/
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
| `distributed.py` | Framework integration decorators |
| `bootstrap/compose.py` | Script generation DSL |
| `worker/partition.py` | MIG partitioning logic |
| `providers/common.py` | Shared provider utilities |

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

1. Add decorator in `skyward/distributed.py`
2. Implement env var builder and initializer
3. Export in module `__all__`
