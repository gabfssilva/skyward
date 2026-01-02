# Core Concepts

This guide explains the fundamental concepts behind Skyward's programming model.

## Ephemeral Compute

Skyward implements **ephemeral compute** — GPU resources that exist only for the duration of your training job.

### The ML Infrastructure Problem

You want to train a model. Here's what you actually do:

1. Provision instance via console/Terraform (hours)
2. SSH in, debug connection issues (minutes to hours)
3. Install CUDA, cuDNN, PyTorch, your dependencies (hours)
4. Discover version mismatch, reinstall (more hours)
5. Finally run training (the actual work)
6. Job finishes at 3am
7. Instance runs idle until Monday
8. $800 wasted on idle compute

**This is backwards.** You're a researcher — your job is to improve models, not manage infrastructure. Yet infrastructure consumes 40%+ of ML engineering time.

### Ephemeral: Infrastructure That Matches Your Workflow

Training jobs have a **defined end**. Your infrastructure should too.

```python
with ComputePool(
    provider=AWS(),
    accelerator="H100",
    nodes=4,
    image=Image(pip=["torch", "transformers"]),
) as pool:
    # 4x H100 instances appear, configured and ready
    metrics = train_llm(dataset) >> pool
    # Job complete — instances terminate automatically
```

The infrastructure lifecycle matches the job lifecycle:

```
┌─────────────────────────────────────────────────┐
│  YOUR CODE          │  INFRASTRUCTURE           │
├─────────────────────┼───────────────────────────┤
│  with ComputePool   │  → Instances launched     │
│                     │  → PyTorch installed      │
│                     │  → NCCL configured        │
│  train() >> pool    │  → Training runs          │
│  (end of block)     │  → Instances terminated   │
└─────────────────────┴───────────────────────────┘
```

**Nothing persists. Nothing to forget. Nothing to clean up.**

### Why Ephemeral is Perfect for ML

Machine learning workloads have unique characteristics that make ephemeral compute ideal:

**1. Jobs Have a Defined End**

Unlike web servers, training jobs complete:
- Fine-tuning run: 2-8 hours
- Pretraining run: days to weeks
- Hyperparameter sweep: hours per trial
- Batch inference: minutes to hours

When the job ends, why should the infrastructure continue?

**2. Experiments Should Be Reproducible**

"It worked on my GPU server" is the ML equivalent of "works on my machine."

Persistent servers accumulate state:
- Random pip installs during debugging
- Cached datasets in /tmp
- Environment variables set months ago
- CUDA versions that silently changed

Ephemeral compute starts fresh every time:
```python
# Run 1: Clean Ubuntu → your Image → training
# Run 2: Clean Ubuntu → your Image → training
# Identical environments, reproducible results
```

**3. GPU Idle Costs Are Brutal**

| Instance | Hourly Cost | Weekend Idle Cost |
|----------|-------------|-------------------|
| g5.xlarge (A10G) | $1.00 | $48 |
| p4d.24xlarge (8x A100) | $32.77 | $1,573 |
| p5.48xlarge (8x H100) | $98.32 | $4,719 |

One forgotten instance can cost more than a month of your cloud budget.

**4. Different Experiments Need Different Resources**

Monday: Quick prototype on a T4
Tuesday: Full training on 4x A100
Wednesday: Distributed training on 8 nodes
Thursday: Inference benchmarks on H100

With persistent infrastructure, you either:
- Maintain multiple server configurations (ops burden)
- Over-provision and waste money
- Under-provision and wait for scaling

With ephemeral compute:
```python
# Each experiment gets exactly what it needs
ComputePool(accelerator="T4")           # Cheap prototyping
ComputePool(accelerator="A100", nodes=4) # Serious training
ComputePool(accelerator="H100", nodes=8) # Scale out
```

### Ephemeral vs Serverless for ML

You might wonder: "Why not Lambda or Cloud Functions?"

| | Serverless | Ephemeral (Skyward) |
|--|------------|---------------------|
| GPU support | None/Limited | Full (T4 to H100) |
| Max runtime | 15 minutes | Days |
| Memory | 10GB max | Up to 2TB |
| Control | None | Full (instance type, spot, etc.) |
| Distributed | Complex | Built-in (DDP, FSDP) |
| Cold start | Every call | Once per pool |

Serverless is great for web APIs. It's not designed for:
- Loading 70B parameter models into VRAM
- 8-hour training runs
- Multi-node distributed training
- GPU memory management

Skyward gives you **serverless ergonomics** (no infrastructure management) with **full GPU control** (pick your hardware, run for hours).

### Ephemeral vs Managed Platforms

Platforms like SageMaker, Vertex AI, and Azure ML also manage infrastructure. How is Skyward different?

| | Managed Platforms | Skyward |
|--|-------------------|---------|
| Definition | YAML/JSON configs | Python code |
| Vendor lock-in | High | Low (multi-cloud) |
| Framework support | Their SDK | Any Python code |
| Local testing | Limited | Full (`fn.local()`) |
| Debugging | Logs only | Interactive |
| Cost | Platform markup | Direct cloud pricing |

Managed platforms wrap your code in their abstractions. Skyward runs your code as-is — the only change is a decorator.

```python
# Your existing training code
def train(config):
    model = load_model(config.model_name)
    model.fit(config.dataset)
    return model.evaluate()

# On Skyward: add one decorator, nothing else changes
@compute
def train(config):
    model = load_model(config.model_name)
    model.fit(config.dataset)
    return model.evaluate()
```

### When to Use Ephemeral Compute

**Perfect for:**
- Training runs (fine-tuning, pretraining)
- Hyperparameter sweeps (GridSearchCV, Optuna)
- Batch inference (process dataset, generate embeddings)
- Distributed training (multi-GPU, multi-node)
- CI/CD for ML (test training pipelines)
- Research experiments (quick iterations)

**Not designed for:**
- Serving models (use inference endpoints)
- Real-time APIs (use Lambda/Cloud Run)
- Databases (use managed databases)
- Long-running services (use ECS/Kubernetes)

If your workload **starts, runs, and ends** — it's a fit for ephemeral compute.

## The @compute Decorator

The `@compute` decorator transforms a regular Python function into a **lazy computation**. When you call a decorated function, it doesn't execute immediately—instead, it returns a `PendingCompute` object that can be sent to a pool for execution.

```python
from skyward import compute

@compute
def train(data: list[int]) -> float:
    return sum(data) / len(data)

# Calling train() returns PendingCompute[float], not float
pending = train([1, 2, 3, 4])
print(type(pending))  # <class 'PendingCompute'>

# Execution happens when sent to a pool
with ComputePool(provider=AWS()) as pool:
    result = pending >> pool  # Now it executes, returns 2.5
```

### Why Lazy Evaluation?

Lazy evaluation enables:

1. **Serialization**: The function and arguments are serialized and sent to remote workers
2. **Batching**: Multiple computations can be grouped for parallel execution
3. **Type Safety**: Return types are preserved through the pipeline
4. **Resource Efficiency**: You control exactly when and where execution happens

### Local Execution

If you need to run the original function locally (for testing or debugging), access it via `.local`:

```python
@compute
def process(x: int) -> int:
    return x * 2

# Local execution (bypasses serialization)
result = process.local(10)  # Returns 20 directly
```

## PendingCompute

`PendingCompute[R]` represents a deferred computation that will return type `R` when executed.

```python
@compute
def add(a: int, b: int) -> int:
    return a + b

pending: PendingCompute[int] = add(1, 2)
```

### Properties

- **Immutable**: PendingCompute is a frozen dataclass
- **Serializable**: Can be pickled and sent to remote workers
- **Type-preserving**: Generic type `R` flows through to the result

## ComputePool

`ComputePool` is a context manager that provisions cloud resources and executes computations.

```python
from skyward import ComputePool, AWS, Image

pool = ComputePool(
    provider=AWS(),                    # Cloud provider
    nodes=2,                           # Number of instances
    accelerator="A100",                # GPU type
    image=Image(pip=["torch"]),        # Dependencies
    spot="always",                     # Use spot instances
    timeout=3600,                      # Auto-shutdown after 1 hour
)

with pool:
    result = my_function() >> pool
# Resources automatically released
```

### Lifecycle

1. **Enter**: Provisions instances, installs dependencies, starts RPC servers
2. **Execute**: Receives computations, serializes, sends to workers, returns results
3. **Exit**: Terminates instances, emits cost summary

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | `Provider` | AWS, DigitalOcean, or Verda |
| `nodes` | `int` | Number of instances (default: 1) |
| `accelerator` | `str` or `Accelerator` | GPU specification |
| `image` | `Image` | Environment (pip, apt, env vars) |
| `spot` | `SpotLike` | Spot instance strategy |
| `timeout` | `int` | Auto-shutdown in seconds |
| `volume` | `Sequence[Volume]` | Mounted volumes |
| `display` | `str` | "log", "spinner", or "quiet" |
| `on_event` | `Callback` | Custom event handler |

### MultiPool

For managing multiple pools with different configurations, use `MultiPool`:

```python
from skyward import ComputePool, MultiPool, AWS

with MultiPool(
    ComputePool(provider=AWS(), accelerator="T4"),
    ComputePool(provider=AWS(), accelerator="A100"),
) as (t4_pool, a100_pool):
    # Pools provisioned in parallel
    result_t4 = benchmark() >> t4_pool
    result_a100 = benchmark() >> a100_pool
```

`MultiPool` provisions all pools concurrently, reducing total setup time from `sum(t_i)` to `max(t_i)`.

## Execution Operators

Skyward uses Python operators to control how computations are executed.

### `>>` (Single Execution)

Execute on one worker, return single result:

```python
result = my_function(x) >> pool
```

### `@` (Broadcast)

Execute on ALL workers, return tuple of results:

```python
# With 4 nodes, returns tuple of 4 results
results = my_function(x) @ pool

# Useful for initialization
models = load_model(path) @ pool  # Load on all nodes
```

### `&` (Parallel Chain)

Chain multiple computations for parallel execution with type safety:

```python
# Full type inference
a: int
b: float
c: str
a, b, c = (fn1() & fn2() & fn3()) >> pool

# Supports up to 8 chained computations
r1, r2, r3, r4, r5 = (f1() & f2() & f3() & f4() & f5()) >> pool
```

### `gather()` (Dynamic Parallel)

Group any number of computations for parallel execution:

```python
from skyward import gather

# Fixed number
r1, r2, r3 = gather(fn(1), fn(2), fn(3)) >> pool

# Dynamic (e.g., from list comprehension)
results = gather(*[process(x) for x in data]) >> pool
```

## Image

`Image` specifies the execution environment on remote workers.

```python
from skyward import Image

image = Image(
    python="3.13",                    # Python version
    pip=["torch", "numpy"],           # pip packages
    apt=["ffmpeg", "libsndfile1"],    # apt packages
    env={"KERAS_BACKEND": "jax"},     # Environment variables
)

pool = ComputePool(provider=AWS(), image=image)
```

## Data Sharding

For distributed workloads, Skyward provides automatic data partitioning.

### `shard()`

Automatically partition data based on the current worker:

```python
from skyward import compute, shard, instance_info

@compute
def process_data(full_dataset: list[int]) -> int:
    # shard() returns only this worker's portion
    local_data = shard(full_dataset)
    return sum(local_data)

with ComputePool(provider=AWS(), nodes=4) as pool:
    # Each node processes 1/4 of the data
    results = process_data(list(range(1000))) @ pool
    total = sum(results)
```

### Multiple Arrays

Shard multiple arrays consistently:

```python
@compute
def train(x_full, y_full):
    x_local, y_local = shard(x_full, y_full, shuffle=True, seed=42)
    # x_local and y_local have matching indices
    return fit(x_local, y_local)
```

### `DistributedSampler`

PyTorch DataLoader integration:

```python
from skyward import DistributedSampler
from torch.utils.data import DataLoader

@compute
def train(epochs: int):
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for shuffling
        for batch in loader:
            train_step(batch)
```

## Cluster Information

Access information about the current execution environment:

```python
from skyward import compute, instance_info, InstanceInfo

@compute
def worker_role() -> str:
    info: InstanceInfo = instance_info()

    if info.is_head:
        return "I am the head node"
    else:
        return f"I am worker {info.node} of {info.total_nodes}"
```

### InstanceInfo Properties

| Property | Type | Description |
|----------|------|-------------|
| `node` | `int` | Current node index (0-based) |
| `total_nodes` | `int` | Total number of nodes |
| `is_head` | `bool` | True if this is node 0 |
| `accelerators` | `list[str]` | Available accelerators |
| `head_addr` | `str` | Head node IP address |
| `head_port` | `int` | Head node port |
| `job_id` | `str` | Unique job identifier |

## Events and Callbacks

Skyward emits events throughout the execution lifecycle.

### Event Types

**Provision Phase:**
- `InfraCreating` - Infrastructure being created
- `InfraCreated` - Infrastructure ready
- `InstanceLaunching` - Instance starting
- `InstanceProvisioned` - Instance ready

**Setup Phase:**
- `BootstrapStarting` - Dependency installation starting
- `BootstrapProgress` - Installation progress
- `BootstrapCompleted` - Setup complete

**Execute Phase:**
- `PoolStarted` - Pool ready for computations
- `LogLine` - stdout from remote function
- `Metrics` - CPU/memory/GPU utilization

**Shutdown Phase:**
- `PoolStopping` - Pool shutting down
- `InstanceStopping` - Instance terminating
- `CostFinal` - Final cost summary

### Custom Callbacks

```python
from skyward import ComputePool, Metrics, LogLine

def my_callback(event):
    match event:
        case Metrics(cpu=cpu, gpu=gpu):
            print(f"CPU: {cpu}%, GPU: {gpu}%")
        case LogLine(line=line):
            print(f"[remote] {line}")

pool = ComputePool(
    provider=AWS(),
    on_event=my_callback,
)
```

### Using Callbacks

Pass a callback function to `on_event` to handle events:

```python
def my_handler(event):
    match event:
        case Metrics(cpu_percent=cpu, gpu_utilization=gpu):
            print(f"CPU: {cpu}%, GPU: {gpu}%")
        case LogLine(line=line):
            print(f"[remote] {line}")

with ComputePool(provider=AWS(), on_event=my_handler) as pool:
    result = train() >> pool
```

## Spot Instances

Reduce costs by 60-90% with spot instances:

```python
from skyward import Spot

# Always use spot (fail if unavailable)
ComputePool(spot="always")
ComputePool(spot=Spot.Always())

# Try spot, fall back to on-demand
ComputePool(spot="if-available")
ComputePool(spot=Spot.IfAvailable())

# Never use spot
ComputePool(spot="never")
ComputePool(spot=Spot.Never)

# At least 80% spot instances
ComputePool(spot=0.8)
ComputePool(spot=Spot.Percent(0.8))
```

## Volumes

Mount external storage:

```python
from skyward import S3Volume

pool = ComputePool(
    provider=AWS(),
    volume=[
        S3Volume(
            mount_path="/data",
            bucket="my-bucket",
            prefix="datasets/",
            read_only=True,
        ),
        S3Volume(
            mount_path="/checkpoints",
            bucket="my-bucket",
            prefix="models/",
            read_only=False,
        ),
    ],
)
```

## Next Steps

- [API Reference](api-reference.md) - Complete API documentation
- [Distributed Training](distributed-training.md) - Multi-GPU training guides
- [Examples](examples.md) - Working code examples
