# API Reference

Complete reference for the Skyward API.

## Contents

- [Core API](#core-api) — @compute, PendingCompute, gather, ComputePool, @pool, MultiPool
- [Image](#image) — Environment specification
- [Providers](#providers) — AWS, DigitalOcean, Verda
- [Accelerators](#accelerators) — GPU types and MIG
- [Distributed Training](#distributed-training) — keras, torch, jax, tensorflow, transformers
- [Data Sharding](#data-sharding) — shard(), DistributedSampler
- [Cluster Information](#cluster-information) — instance_info(), InstanceInfo
- [Events](#events) — Provision, Setup, Execute, Shutdown, Cost events
- [Callbacks](#callbacks) — Callback type, compose(), emit()
- [Volumes](#volumes) — S3Volume
- [Allocation Strategies](#allocation-strategies) — always-spot, spot-if-available, on-demand, cheapest
- [Utility Functions](#utility-functions) — is_nvidia(), is_trainium(), etc.
- [Integrations](#integrations) — JoblibPool, ScikitLearnPool

---

## Core API

### @sky.compute

```python
import skyward as sky

@sky.compute
def my_function(x: int) -> int:
    return x * 2
```

Decorator that makes a function lazy. When called, returns `PendingCompute[R]` instead of executing.

**Returns:** `ComputeFunction[P, R]` - Callable that creates `PendingCompute` on call

**Properties:**
- `.local` - Access original function for local execution

---

### PendingCompute[R]

Represents a deferred computation.

```python
import skyward as sky

@sky.compute
def add(a: int, b: int) -> int:
    return a + b

pending: sky.PendingCompute[int] = add(1, 2)
```

**Operators:**

| Operator | Method | Result |
|----------|--------|--------|
| `>> pool` or `>> sky` | `__rshift__` | Execute, return `R` |
| `@ pool` or `@ sky` | `__matmul__` | Broadcast, return `tuple[R, ...]` |
| `& other` | `__and__` | Chain, return `PendingBatch` |

Note: `>> sky` and `@ sky` work inside `@sky.pool`-decorated functions.

---

### gather()

```python
import skyward as sky

sky.gather(*computations: sky.PendingCompute) -> sky.PendingBatch
```

Group computations for parallel execution.

**Parameters:**
- `*computations` - Variable number of `PendingCompute` objects

**Returns:** `PendingBatch` that executes all in parallel

**Example:**
```python
r1, r2, r3 = sky.gather(fn(1), fn(2), fn(3)) >> pool
```

---

### ComputePool

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.AWS(),
    nodes=2,
    accelerator="A100",
    image=sky.Image(pip=["torch"]),
    allocation="always-spot",
    timeout=3600,
)
```

Context manager for cloud resource management.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `ProviderLike` | **required** | Single provider or list for multi-provider |
| `selection` | `SelectionLike` | `"first"` | Provider selection strategy |
| `image` | `Image` | `Image()` | Environment specification |
| `nodes` | `int` | `1` | Number of instances |
| `machine` | `str` | `None` | Direct instance type override (e.g., "p5.48xlarge") |
| `accelerator` | `str \| Accelerator` | `None` | GPU specification |
| `cpu` | `int` | `None` | CPU cores per worker |
| `memory` | `str` | `None` | Memory per worker (e.g., "32GB") |
| `volume` | `Sequence[Volume]` | `None` | Mounted volumes |
| `allocation` | `AllocationLike` | `"spot-if-available"` | Instance allocation strategy |
| `timeout` | `int` | `3600` | Auto-shutdown seconds |
| `env` | `dict[str, str]` | `None` | Environment variables |
| `concurrency` | `int` | `1` | Concurrent tasks per instance |
| `display` | `str` | `"log"` | "log", "spinner", or "quiet" |
| `on_event` | `Callback` | `None` | Custom event handler |
| `collect_metrics` | `bool` | `True` | Enable metrics polling |

**Methods:**

```python
def run[R](pending: PendingCompute[R]) -> R
```
Execute single computation on first available worker.

```python
def run_batch(batch: PendingBatch) -> tuple[Any, ...]
```
Execute batch in parallel across workers.

```python
def broadcast[R](pending: PendingCompute[R]) -> tuple[R, ...]
```
Execute on ALL workers simultaneously.

**Properties:**
- `is_active: bool` - True if pool is provisioned
- `instance_count: int` - Number of instances

---

### Provider Selection

When using multiple providers, the `selection` parameter controls which one is tried first:

```python
import skyward as sky

pool = sky.ComputePool(
    provider=[sky.AWS(), sky.Verda()],
    selection="cheapest",
    accelerator="A100",
)
```

**Selection Strategies:**

| Strategy | Behavior |
|----------|----------|
| `"first"` | Use first provider in list (default) |
| `"cheapest"` | Compare prices across providers, pick lowest |
| `"available"` | First provider with matching instances |
| `callable` | Custom `(tuple[Provider, ...], ComputeSpec) -> Provider` |

**Automatic Fallback:** If the selected provider fails (no capacity, provisioning error), Skyward automatically tries the next provider in the list.

**Custom Selector Example:**

```python
def prefer_spot(providers, spec):
    """Prefer providers with spot capacity."""
    for p in providers:
        if p.has_spot_capacity(spec):
            return p
    return providers[0]

pool = sky.ComputePool(
    provider=[sky.AWS(), sky.Verda()],
    selection=prefer_spot,
    accelerator="A100",
)
```

**Types:**

| Type | Definition |
|------|------------|
| `ProviderLike` | `ProviderConfig \| Sequence[ProviderConfig]` |
| `SelectionLike` | `Literal["first", "cheapest", "available"] \| ProviderSelector` |
| `ProviderSelector` | `Callable[[tuple[Provider, ...], ComputeSpec], Provider]` |

**Built-in Selector Functions:**

```python
import skyward as sky

sky.select_first      # Use first provider
sky.select_cheapest   # Compare prices
sky.select_available  # First with matching instances
```

**Exceptions:**

| Exception | Description |
|-----------|-------------|
| `NoAvailableProviderError` | No provider has instances matching requirements |
| `AllProvidersFailedError` | All providers failed during provisioning |

---

### @sky.pool

```python
import skyward as sky

@sky.pool(
    provider=sky.AWS(),
    accelerator="A100",
    nodes=4,
)
def main():
    result = train(data) >> sky
    return result

main()  # provisions -> executes -> terminates
```

Decorator that provisions a pool for the duration of the function.

**Parameters:** Same as `ComputePool`

**Implicit Execution:**

Inside a `@pool`-decorated function, use `>> sky` instead of `>> pool`:

| Operator | Syntax | Description |
|----------|--------|-------------|
| `>>` | `fn() >> sky` | Execute on one worker |
| `@` | `fn() @ sky` | Broadcast to all workers |
| `&` | `fn1() & fn2() >> sky` | Parallel execution |
| `gather()` | `sky.gather(*fns) >> sky` | Dynamic parallel execution |

**When to use:**
- Simple, self-contained jobs with a single entry point
- Cleaner, more declarative code
- Most common use cases

**When to use `with ComputePool` instead:**
- Need pool access before execution (e.g., `pool.on()` for event handlers)
- Dynamic pool creation
- Advanced patterns

---

### MultiPool

```python
import skyward as sky

with sky.MultiPool(
    sky.ComputePool(provider=sky.AWS(), accelerator="T4"),
    sky.ComputePool(provider=sky.AWS(), accelerator="A100"),
) as (pool_t4, pool_a100):
    r1 = benchmark() >> pool_t4
    r2 = benchmark() >> pool_a100
```

Context manager that provisions multiple pools **in parallel**.

**Parameters:**
- `*pools` - Variable number of `ComputePool` instances

**Returns:** `tuple[ComputePool, ...]` - Tuple of provisioned pools

**Benefits:**
- Parallel provisioning reduces setup time from `sum(t_i)` to `max(t_i)`
- Automatic cleanup if any pool fails during provisioning
- Parallel shutdown on exit

**Use Cases:**
- Compare performance across different GPUs
- Run different workloads on different configurations
- Multi-provider workflows

---

## Image

```python
import skyward as sky

image = sky.Image(
    python="3.13",
    pip=["torch", "numpy"],
    apt=["ffmpeg"],
    env={"CUDA_VISIBLE_DEVICES": "0"},
)
```

Declarative environment specification.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `python` | `str` | `"3.13"` | Python version |
| `pip` | `list[str]` | `[]` | pip packages |
| `pip_extra_index_url` | `str` | `None` | Extra pip index URL |
| `apt` | `list[str]` | `[]` | apt packages |
| `env` | `dict[str, str]` | `{}` | Environment variables |

**Methods:**

```python
def content_hash() -> str
```
12-character hash for AMI/snapshot caching.

```python
def bootstrap(ttl: int = 0) -> str
```
Generate idempotent shell script.

---

## Providers

### AWS

```python
import skyward as sky

provider = sky.AWS(
    region="us-east-1",
    use_ssm=False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `"us-east-1"` | AWS region |
| `use_ssm` | `bool` | `False` | Use Session Manager for connectivity |

### DigitalOcean

```python
import skyward as sky

provider = sky.DigitalOcean(
    region="nyc1",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `"nyc1"` | DO region |

**Available Regions:** `nyc1`, `nyc3`, `sfo2`, `sfo3`, `ams3`, `sgp1`, `lon1`, `fra1`, `tor1`

### Verda

```python
import skyward as sky

provider = sky.Verda(
    region="us-east",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `None` | Region (auto-discovers if None) |

---

## Accelerators

### Accelerator Class

Factory class for type-safe accelerator specifications.

```python
import skyward as sky

# NVIDIA GPUs
sky.AcceleratorSpec.NVIDIA.H100()  # 1x H100 80GB
sky.AcceleratorSpec.NVIDIA.H100(count=4)  # 4x H100
sky.AcceleratorSpec.NVIDIA.H100(mig="3g.40gb")  # MIG partition
sky.AcceleratorSpec.NVIDIA.A100(memory="40GB")  # A100 40GB variant
sky.AcceleratorSpec.NVIDIA.T4()  # 1x T4

# AWS Accelerators
sky.AcceleratorSpec.AWS.Trainium(version=2)  # Trainium2
sky.AcceleratorSpec.AWS.Inferentia(version=2)  # Inferentia2

# Google TPUs
sky.AcceleratorSpec.Google.TPU(version="v5p")
sky.AcceleratorSpec.Google.TPUSlice("v5p-8")

# AMD GPUs
sky.AcceleratorSpec.AMD.MI("300X")
```

### NVIDIA Factory Methods

| Method | Memory | Notes |
|--------|--------|-------|
| `H100(count, memory, mig)` | 80GB/40GB | MIG capable |
| `H200(count)` | 141GB | |
| `A100(count, memory, mig)` | 80GB/40GB | MIG capable |
| `B100(count)` | 192GB | Blackwell |
| `B200(count)` | 192GB | Blackwell |
| `GB200(count)` | 384GB | Grace Blackwell |
| `GH200(count)` | 96GB | Grace Hopper |
| `L4(count)` | 24GB | |
| `L40(count)` | 48GB | |
| `L40S(count)` | 48GB | |
| `T4(count)` | 16GB | |
| `A10(count)` | 24GB | |
| `A10G(count)` | 24GB | |
| `V100(count, memory)` | 16GB/32GB | |

### MIG Profiles

**H100/A100 (80GB):**

| Profile | Workers | Memory Each |
|---------|---------|-------------|
| `1g.10gb` | 7 | 10GB |
| `1g.20gb` | 4 | 20GB |
| `2g.20gb` | 3 | 20GB |
| `3g.40gb` | 2 | 40GB |
| `4g.40gb` | 1 | 40GB |
| `7g.80gb` | 1 | 80GB |

**A100 (40GB):**

| Profile | Workers | Memory Each |
|---------|---------|-------------|
| `1g.5gb` | 7 | 5GB |
| `2g.10gb` | 3 | 10GB |
| `3g.20gb` | 2 | 20GB |
| `4g.20gb` | 1 | 20GB |

### NVIDIA Literal Type

```python
import skyward as sky

# Use as type hint
def train(gpu: sky.NVIDIA) -> None: ...

# Values
sky.NVIDIA = Literal[
    "T4", "L4", "L40", "L40S",
    "A10", "A10G",
    "A100-40GB", "A100-80GB",
    "H100-80GB", "H100-SXM", "H100-PCIe", "H100-NVL",
    "H200", "B100", "B200", "GB200", "GH200",
    ...
]
```

---

## Distributed Training

### keras

```python
import skyward as sky

@sky.integrations.keras(backend="jax")
@sky.compute
def train():
    import keras
    model = keras.Sequential([...])
    model.fit(x, y)
```

**Parameters:**
- `backend` - "jax", "tensorflow", or "torch"

### torch

```python
import skyward as sky

@sky.integrations.torch(backend="nccl")
@sky.compute
def train():
    import torch.distributed as dist
    # dist.is_initialized() is True
```

**Parameters:**
- `backend` - "nccl" (GPU) or "gloo" (CPU)

### jax

```python
import skyward as sky

@sky.integrations.jax()
@sky.compute
def train():
    import jax
    # jax.distributed already initialized
```

### tensorflow

```python
import skyward as sky

@sky.integrations.tensorflow()
@sky.compute
def train():
    import tensorflow as tf
    # TF_CONFIG set automatically
```

### transformers

```python
import skyward as sky

@sky.integrations.transformers(backend="nccl")
@sky.compute
def train():
    from transformers import Trainer
    # Trainer auto-detects distributed
```

---

## Data Sharding

### shard()

```python
import skyward as sky

def sky.shard(
    *arrays,
    shuffle: bool = False,
    seed: int | None = None,
) -> tuple | Any
```

Partition data based on current worker.

**Parameters:**
- `*arrays` - Arrays to shard (lists, tuples, numpy arrays)
- `shuffle` - Shuffle before sharding
- `seed` - Random seed for reproducibility

**Returns:** Single array if one input, tuple if multiple

**Example:**
```python
import skyward as sky

@sky.compute
def process(x_full, y_full):
    x_local, y_local = sky.shard(x_full, y_full, shuffle=True, seed=42)
    return train(x_local, y_local)
```

### DistributedSampler

```python
import skyward as sky

sampler = sky.DistributedSampler(
    dataset,
    shuffle=True,
)
```

PyTorch DataLoader integration.

**Parameters:**
- `dataset` - PyTorch Dataset
- `shuffle` - Shuffle samples

**Methods:**
- `set_epoch(epoch)` - Set epoch for proper shuffling

**Example:**
```python
import skyward as sky

sampler = sky.DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler)

for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        train_step(batch)
```

---

## Cluster Information

### instance_info()

```python
import skyward as sky

def sky.instance_info() -> sky.InstanceInfo
```

Get current execution environment information.

### InstanceInfo

```python
@dataclass
class InstanceInfo:
    node: int              # Current node index (0-based)
    total_nodes: int       # Total nodes in cluster
    is_head: bool          # True if node == 0
    accelerators: list[str] # Available accelerators
    head_addr: str         # Head node IP
    head_port: int         # Head node port
    job_id: str            # Unique job identifier
    worker: int            # Worker index within node
    workers_per_node: int  # Total workers per node
```

---

## Events

### Provision Phase

```python
@dataclass
class InfraCreating:
    pass

@dataclass
class InfraCreated:
    region: str

@dataclass
class InstanceLaunching:
    count: int
    instance_type: str
    provider: ProviderName

@dataclass
class InstanceProvisioned:
    instance_id: str
    node: int
    spot: bool
    provider: ProviderName
    ip: str | None
    instance_type: str | None
    price_on_demand: float | None
    price_spot: float | None
```

### Setup Phase

```python
@dataclass
class BootstrapStarting:
    instance_id: str

@dataclass
class BootstrapProgress:
    instance_id: str
    step: str  # "uv", "apt", "deps", "skyward", "server", "volumes"

@dataclass
class BootstrapCompleted:
    instance_id: str
```

### Execute Phase

```python
@dataclass
class Metrics:
    instance_id: str
    node: int
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    gpu_utilization: float | None
    gpu_memory_used_mb: float | None
    gpu_memory_total_mb: float | None
    gpu_temperature: float | None

@dataclass
class LogLine:
    node: int
    instance_id: str
    line: str
    timestamp: float
```

### Shutdown Phase

```python
@dataclass
class InstanceStopping:
    instance_id: str

@dataclass
class PoolStopping:
    pass
```

### Cost Events

```python
@dataclass
class CostUpdate:
    elapsed_seconds: float
    accumulated_cost: float
    hourly_rate: float
    spot_count: int
    ondemand_count: int

@dataclass
class CostFinal:
    total_cost: float
    total_seconds: float
    hourly_rate: float
    spot_count: int
    ondemand_count: int
    savings_vs_ondemand: float
```

### Error Event

```python
@dataclass
class Error:
    message: str
    instance_id: str | None
```

---

## Callbacks

### Callback Type

```python
type Callback = Callable[[SkywardEvent], SkywardEvent | None]
```

### compose()

```python
import skyward as sky

combined = sky.compose(callback1, callback2, callback3)
```

Merge multiple callbacks into one.

### emit()

```python
import skyward as sky

sky.emit(sky.Metrics(...))
```

Emit an event to all registered callbacks.

### use_callback()

```python
import skyward as sky

with sky.use_callback(my_callback):
    # Events in this context go to my_callback
    ...
```

Context manager for callback registration.

---

## Volumes

### S3Volume

```python
import skyward as sky

volume = sky.S3Volume(
    mount_path="/data",
    bucket="my-bucket",
    prefix="datasets/",
    read_only=True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mount_path` | `str` | **required** | Local mount path |
| `bucket` | `str` | **required** | S3 bucket name |
| `prefix` | `str` | `""` | S3 key prefix |
| `read_only` | `bool` | `True` | Mount read-only |

---

## Allocation Strategies

Control how instances are provisioned (spot vs on-demand):

```python
import skyward as sky

# String shortcuts
allocation = "spot-if-available"  # Default: try spot, fallback to on-demand
allocation = "always-spot"        # Must use spot, fail if unavailable
allocation = "on-demand"          # Always on-demand
allocation = "cheapest"           # Compare prices, pick cheapest option

# Float shortcut (minimum spot percentage)
allocation = 0.8                  # At least 80% spot

# Class instances
allocation = sky.Allocation.SpotIfAvailable()
allocation = sky.Allocation.AlwaysSpot()
allocation = sky.Allocation.OnDemand
allocation = sky.Allocation.Cheapest()
allocation = sky.Allocation.Percent(spot=0.8)
```

**Strategy comparison:**

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `spot-if-available` | Try spot, fallback to on-demand | Default, balanced |
| `always-spot` | Spot only, fail if unavailable | Maximum savings |
| `on-demand` | On-demand only | Critical workloads |
| `cheapest` | Compare spot vs on-demand prices | Optimal cost |
| `0.8` / `Percent(spot=0.8)` | At least 80% spot | Mixed allocation |

---

## Utility Functions

### is_nvidia()

```python
import skyward as sky

sky.is_nvidia("H100")  # True
sky.is_nvidia("MI300X")  # False
```

### is_trainium()

```python
import skyward as sky

sky.is_trainium("Trainium2")  # True
sky.is_trainium("H100")  # False
```

### current_accelerator()

```python
import skyward as sky

acc = sky.current_accelerator()  # Returns accelerator type in compute context
```

### select_instance()

```python
import skyward as sky

instance = sky.select_instance(
    available_instances,
    accelerator="H100",
    cpu=32,
    memory="128GB",
)
```

Find smallest matching instance from available options.

---

## Integrations

### JoblibPool

```python
import skyward as sky
from joblib import Parallel, delayed

with sky.integrations.JoblibPool(
    provider=sky.AWS(),
    nodes=4,
    concurrency=4,
    image=sky.Image(pip=["joblib"]),
) as pool:
    # Use joblib Parallel with n_jobs=-1
    results = Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)
```

Context manager that provisions a pool and auto-registers Skyward as joblib backend.

**Parameters:**

Same as `ComputePool`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `joblib_version` | `str` | `None` | Specific joblib version |

### ScikitLearnPool

```python
import skyward as sky
from sklearn.model_selection import GridSearchCV

with sky.integrations.ScikitLearnPool(
    provider=sky.AWS(),
    nodes=4,
    concurrency=4,
    image=sky.Image(pip=["scikit-learn"]),
):
    grid = GridSearchCV(estimator, params, n_jobs=-1)
    grid.fit(X, y)  # Distributed!
```

Context manager for distributed scikit-learn. Auto-adds sklearn to dependencies.

**Parameters:**

Same as `ComputePool`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sklearn_version` | `str` | `None` | Specific sklearn version |

### sklearn_backend / joblib_backend

```python
import skyward as sky

with sky.ComputePool(provider=sky.AWS()) as pool:
    with sky.integrations.sklearn_backend(pool):
        # Any sklearn code with n_jobs=-1 uses Skyward
        pass
```

Context manager to use an existing pool as joblib backend.

---

## Related Topics

- [Getting Started](getting-started.md) — Installation and first examples
- [Core Concepts](concepts.md) — Understanding the programming model
- [Examples](examples.md) — All examples explained
- [Troubleshooting](troubleshooting.md) — Common issues and solutions
