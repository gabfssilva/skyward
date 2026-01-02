# API Reference

Complete reference for the Skyward API.

## Core API

### @compute

```python
from skyward import compute

@compute
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
@compute
def add(a: int, b: int) -> int:
    return a + b

pending: PendingCompute[int] = add(1, 2)
```

**Operators:**

| Operator | Method | Result |
|----------|--------|--------|
| `>> pool` | `__rshift__` | Execute, return `R` |
| `@ pool` | `__matmul__` | Broadcast, return `tuple[R, ...]` |
| `& other` | `__and__` | Chain, return `PendingBatch` |

---

### gather()

```python
from skyward import gather

gather(*computations: PendingCompute) -> PendingBatch
```

Group computations for parallel execution.

**Parameters:**
- `*computations` - Variable number of `PendingCompute` objects

**Returns:** `PendingBatch` that executes all in parallel

**Example:**
```python
r1, r2, r3 = gather(fn(1), fn(2), fn(3)) >> pool
```

---

### ComputePool

```python
from skyward import ComputePool, AWS, Image

pool = ComputePool(
    provider=AWS(),
    nodes=2,
    accelerator="A100",
    image=Image(pip=["torch"]),
    spot="always",
    timeout=3600,
)
```

Context manager for cloud resource management.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `Provider` | **required** | Cloud provider (AWS, DigitalOcean, Verda) |
| `image` | `Image` | `Image()` | Environment specification |
| `nodes` | `int` | `1` | Number of instances |
| `machine` | `str` | `None` | Direct instance type override (e.g., "p5.48xlarge") |
| `accelerator` | `str \| Accelerator` | `None` | GPU specification |
| `cpu` | `int` | `None` | CPU cores per worker |
| `memory` | `str` | `None` | Memory per worker (e.g., "32GB") |
| `volume` | `Sequence[Volume]` | `None` | Mounted volumes |
| `spot` | `SpotLike` | `"always"` | Spot instance strategy |
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

### MultiPool

```python
from skyward import ComputePool, MultiPool, AWS

with MultiPool(
    ComputePool(provider=AWS(), accelerator="T4"),
    ComputePool(provider=AWS(), accelerator="A100"),
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
from skyward import Image

image = Image(
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
from skyward import AWS

provider = AWS(
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
from skyward import DigitalOcean

provider = DigitalOcean(
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
from skyward import Verda

provider = Verda(
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
from skyward import Accelerator

# NVIDIA GPUs
Accelerator.NVIDIA.H100()                    # 1x H100 80GB
Accelerator.NVIDIA.H100(count=4)             # 4x H100
Accelerator.NVIDIA.H100(mig="3g.40gb")       # MIG partition
Accelerator.NVIDIA.A100(memory="40GB")       # A100 40GB variant
Accelerator.NVIDIA.T4()                      # 1x T4

# AWS Accelerators
Accelerator.AWS.Trainium(version=2)          # Trainium2
Accelerator.AWS.Inferentia(version=2)        # Inferentia2

# Google TPUs
Accelerator.Google.TPU(version="v5p")
Accelerator.Google.TPUSlice("v5p-8")

# AMD GPUs
Accelerator.AMD.MI("300X")
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
from skyward import NVIDIA

# Use as type hint
def train(gpu: NVIDIA) -> None: ...

# Values
NVIDIA = Literal[
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

### distributed.keras

```python
from skyward import compute, distributed

@compute
@distributed.keras(backend="jax")
def train():
    import keras
    model = keras.Sequential([...])
    model.fit(x, y)
```

**Parameters:**
- `backend` - "jax", "tensorflow", or "torch"

### distributed.torch

```python
@compute
@distributed.torch(backend="nccl")
def train():
    import torch.distributed as dist
    # dist.is_initialized() is True
```

**Parameters:**
- `backend` - "nccl" (GPU) or "gloo" (CPU)

### distributed.jax

```python
@compute
@distributed.jax()
def train():
    import jax
    # jax.distributed already initialized
```

### distributed.tensorflow

```python
@compute
@distributed.tensorflow()
def train():
    import tensorflow as tf
    # TF_CONFIG set automatically
```

### distributed.transformers

```python
@compute
@distributed.transformers(backend="nccl")
def train():
    from transformers import Trainer
    # Trainer auto-detects distributed
```

---

## Data Sharding

### shard()

```python
from skyward import shard

def shard(
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
@compute
def process(x_full, y_full):
    x_local, y_local = shard(x_full, y_full, shuffle=True, seed=42)
    return train(x_local, y_local)
```

### DistributedSampler

```python
from skyward import DistributedSampler

sampler = DistributedSampler(
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
sampler = DistributedSampler(dataset, shuffle=True)
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
from skyward import instance_info, InstanceInfo

def instance_info() -> InstanceInfo
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
from skyward import compose

combined = compose(callback1, callback2, callback3)
```

Merge multiple callbacks into one.

### emit()

```python
from skyward import emit

emit(Metrics(...))
```

Emit an event to all registered callbacks.

### use_callback()

```python
from skyward import use_callback

with use_callback(my_callback):
    # Events in this context go to my_callback
    ...
```

Context manager for callback registration.

---

## Volumes

### S3Volume

```python
from skyward import S3Volume

volume = S3Volume(
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

## Spot Strategies

```python
from skyward import Spot

# String shortcuts
spot = "always"       # Must use spot
spot = "if-available" # Try spot, fallback
spot = "never"        # Always on-demand

# Float shortcut
spot = 0.8            # At least 80% spot

# Class instances
spot = Spot.Always(retries=10, interval=1.0)
spot = Spot.IfAvailable()
spot = Spot.Never
spot = Spot.Percent(0.8)
```

---

## Utility Functions

### is_nvidia()

```python
from skyward import is_nvidia

is_nvidia("H100")  # True
is_nvidia("MI300X")  # False
```

### is_trainium()

```python
from skyward import is_trainium

is_trainium("Trainium2")  # True
is_trainium("H100")  # False
```

### current_accelerator()

```python
from skyward import current_accelerator

acc = current_accelerator()  # Returns accelerator type in compute context
```

### select_instance()

```python
from skyward import select_instance

instance = select_instance(
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
from skyward import AWS, Image
from skyward.integrations import JoblibPool

with JoblibPool(
    provider=AWS(),
    nodes=4,
    concurrency=4,
    image=Image(pip=["joblib"]),
) as pool:
    # Use joblib Parallel with n_jobs=-1
    from joblib import Parallel, delayed
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
from skyward import AWS, Image
from skyward.integrations import ScikitLearnPool

with ScikitLearnPool(
    provider=AWS(),
    nodes=4,
    concurrency=4,
    image=Image(pip=["scikit-learn"]),
):
    from sklearn.model_selection import GridSearchCV
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
from skyward import ComputePool, AWS
from skyward.integrations import sklearn_backend

with ComputePool(provider=AWS()) as pool:
    with sklearn_backend(pool):
        # Any sklearn code with n_jobs=-1 uses Skyward
        pass
```

Context manager to use an existing pool as joblib backend.
