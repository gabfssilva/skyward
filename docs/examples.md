# Examples

Complete guide to all Skyward examples with explanations.

## Examples Overview

| # | Example | Concepts | Difficulty |
|---|---------|----------|------------|
| 0 | [Providers](#0-providers) | Instance discovery | Beginner |
| 1 | [Hello World](#1-hello-world) | @compute, >> | Beginner |
| 2 | [Parallel Execution](#2-parallel-execution) | gather(), & | Beginner |
| 3 | [GPU Accelerators](#3-gpu-accelerators) | accelerator, Image | Beginner |
| 4 | [Verda](#4-verda) | Alternative provider | Beginner |
| 5 | [Broadcast](#5-broadcast) | @ operator, shard() | Intermediate |
| 6 | [Data Sharding](#6-data-sharding) | shard(), DistributedSampler | Intermediate |
| 7 | [Cluster Coordination](#7-cluster-coordination) | instance_info(), roles | Intermediate |
| 8 | [S3 Volumes](#8-s3-volumes) | S3Volume | Intermediate |
| 9 | [Event Monitoring](#9-event-monitoring) | Events, callbacks | Intermediate |
| 10 | [PyTorch Distributed](#10-pytorch-distributed) | DDP, multi-node | Advanced |
| 11 | [Keras ViT](#11-keras-vit) | Keras 3, JAX backend | Advanced |
| 12 | [HuggingFace](#12-huggingface) | Transformers, fine-tuning | Advanced |
| 13 | [MultiPool](#13-multipool) | Parallel provisioning, comparison | Intermediate |
| 14 | [Grid Search](#14-distributed-scikit-learn-grid-search) | ScikitLearnPool | Intermediate |
| 15 | [Concurrency](#15-concurrent-task-execution) | concurrency, map_async | Intermediate |
| 16 | [Joblib](#16-joblib-integration) | JoblibPool | Intermediate |
| 17 | [Executor](#17-executor) | concurrent.futures API | Intermediate |
| 18 | [MultiPool Advanced](#18-multipool-advanced) | Multi-stage workflows | Intermediate |
| 19 | [Executor API](#19-executor-api) | submit, map, as_completed | Intermediate |
| 20 | [S3 Volumes](#20-s3-volumes-advanced) | Volume mounting patterns | Intermediate |
| 21 | [Custom Callbacks](#21-custom-callbacks) | Event handling, compose | Advanced |
| 22 | [VastAI Overlay](#22-vastai-overlay) | NCCL, overlay networks | Advanced |

---

## 0. Providers

**File:** `examples/0_providers.py`

Discover available instances from each provider.

```python
import skyward as sky

# List AWS GPU instances
for instance in sky.AWS().available_instances():
    if instance.Accelerator:
        print(f"{instance.name}: {instance.Accelerator}")

# List Verda instances
for instance in sky.Verda().available_instances():
    print(f"{instance.name}: ${instance.price_spot}/hr (spot)")
```

**Key Concepts:**
- `provider.available_instances()` returns `InstanceSpec` objects
- Useful for understanding pricing and availability

---

## 1. Hello World

**File:** `examples/1_hello.py`

The simplest Skyward example.

```python
import skyward as sky

@sky.compute
def remote_sum(x: int, y: int) -> int:
    print("That's one expensive sum.")
    return x + y

@sky.pool(provider=sky.AWS())
def main():
    result = remote_sum(x=1, y=2) >> sky
    print(result)  # 3
```

**Key Concepts:**
- `@compute` - Makes function remotely executable
- `@pool` - Decorator for resource management
- `>> sky` - Execute on the pool from context

---

## 2. Parallel Execution

**File:** `examples/2_parallel_execution.py`

Execute multiple functions concurrently.

```python
import skyward as sky

@sky.compute
def process_chunk(data: list[int]) -> int:
    return sum(data)

@sky.compute
def multiply(x: int, y: int) -> int:
    return x * y

@sky.pool(provider=sky.AWS(), allocation="spot-if-available")
def main():
    # Using gather()
    chunks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    results = sky.gather(*[process_chunk(c) for c in chunks]) >> sky
    print(results)  # (6, 15, 24)

    # Using & operator (type-safe)
    a, b = (multiply(2, 3) & multiply(4, 5)) >> sky
    print(a, b)  # 6, 20
```

**Key Concepts:**
- `gather()` - Group computations for parallel execution
- `&` operator - Chain computations with type safety
- Supports up to 8 chained computations

---

## 3. GPU Accelerators

**File:** `examples/3_gpu_accelerators.py`

Request GPUs and benchmark performance.

```python
import skyward as sky

@sky.compute
def matrix_multiply(size: int) -> dict:
    import torch

    # GPU benchmark
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    _ = torch.matmul(a, b)

    return {"gpu_available": torch.cuda.is_available()}

@sky.pool(
    provider=sky.Verda(),
    image=sky.Image(pip=["torch", "numpy"]),
    accelerator="L40S",
    allocation="spot-if-available",
)
def main():
    result = matrix_multiply(4096) >> sky
```

**Key Concepts:**
- `accelerator="L40S"` - Request specific GPU
- `Image(pip=[...])` - Install dependencies
- `allocation="spot-if-available"` - Use spot instances when available

---

## 4. Verda

**File:** `examples/4_verda_provider.py`

Alternative provider example using Verda.

```python
import skyward as sky

@sky.compute
def gpu_info() -> dict:
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }

@sky.pool(
    provider=sky.Verda(),
    accelerator="L40S",
)
def main():
    result = gpu_info() >> sky
```

**Key Concepts:**
- `Verda()` - Alternative provider
- `accelerator="L40S"` - GPU specification
- Verda specializes in GPU cloud infrastructure

---

## 5. Broadcast

**File:** `examples/5_broadcast.py`

Execute on ALL nodes with automatic data partitioning.

```python
import skyward as sky

@sky.compute
def process_partition(data: list[int]) -> dict:
    info = sky.instance_info()
    local_data = sky.shard(data)  # Auto-partition

    return {
        "node": info.node,
        "partition_size": len(local_data),
        "partition_sum": sum(local_data),
    }

@sky.pool(provider=sky.AWS(), nodes=4, accelerator="T4")
def main():
    data = list(range(1000))
    results = process_partition(data) @ sky  # Broadcast!

    total = sum(r["partition_sum"] for r in results)
    print(f"Total: {total}")  # 499500
```

**Key Concepts:**
- `@ sky` - Broadcast to ALL nodes
- `shard(data)` - Automatic data partitioning
- Returns tuple of results (one per node)

---

## 6. Data Sharding

**File:** `examples/6_data_sharding.py`

Distributed data loading patterns.

```python
import skyward as sky

@sky.compute
def train_with_sampler():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Create dataset
    x = torch.randn(10000, 100)
    y = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(x, y)

    # Distributed sampler
    sampler = sky.DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=64)

    for epoch in range(10):
        sampler.set_epoch(epoch)  # Important!
        for batch_x, batch_y in loader:
            # Training step
            pass

@sky.compute
def train_with_shard(x_full, y_full):
    # Simple sharding
    x_local, y_local = sky.shard(x_full, y_full, shuffle=True, seed=42)
    # Train on local data

@sky.pool(provider=sky.AWS(), nodes=4, image=sky.Image(pip=["torch"]))
def main():
    results = train_with_sampler() @ sky
```

**Key Concepts:**
- `DistributedSampler` - PyTorch integration
- `sampler.set_epoch(epoch)` - Proper shuffling
- `shard(..., shuffle=True)` - Deterministic sharding

---

## 7. Cluster Coordination

**File:** `examples/7_cluster_coordination.py`

Implement distributed patterns like map-reduce.

```python
import skyward as sky

@sky.compute
def worker_task(data: list[int]) -> dict:
    info = sky.instance_info()

    local_data = sky.shard(data)
    result = sum(local_data)

    if info.is_head:
        print("I am the coordinator")

    return {
        "node": info.node,
        "is_head": info.is_head,
        "result": result,
    }

@sky.pool(provider=sky.AWS(), nodes=4)
def main():
    results = worker_task(list(range(100))) @ sky

    # Head node aggregates
    total = sum(r["result"] for r in results)
```

**Key Concepts:**
- `instance_info()` - Access cluster topology
- `info.is_head` - Identify coordinator
- `info.node`, `info.total_nodes` - Node identity

---

## 8. S3 Volumes

**File:** `examples/8_s3_volumes.py`

Mount S3 buckets as local filesystems.

```python
import skyward as sky
from pathlib import Path

@sky.compute
def use_s3_data() -> dict:
    data_path = Path("/data")
    checkpoint_path = Path("/checkpoints")

    # Read from S3
    files = list(data_path.glob("*.json"))

    # Write to S3
    (checkpoint_path / "model.pt").write_bytes(model_bytes)

    return {"files_found": len(files)}

with sky.ComputePool(
    provider=sky.AWS(),
    volume=[
        sky.S3Volume(mount_path="/data", bucket="my-data", read_only=True),
        sky.S3Volume(mount_path="/checkpoints", bucket="my-models"),
    ],
) as pool:
    result = use_s3_data() >> pool
```

**Key Concepts:**
- `S3Volume` - Mount S3 as local path
- `read_only=True` - Read-only access
- Standard file operations work

---

## 9. Event Monitoring

**File:** `examples/9_event_monitoring.py`

Monitor execution with events.

```python
import skyward as sky

def my_callback(event):
    match event:
        case sky.Metrics(cpu_percent=cpu, gpu_utilization=gpu):
            print(f"CPU: {cpu:.1f}%, GPU: {gpu}%")
        case sky.LogLine(line=line, node=node):
            print(f"[Node {node}] {line}")
        case sky.BootstrapProgress(step=step):
            print(f"Installing: {step}")
        case sky.CostFinal(total_cost=cost):
            print(f"Total cost: ${cost:.2f}")

@sky.compute
def long_running_task():
    import time
    for i in range(10):
        print(f"Step {i}")
        time.sleep(1)
    return "done"

with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="T4",
    on_event=my_callback,
) as pool:
    result = long_running_task() >> pool
```

**Key Concepts:**
- `on_event=callback` - Custom event handler
- Pattern matching for event types
- Events: Metrics, LogLine, Cost, Bootstrap, etc.

---

## 10. PyTorch Distributed

**File:** `examples/10_pytorch_distributed.py`

Full DDP training example.

```python
import skyward as sky

@sky.compute
def train_model(epochs: int, batch_size: int) -> dict:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    info = sky.instance_info()

    # Model
    model = SimpleNet().cuda()
    if dist.is_initialized():
        model = DDP(model)

    # Data
    dataset = create_dataset()
    sampler = sky.DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    # Training
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            train_step(model, batch)

        if info.is_head:
            print(f"Epoch {epoch}: loss={loss:.4f}")

    return {"node": info.node, "final_loss": loss}

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=2,
    accelerator=sky.NVIDIA.A100,
    image=sky.Image(pip=["torch"]),
) as pool:
    results = train_model(epochs=10, batch_size=64) @ pool
```

**Key Concepts:**
- Multi-node training with DDP
- `DistributedSampler` for data sharding
- Head node coordination
- Auto environment setup (MASTER_ADDR, etc.)

---

## 11. Keras ViT

**File:** `examples/11_keras_training.py`

Vision Transformer with Keras 3 and JAX backend.

```python
import skyward as sky


@sky.integrations.keras(backend="jax")
@sky.compute
def train_vit(epochs: int) -> dict:
    import keras
    import numpy as np

    info = sky.instance_info()

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0

    # Shard data
    x_local, y_local = sky.shard(x_train, y_train, shuffle=True)

    # Build ViT model
    model = build_vit(embed_dim=64, num_heads=4, num_layers=4)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    # Train
    model.fit(x_local, y_local, epochs=epochs, batch_size=128)

    # Evaluate
    test_acc = model.evaluate(x_test, y_test)[1]

    return {"node": info.node, "test_accuracy": test_acc}


with sky.ComputePool(
    provider=sky.Verda(),
    accelerator=sky.AcceleratorSpec.NVIDIA.A100(mig=["3g.40gb", "3g.40gb"]),
    image=sky.Image(
        pip=["keras>=3.2", "jax[cuda12]"],
        env={"KERAS_BACKEND": "jax"},
    ),
) as pool:
    results = train_vit(epochs=10) @ pool
```

**Key Concepts:**
- `@keras(backend="jax")` - Keras 3 distributed from `skyward.integrations`
- MIG partitioning for multiple workers
- JAX backend for Keras

---

## 12. HuggingFace

**File:** `examples/12_huggingface_finetuning.py`

Fine-tune transformers models.

```python
import skyward as sky

@sky.integrations.transformers(backend="nccl")
@sky.compute
def fine_tune_bert():
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from datasets import load_dataset

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load dataset
    dataset = load_dataset("imdb")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized = dataset.map(tokenize, batched=True)

    # Training
    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )

    trainer.train()
    return trainer.evaluate()

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=2,
    accelerator="A100",
    image=sky.Image(pip=["transformers", "datasets", "torch", "accelerate"]),
) as pool:
    results = fine_tune_bert() @ pool
```

**Key Concepts:**
- `@transformers` - HuggingFace integration from `skyward.integrations`
- Trainer auto-detects distributed setup
- Multi-node fine-tuning

---

## 13. MultiPool

**File:** `examples/13_multi_pool.py`

Provision multiple pools in parallel and compare performance.

```python
import skyward as sky

@sky.compute
def matmul_bench(size: int) -> float:
    import time
    import jax.numpy as jnp

    x = jnp.ones((size, size))
    x @ x  # warmup

    t0 = time.perf_counter()
    for _ in range(100):
        (x @ x).block_until_ready()
    return time.perf_counter() - t0

image = sky.Image(pip=["jax[cuda12]"])

with sky.MultiPool(
    sky.ComputePool(provider=sky.AWS(), image=image, accelerator="T4"),
    sky.ComputePool(provider=sky.AWS(), image=image, accelerator="L4"),
) as (t4_pool, l4_pool):
    t_t4, t_l4 = matmul_bench(2048) >> t4_pool, matmul_bench(2048) >> l4_pool
    print(f"T4: {t_t4:.2f}s | L4: {t_l4:.2f}s | Speedup: {t_t4 / t_l4:.1f}x")
```

**Key Concepts:**
- `MultiPool` - Provisions multiple pools concurrently
- Parallel setup reduces total time from `sum(t_i)` to `max(t_i)`
- Tuple unpacking: `as (pool1, pool2)`
- Use case: Compare different GPUs, split workloads across configurations

---

## 14. Distributed Scikit-Learn Grid Search

**File:** `examples/14_distributed_scikit_grid_search.py`

Run distributed hyperparameter search with scikit-learn.

```python
import skyward as sky
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pipeline with swappable classifier
pipe = Pipeline([("clf", SVC())])

# Search across different estimators AND their hyperparameters
param_grid = [
    {"clf": [RandomForestClassifier()], "clf__n_estimators": [50, 100, 200]},
    {"clf": [GradientBoostingClassifier()], "clf__learning_rate": [0.01, 0.1]},
    {"clf": [SVC()], "clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "poly"]},
]

with sky.integrations.ScikitLearnPool(
    provider=sky.AWS(),
    nodes=3,
    concurrency=4,
    image=sky.Image(pip=["scikit-learn"]),
):
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

print(f"Best: {type(grid.best_params_['clf']).__name__}")
print(f"Test score: {grid.score(X_test, y_test):.2%}")
```

**Key Concepts:**
- `ScikitLearnPool` - Auto-registers Skyward as joblib backend
- `n_jobs=-1` distributes to all available slots
- Unified search across estimator families

---

## 15. Concurrent Task Execution

**File:** `examples/15_concurrency.py`

Run many tasks concurrently using pool slots.

```python
import skyward as sky

@sky.compute
def heavy_stuff(x: int, y: int) -> int:
    from time import sleep
    print(f"Processing {x} + {y}")
    sleep(10)
    return x + y

@sky.pool(
    provider=sky.AWS(),
    cpu=4,
    concurrency=10,  # 10 concurrent tasks per node
    nodes=5,         # 5 nodes = 50 total slots
)
def main():
    # Process 100 tasks across 50 concurrent slots
    results = sky.conc.map_async(
        lambda x: heavy_stuff(x, x) >> sky,
        list(range(100))
    )
    print(list(results))
```

**Key Concepts:**
- `concurrency` parameter controls tasks per node
- `skyward.conc.map_async` for parallel task submission
- Total slots = `nodes * concurrency`

---

## 16. Joblib Integration

**File:** `examples/16_joblib_concurrency.py`

Use Skyward as a joblib backend for distributed parallel execution.

```python
import skyward as sky
from time import sleep
from joblib import Parallel, delayed

def slow_task(x):
    print(f"Task {x} starting")
    sleep(5)
    return x * 2

with sky.integrations.JoblibPool(
    provider=sky.AWS(),
    nodes=5,
    concurrency=5,
    image=sky.Image(pip=["joblib"]),
):
    # Distribute 100 tasks across 25 slots
    results = Parallel(n_jobs=50)(
        delayed(slow_task)(i) for i in range(100)
    )
    print(results)
```

**Key Concepts:**
- `JoblibPool` - Auto-registers Skyward as joblib backend
- Works with existing joblib code (`Parallel`, `delayed`)
- `n_jobs=-1` or any value uses pool slots

---

## 17. Executor

**File:** `examples/17_executor.py`

Drop-in replacement for `concurrent.futures.ThreadPoolExecutor`.

```python
import skyward as sky
from concurrent.futures import as_completed

def slow_task(x: int) -> int:
    from time import sleep
    print(f"Task {x} starting")
    sleep(5)
    return x * 2

# Executor works exactly like ThreadPoolExecutor
with sky.Executor(
    provider=sky.AWS(),
    nodes=5,
    concurrency=5,  # 25 total slots
) as executor:
    # submit() individual tasks
    futures = [executor.submit(slow_task, i) for i in range(25)]

    # as_completed() for results as they finish
    for future in as_completed(futures):
        print(f"Result: {future.result()}")

    # map() for ordered results
    results = list(executor.map(slow_task, range(25, 50)))
```

**Key Concepts:**
- `Executor` - Drop-in for `ThreadPoolExecutor`
- `submit()` - Submit individual tasks
- `as_completed()` - Process results as they arrive
- `map()` - Ordered results

---

## 18. MultiPool Advanced

**File:** `examples/18_multipool.py`

Multi-stage workflows with parallel pool provisioning.

```python
import skyward as sky

@sky.compute
def preprocess(data: list) -> list:
    return [x * 2 for x in data]

@sky.compute
def train(data: list) -> dict:
    import torch
    # GPU training logic
    return {"loss": 0.01, "accuracy": 0.99}

@sky.compute
def evaluate(model_path: str) -> dict:
    return {"test_accuracy": 0.98}

# Provision all pools in parallel
with sky.MultiPool(
    sky.ComputePool(provider=sky.AWS(), cpu=8, name="preprocess"),
    sky.ComputePool(provider=sky.AWS(), accelerator="A100", name="train"),
    sky.ComputePool(provider=sky.AWS(), accelerator="T4", name="eval"),
) as (prep_pool, train_pool, eval_pool):
    # Stage 1: Preprocess on CPU
    processed = preprocess(raw_data) >> prep_pool

    # Stage 2: Train on A100
    model = train(processed) >> train_pool

    # Stage 3: Evaluate on T4
    metrics = evaluate(model) >> eval_pool
```

**Key Concepts:**
- `MultiPool` - Parallel pool provisioning
- Named pools for clarity
- Pipeline stages with different hardware requirements
- Total setup time = `max(pool_setup_times)` instead of sum

---

## 19. Executor API

**File:** `examples/19_executor.py`

Complete `concurrent.futures` API compatibility.

```python
import skyward as sky
import concurrent.futures

def process_item(item: int) -> dict:
    import time
    time.sleep(0.5)
    return {"item": item, "result": item ** 2}

with sky.Executor(
    provider=sky.AWS(),
    nodes=4,
    concurrency=4,  # 16 parallel slots
    cpu=2,
    memory="4GB",
    allocation="spot-if-available",
) as executor:
    print(f"Ready with {executor.total_slots} slots")

    # submit() - individual tasks
    future = executor.submit(process_item, 42)
    result = future.result()

    # map() - preserves order
    results = list(executor.map(process_item, range(5)))

    # as_completed() - results as they finish
    futures = [executor.submit(process_item, x) for x in range(20)]
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print(f"Completed: {result}")

    # Exception handling
    def might_fail(x: int) -> int:
        if x == 7:
            raise ValueError("Unlucky!")
        return x * 2

    futures = [executor.submit(might_fail, x) for x in range(10)]
    for i, future in enumerate(futures):
        try:
            print(f"Item {i}: {future.result()}")
        except ValueError as e:
            print(f"Item {i}: Error - {e}")
```

**Key Concepts:**
- Full `concurrent.futures.Executor` interface
- `executor.total_slots` - Total parallel capacity
- Exception handling via `future.result()`
- Works with `as_completed()` from stdlib

---

## 20. S3 Volumes Advanced

**File:** `examples/20_volumes.py`

S3 volume mounting patterns.

```python
import skyward as sky

@sky.compute
def list_data_files() -> list[str]:
    import os
    files = []
    for root, _, filenames in os.walk("/data"):
        for f in filenames:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            files.append(f"{path} ({size} bytes)")
    return files[:10]

@sky.compute
def save_checkpoint(data: dict) -> str:
    import json, os
    os.makedirs("/checkpoints/run1", exist_ok=True)
    path = "/checkpoints/run1/checkpoint.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return f"Saved to {path}"

# Read-only volume for input data
data_volume = sky.S3Volume(
    mount_path="/data",
    bucket="my-ml-datasets",
    prefix="training/",
    read_only=True,
)

# Read-write volume for outputs
checkpoint_volume = sky.S3Volume(
    mount_path="/checkpoints",
    bucket="my-ml-outputs",
    prefix="checkpoints/",
    read_only=False,
)

# Dict syntax also works
volumes_dict = {
    "/data": "s3://my-ml-datasets/training/",
    "/checkpoints": "s3://my-ml-outputs/checkpoints/",
}

with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="T4",
    volume=[data_volume, checkpoint_volume],
    image=sky.Image(pip=["numpy"]),
) as pool:
    files = list_data_files() >> pool
    result = save_checkpoint({"epoch": 10, "loss": 0.01}) >> pool
```

**Key Concepts:**
- `S3Volume` - Mount S3 as local filesystem
- `read_only=True` - Read-only access (faster)
- `prefix` - Mount subdirectory of bucket
- Dict syntax: `{"/mount": "s3://bucket/prefix/"}`

---

## 21. Custom Callbacks

**File:** `examples/21_custom_callbacks.py`

Event monitoring with custom callbacks.

```python
import skyward as sky
from skyward.callback import compose, use_callback
from skyward.events import (
    ProvisioningStarted, InstanceProvisioned, BootstrapProgress,
    PoolReady, FunctionCall, FunctionResult, CostUpdate, CostFinal,
    Metrics, SkywardEvent,
)

def logging_callback(event: SkywardEvent) -> None:
    match event:
        case ProvisioningStarted():
            print("[LOG] Provisioning started...")
        case InstanceProvisioned(instance=inst):
            print(f"[LOG] Instance {inst.instance_id} at {inst.ip}")
        case PoolReady():
            print("[LOG] Pool ready!")
        case FunctionResult(function_name=name, duration_ms=ms):
            print(f"[LOG] {name} completed in {ms:.1f}ms")

def cost_tracker(event: SkywardEvent) -> None:
    match event:
        case CostUpdate(total_cost=cost, hourly_rate=rate):
            print(f"[COST] Running: ${cost:.3f} (${rate:.2f}/hr)")
        case CostFinal(total_cost=cost, duration_seconds=secs):
            print(f"[COST] Final: ${cost:.3f} ({secs/60:.1f} min)")

def metrics_monitor(event: SkywardEvent) -> None:
    match event:
        case Metrics(gpu_utilization=gpu) if gpu and gpu > 80:
            print(f"[METRICS] GPU high: {gpu}%")

@sky.compute
def train_step(batch_id: int) -> dict:
    import time
    time.sleep(0.1)
    return {"batch": batch_id, "loss": 1.0 / (batch_id + 1)}

# Compose multiple callbacks
combined = compose(logging_callback, cost_tracker, metrics_monitor)

# Option 1: use_callback context manager
with use_callback(combined):
    with sky.ComputePool(
        provider=sky.AWS(),
        accelerator="T4",
    ) as pool:
        for i in range(5):
            result = train_step(i) >> pool

# Option 2: on_event parameter
with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="T4",
    on_event=logging_callback,
) as pool:
    result = train_step(0) >> pool
```

**Key Concepts:**
- `compose()` - Combine multiple callbacks
- `use_callback()` - Context manager for callbacks
- `on_event=` - Pool parameter for callbacks
- Pattern matching on event types

---

## 22. VastAI Overlay

**File:** `examples/22_vastai_overlay.py`

VastAI multi-node with overlay networks for NCCL.

```python
import skyward as sky

@sky.compute
@sky.integrations.torch()
def distributed_train(batch_data: list) -> dict:
    import torch
    import torch.distributed as dist

    # Initialize distributed (env vars set by torch integration)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    info = sky.instance_info()

    # All-reduce example
    tensor = torch.tensor([rank + 1.0]).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    expected_sum = sum(range(1, world_size + 1))
    dist.destroy_process_group()

    return {
        "rank": rank,
        "world_size": world_size,
        "node": info.node,
        "is_head": info.is_head,
        "overlay_ip": info.peers[info.node].ip if info.peers else "N/A",
        "all_reduce_result": tensor.item(),
        "nccl_working": tensor.item() == expected_sum,
    }

# VastAI with overlay networking
provider = sky.VastAI(
    geolocation="US",
    min_reliability=0.95,
    bid_multiplier=1.3,
    use_overlay=True,  # Enable overlay (default for nodes > 1)
)

with sky.ComputePool(
    provider=provider,
    accelerator="RTX 4090",
    nodes=4,  # Multi-node triggers overlay creation
    image=sky.Image(
        pip=["torch"],
        env={"NCCL_DEBUG": "INFO"},
    ),
    allocation="spot-if-available",
) as pool:
    # Broadcast to all nodes
    results = distributed_train([]) @ pool

    # Verify NCCL
    all_working = all(r["nccl_working"] for r in results)
    print(f"NCCL: {'SUCCESS' if all_working else 'FAILED'}")
```

**Key Concepts:**
- `VastAI(use_overlay=True)` - Enable overlay networking
- Overlay creates virtual LAN for NCCL
- `geolocation`, `min_reliability` - Filter marketplace GPUs
- `bid_multiplier` - Bid above minimum for faster provisioning
- `@sky.integrations.torch()` - Auto-setup NCCL env vars

---

## Running Examples

```bash
# Clone repo
git clone https://github.com/example/skyward
cd skyward

# Run example
uv run python examples/1_hello.py
```

## Next Steps

- [Getting Started](getting-started.md) — Installation guide
- [Concepts](concepts.md) — Core concepts explained
- [API Reference](api-reference.md) — Full API documentation
- [Troubleshooting](troubleshooting.md) — Common issues and solutions