# Examples

Complete guide to all Skyward examples with explanations.

## Examples Overview

| # | Example | Concepts | Difficulty |
|---|---------|----------|------------|
| 0 | [Providers](#0-providers) | Instance discovery | Beginner |
| 1 | [Hello World](#1-hello-world) | @compute, >> | Beginner |
| 2 | [Parallel Execution](#2-parallel-execution) | gather(), & | Beginner |
| 3 | [GPU Accelerators](#3-gpu-accelerators) | accelerator, Image | Beginner |
| 4 | [DigitalOcean](#4-digitalocean) | Alternative provider | Beginner |
| 5 | [Broadcast](#5-broadcast) | @ operator, shard() | Intermediate |
| 6 | [Data Sharding](#6-data-sharding) | shard(), DistributedSampler | Intermediate |
| 7 | [Cluster Coordination](#7-cluster-coordination) | instance_info(), roles | Intermediate |
| 8 | [S3 Volumes](#8-s3-volumes) | S3Volume | Intermediate |
| 9 | [Event Monitoring](#9-event-monitoring) | Events, callbacks | Intermediate |
| 10 | [PyTorch Distributed](#10-pytorch-distributed) | DDP, multi-node | Advanced |
| 11 | [Keras ViT](#11-keras-vit) | Keras 3, JAX backend | Advanced |
| 12 | [HuggingFace](#12-huggingface) | Transformers, fine-tuning | Advanced |
| 13 | [MultiPool](#13-multipool) | Parallel provisioning, comparison | Intermediate |

---

## 0. Providers

**File:** `examples/0_providers.py`

Discover available instances from each provider.

```python
import skyward as sky

# List AWS GPU instances
for instance in sky.AWS().available_instances():
    if instance.accelerator:
        print(f"{instance.name}: {instance.accelerator}")

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

## 4. DigitalOcean

**File:** `examples/4_digitalocean_provider.py`

CPU workloads on DigitalOcean.

```python
import skyward as sky

@sky.compute
def cpu_intensive() -> dict:
    import multiprocessing
    return {"cpus": multiprocessing.cpu_count()}

@sky.pool(
    provider=sky.DigitalOcean(region="nyc1"),
    cpu=4,
    memory="8GB",
)
def main():
    result = cpu_intensive() >> sky
```

**Key Concepts:**
- `DigitalOcean(region="nyc1")` - Alternative provider
- `cpu=4, memory="8GB"` - Resource specification
- No GPU support on DigitalOcean

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
    accelerator=sky.Accelerator.NVIDIA.A100(mig=["3g.40gb", "3g.40gb"]),
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