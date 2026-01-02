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
from skyward import AWS, DigitalOcean, Verda

# List AWS GPU instances
for instance in AWS().available_instances():
    if instance.accelerator:
        print(f"{instance.name}: {instance.accelerator}")

# List Verda instances
for instance in Verda().available_instances():
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
from skyward import compute, ComputePool, Verda

@compute
def remote_sum(x: int, y: int) -> int:
    print("That's one expensive sum.")
    return x + y

with ComputePool(provider=Verda()) as pool:
    result = remote_sum(x=1, y=2) >> pool
    print(result)  # 3
```

**Key Concepts:**
- `@compute` - Makes function remotely executable
- `>> pool` - Execute on single worker
- `ComputePool` - Context manager for resources

---

## 2. Parallel Execution

**File:** `examples/2_parallel_execution.py`

Execute multiple functions concurrently.

```python
from skyward import AWS, ComputePool, compute, gather

@compute
def process_chunk(data: list[int]) -> int:
    return sum(data)

@compute
def multiply(x: int, y: int) -> int:
    return x * y

with ComputePool(provider=AWS(), spot="always") as pool:
    # Using gather()
    chunks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    results = gather(*[process_chunk(c) for c in chunks]) >> pool
    print(results)  # (6, 15, 24)

    # Using & operator (type-safe)
    a, b = (multiply(2, 3) & multiply(4, 5)) >> pool
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
from skyward import AWS, ComputePool, compute, instance_info, Image

@compute
def matrix_multiply(size: int) -> dict:
    import torch

    # GPU benchmark
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    _ = torch.matmul(a, b)

    return {"gpu_available": torch.cuda.is_available()}

with ComputePool(
    provider=AWS(),
    image=Image(pip=["torch", "numpy"]),
    accelerator="T4",
    spot="always",
) as pool:
    result = matrix_multiply(4096) >> pool
```

**Key Concepts:**
- `accelerator="T4"` - Request specific GPU
- `Image(pip=[...])` - Install dependencies
- `spot="always"` - Use spot instances

---

## 4. DigitalOcean

**File:** `examples/4_digitalocean_provider.py`

CPU workloads on DigitalOcean.

```python
from skyward import ComputePool, DigitalOcean, compute

@compute
def cpu_intensive() -> dict:
    import multiprocessing
    return {"cpus": multiprocessing.cpu_count()}

with ComputePool(
    provider=DigitalOcean(region="nyc1"),
    cpu=4,
    memory="8GB",
) as pool:
    result = cpu_intensive() >> pool
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
from skyward import ComputePool, compute, shard, instance_info, AWS

@compute
def process_partition(data: list[int]) -> dict:
    pool = instance_info()
    local_data = shard(data)  # Auto-partition

    return {
        "node": pool.node,
        "partition_size": len(local_data),
        "partition_sum": sum(local_data),
    }

with ComputePool(provider=AWS(), nodes=4, accelerator="T4") as pool:
    data = list(range(1000))
    results = process_partition(data) @ pool  # Broadcast!

    total = sum(r["partition_sum"] for r in results)
    print(f"Total: {total}")  # 499500
```

**Key Concepts:**
- `@ pool` - Broadcast to ALL nodes
- `shard(data)` - Automatic data partitioning
- Returns tuple of results (one per node)

---

## 6. Data Sharding

**File:** `examples/6_data_sharding.py`

Distributed data loading patterns.

```python
from skyward import compute, shard, DistributedSampler
from torch.utils.data import DataLoader, TensorDataset

@compute
def train_with_sampler():
    import torch

    # Create dataset
    x = torch.randn(10000, 100)
    y = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(x, y)

    # Distributed sampler
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=64)

    for epoch in range(10):
        sampler.set_epoch(epoch)  # Important!
        for batch_x, batch_y in loader:
            # Training step
            pass

@compute
def train_with_shard(x_full, y_full):
    # Simple sharding
    x_local, y_local = shard(x_full, y_full, shuffle=True, seed=42)
    # Train on local data
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
from skyward import compute, instance_info, ComputePool, AWS

@compute
def worker_task(data: list[int]) -> dict:
    pool = instance_info()

    result = sum(data[pool.node::pool.total_nodes])

    if pool.is_head:
        print("I am the coordinator")

    return {
        "node": pool.node,
        "is_head": pool.is_head,
        "result": result,
    }

with ComputePool(provider=AWS(), nodes=4) as pool:
    results = worker_task(list(range(100))) @ pool

    # Head node aggregates
    total = sum(r["result"] for r in results)
```

**Key Concepts:**
- `instance_info()` - Access cluster topology
- `pool.is_head` - Identify coordinator
- `pool.node`, `pool.total_nodes` - Node identity

---

## 8. S3 Volumes

**File:** `examples/8_s3_volumes.py`

Mount S3 buckets as local filesystems.

```python
from skyward import ComputePool, AWS, S3Volume, compute
from pathlib import Path

@compute
def use_s3_data() -> dict:
    data_path = Path("/data")
    checkpoint_path = Path("/checkpoints")

    # Read from S3
    files = list(data_path.glob("*.json"))

    # Write to S3
    (checkpoint_path / "model.pt").write_bytes(model_bytes)

    return {"files_found": len(files)}

with ComputePool(
    provider=AWS(),
    volume=[
        S3Volume(mount_path="/data", bucket="my-data", read_only=True),
        S3Volume(mount_path="/checkpoints", bucket="my-models"),
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
from skyward import (
    ComputePool, AWS, compute,
    Metrics, LogLine, BootstrapProgress, CostFinal,
)

def my_callback(event):
    match event:
        case Metrics(cpu_percent=cpu, gpu_utilization=gpu):
            print(f"CPU: {cpu:.1f}%, GPU: {gpu}%")
        case LogLine(line=line, node=node):
            print(f"[Node {node}] {line}")
        case BootstrapProgress(step=step):
            print(f"Installing: {step}")
        case CostFinal(total_cost=cost):
            print(f"Total cost: ${cost:.2f}")

@compute
def long_running_task():
    import time
    for i in range(10):
        print(f"Step {i}")
        time.sleep(1)
    return "done"

with ComputePool(
    provider=AWS(),
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
from skyward import (
    AWS, NVIDIA, ComputePool, DistributedSampler,
    compute, instance_info,
)

@compute
def train_model(epochs: int, batch_size: int) -> dict:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    pool = instance_info()

    # Model
    model = SimpleNet().cuda()
    if dist.is_initialized():
        model = DDP(model)

    # Data
    dataset = create_dataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    # Training
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            train_step(model, batch)

        if pool.is_head:
            print(f"Epoch {epoch}: loss={loss:.4f}")

    return {"node": pool.node, "final_loss": loss}

with ComputePool(
    provider=AWS(),
    nodes=2,
    accelerator=NVIDIA.A100,
    pip=["torch"],
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
from skyward import (
    compute, distributed, ComputePool, Verda,
    Accelerator, Image, shard, instance_info,
)

@compute
@distributed.keras(backend="jax")
def train_vit(epochs: int) -> dict:
    import keras
    import numpy as np

    pool = instance_info()

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0

    # Shard data
    x_local, y_local = shard(x_train, y_train, shuffle=True)

    # Build ViT model
    model = build_vit(embed_dim=64, num_heads=4, num_layers=4)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    # Train
    model.fit(x_local, y_local, epochs=epochs, batch_size=128)

    # Evaluate
    test_acc = model.evaluate(x_test, y_test)[1]

    return {"node": pool.node, "test_accuracy": test_acc}

with ComputePool(
    provider=Verda(),
    accelerator=Accelerator.NVIDIA.A100(mig=["3g.40gb", "3g.40gb"]),
    image=Image(
        pip=["keras>=3.2", "jax[cuda12]"],
        env={"KERAS_BACKEND": "jax"},
    ),
) as pool:
    results = train_vit(epochs=10) @ pool
```

**Key Concepts:**
- `@distributed.keras(backend="jax")` - Keras 3 distributed
- MIG partitioning for multiple workers
- JAX backend for Keras

---

## 12. HuggingFace

**File:** `examples/12_huggingface_finetuning.py`

Fine-tune transformers models.

```python
from skyward import compute, distributed, ComputePool, AWS

@compute
@distributed.transformers(backend="nccl")
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

with ComputePool(
    provider=AWS(),
    nodes=2,
    accelerator="A100",
    pip=["transformers", "datasets", "torch", "accelerate"],
) as pool:
    results = fine_tune_bert() @ pool
```

**Key Concepts:**
- `@distributed.transformers` - HuggingFace integration
- Trainer auto-detects distributed setup
- Multi-node fine-tuning

---

## 13. MultiPool

**File:** `examples/13_multi_pool.py`

Provision multiple pools in parallel and compare performance.

```python
from skyward import AWS, ComputePool, Image, MultiPool, compute

@compute
def matmul_bench(size: int) -> float:
    import time
    import jax.numpy as jnp

    x = jnp.ones((size, size))
    x @ x  # warmup

    t0 = time.perf_counter()
    for _ in range(100):
        (x @ x).block_until_ready()
    return time.perf_counter() - t0

image = Image(pip=["jax[cuda12]"])

with MultiPool(
    ComputePool(provider=AWS(), image=image, accelerator="T4"),
    ComputePool(provider=AWS(), image=image, accelerator="L4"),
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

## Running Examples

```bash
# Clone repo
git clone https://github.com/example/skyward
cd skyward

# Run example
uv run python examples/1_hello.py
```

## Next Steps

- [Getting Started](getting-started.md) - Installation guide
- [Concepts](concepts.md) - Core concepts explained
- [API Reference](api-reference.md) - Full API documentation
