# Distributed Training

This guide covers distributed training across multiple accelerators and nodes with Skyward.

## Overview

Skyward simplifies distributed training by automatically:

- Provisioning multi-node accelerator clusters
- Setting up distributed environment variables
- Initializing framework-specific process groups
- Partitioning data across workers

Supported frameworks:

- **PyTorch** - DistributedDataParallel (DDP)
- **Keras 3** - DataParallel with JAX/TensorFlow/PyTorch backends
- **JAX** - Native distributed training
- **TensorFlow** - MultiWorkerMirroredStrategy
- **HuggingFace** - Trainer with automatic distributed detection

These come with out-of-the-box integration decorators that handle setup automatically. Any other distributed framework works too — use `sky.instance_info()` inside your `@sky.compute` function to get cluster topology (node index, total nodes, head address) and configure it yourself.

## PyTorch Distributed Training

### Basic Setup

```python
import skyward as sky

@sky.compute
def train(epochs: int) -> dict:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader

    info = sky.instance_info()

    # Model setup
    model = MyModel().cuda()
    if dist.is_initialized():
        model = DDP(model)

    # Data loading with distributed sampler
    loader = DataLoader(dataset, batch_size=64)

    # Training loop
    for epoch in range(epochs):
        for batch in loader:
            train_step(model, batch)

    return {"node": info.node, "loss": final_loss}


# Launch on 2 nodes with A100 GPUs (works with any provider: AWS, RunPod, VastAI, Verda)
with sky.ComputePool(
    provider=sky.AWS(),  # or sky.RunPod(), sky.VastAI(), sky.Verda()
    nodes=2,
    accelerator=sky.accelerators.A100(),
    image=sky.Image(pip=["torch"]),
) as pool:
    results = train(epochs=10) @ pool
```

### Environment Variables

Skyward automatically sets:

| Variable | Description |
|----------|-------------|
| `MASTER_ADDR` | Head node IP address |
| `MASTER_PORT` | Communication port |
| `WORLD_SIZE` | Total number of processes |
| `RANK` | Global rank of current process |
| `LOCAL_RANK` | Rank within current node |
| `LOCAL_WORLD_SIZE` | Processes per node |
| `NODE_RANK` | Node index |

### Explicit Decorator

For explicit control, use the `@sky.integrations.torch` decorator:

```python
import skyward as sky

@sky.compute
@sky.integrations.torch(backend="nccl")
def train():
    import torch.distributed as dist
    # dist.is_initialized() is True
    ...
```

## Keras 3 Training

Keras 3 is backend-agnostic and works with JAX, TensorFlow, or PyTorch.

### JAX Backend (Recommended)

```python
import skyward as sky

@sky.compute
@sky.integrations.keras(backend="jax")
def train_model(epochs: int) -> dict:
    import keras
    import numpy as np

    pool = sky.instance_info()

    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0

    # Shard data across nodes
    x_local, y_local = sky.shard(x_train, y_train, shuffle=True, seed=42)

    # Build model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train
    history = model.fit(x_local, y_local, epochs=epochs, batch_size=64)

    return {
        "node": pool.node,
        "accuracy": history.history["accuracy"][-1],
    }


with sky.ComputePool(
    provider=sky.AWS(),
    nodes=2,
    accelerator=sky.accelerators.A100(),
    image=sky.Image(
        pip=["keras>=3.2", "jax[cuda12]"],
        env={"KERAS_BACKEND": "jax"},
    ),
) as pool:
    results = train_model(epochs=10) @ pool
```

### TensorFlow Backend

```python
import skyward as sky

@sky.compute
@sky.integrations.keras(backend="tensorflow")
def train():
    import keras
    # Uses TF distributed strategy
    ...

with sky.ComputePool(
    provider=sky.AWS(),
    image=sky.Image(
        pip=["keras>=3.2", "tensorflow"],
        env={"KERAS_BACKEND": "tensorflow"},
    ),
) as pool:
    results = train() @ pool
```

## JAX Distributed Training

```python
import skyward as sky

@sky.compute
@sky.integrations.jax()
def train():
    import jax
    import jax.numpy as jnp

    # JAX distributed already initialized
    # Use pmap or jax.experimental.pjit for parallelism

    @jax.pmap
    def train_step(params, batch):
        ...

    return results


with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    accelerator=sky.accelerators.H100(),
    image=sky.Image(pip=["jax[cuda12]"]),
) as pool:
    results = train() @ pool
```

### JAX Environment Variables

| Variable | Description |
|----------|-------------|
| `JAX_COORDINATOR_ADDRESS` | Coordinator address (head:port) |
| `JAX_NUM_PROCESSES` | Total number of JAX processes |
| `JAX_PROCESS_ID` | Current process ID |
| `JAX_LOCAL_DEVICE_COUNT` | Devices on this node |

## HuggingFace Transformers

```python
import skyward as sky

@sky.compute
@sky.integrations.transformers(backend="nccl")
def fine_tune():
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from datasets import load_dataset

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load and tokenize dataset
    dataset = load_dataset("imdb")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized = dataset.map(tokenize, batched=True)

    # Training arguments
    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        fp16=True,
    )

    # Trainer auto-detects distributed setup
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
    accelerator=sky.accelerators.A100(),
    image=sky.Image(pip=["transformers", "datasets", "torch", "accelerate"]),
) as pool:
    results = fine_tune() @ pool
```

## Data Sharding

### Using shard()

The `shard()` function automatically partitions data based on the current worker:

```python
import skyward as sky

@sky.compute
def process_data(full_dataset):
    info = sky.instance_info()

    # Automatically get this worker's portion
    local_data = sky.shard(full_dataset)

    print(f"Node {info.node}: processing {len(local_data)} items")
    return sum(local_data)
```

### Multiple Arrays

Shard multiple arrays consistently:

```python
x_local, y_local = sky.shard(x_full, y_full, shuffle=True, seed=42)
```

### With Shuffle

```python
# Reproducible shuffled sharding
x_local = sky.shard(x_full, shuffle=True, seed=42)
```

### DistributedSampler for PyTorch

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler, batch_size=32)

for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Important for proper shuffling!
    for batch in loader:
        train_step(batch)
```

## Head Node Coordination

Use `sky.instance_info()` to identify roles:

```python
import skyward as sky

@sky.compute
def train():
    info = sky.instance_info()

    # Training code...

    # Only head node saves checkpoint
    if info.is_head:
        save_checkpoint(model)
        print(f"Saved checkpoint from head node")

    # Only head node prints progress
    if info.is_head:
        print(f"Epoch {epoch}: loss={loss:.4f}")

    return {"node": info.node, "is_head": info.is_head}
```

### InstanceInfo Properties

| Property | Type | Description |
|----------|------|-------------|
| `node` | `int` | Node index (0 to total_nodes - 1) |
| `worker` | `int` | Worker index within this node (default 0) |
| `total_nodes` | `int` | Total number of nodes in the pool |
| `workers_per_node` | `int` | Workers per node (e.g., 2 for MIG) |
| `accelerators` | `int` | Number of accelerators on this node |
| `total_accelerators` | `int` | Total accelerators in the pool |
| `head_addr` | `str` | IP address of the head node |
| `head_port` | `int` | Port for head node coordination |
| `job_id` | `str` | Unique identifier for this pool execution |
| `peers` | `list[PeerInfo]` | Information about all peer nodes |
| `accelerator` | `AcceleratorInfo \| None` | Accelerator type, count, and memory |
| `network` | `NetworkInfo` | Network interface and bandwidth |
| `is_head` | `bool` | True if this is the head worker (property) |
| `global_worker_index` | `int` | Global index across all workers (property) |
| `total_workers` | `int` | Total workers across all nodes (property) |

### Output Control

Control stdout/stderr in distributed training:

```python
import skyward as sky
from skyward import stdout, silent

@sky.compute
@stdout(only="head")
def train():
    # Only head node prints progress
    print(f"Epoch {epoch}: loss={loss:.4f}")

@sky.compute
@silent
def background_init():
    # No output from any node
    pass
```

## Best Practices

### 1. Always Set Epoch in Sampler

```python
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Ensures proper shuffling
```

### 2. Use Head Node for I/O

```python
info = sky.instance_info()
if info.is_head:
    model.save("checkpoint.pt")
    wandb.log(metrics)
```

### 3. Synchronize Before Aggregation

```python
import torch.distributed as dist

# Ensure all workers are done
dist.barrier()

# Aggregate results
info = sky.instance_info()
if info.is_head:
    aggregate_metrics()
```

### 4. Use Gradient Accumulation for Large Batches

```python
accumulation_steps = 4

for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5. Enable Mixed Precision

```python
# PyTorch
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)

# Keras
model.compile(..., jit_compile=True)

# HuggingFace
TrainingArguments(fp16=True)
```

## Troubleshooting

### NCCL Timeout

If you see "NCCL timeout" errors:

```python
import os
os.environ["NCCL_SOCKET_TIMEOUT"] = "600"
os.environ["NCCL_DEBUG"] = "INFO"
```

### Memory Issues

- Reduce batch size
- Use gradient checkpointing
- Enable mixed precision (fp16)

### Connectivity Issues

For multi-node training, ensure security groups allow:

- TCP port 29500 (MASTER_PORT)
- All high ports for NCCL (1024-65535)

### Process Group Not Initialized

Always check initialization:

```python
import torch.distributed as dist

if not dist.is_initialized():
    # Single-node fallback
    model = MyModel()
else:
    model = DDP(MyModel())
```

---

## Related Topics

- [Accelerators](accelerators.md) — Accelerator selection guide
- [Integrations](integrations.md) — Framework integration details
- [API Reference](reference/pool.md) — Complete API documentation