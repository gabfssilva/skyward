# Distributed Training

This guide covers distributed training across multiple GPUs and nodes with Skyward.

## Overview

Skyward simplifies distributed training by automatically:
- Provisioning multi-node GPU clusters
- Setting up distributed environment variables
- Initializing framework-specific process groups
- Partitioning data across workers

Supported frameworks:
- **PyTorch** - DistributedDataParallel (DDP)
- **Keras 3** - DataParallel with JAX/TensorFlow/PyTorch backends
- **JAX** - Native distributed training
- **TensorFlow** - MultiWorkerMirroredStrategy
- **HuggingFace** - Trainer with automatic distributed detection

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
    sampler = sky.DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=64)

    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important!
        for batch in loader:
            train_step(model, batch)

    return {"node": info.node, "loss": final_loss}


# Launch on 2 nodes with A100 GPUs
with sky.ComputePool(
    provider=sky.AWS(),
    nodes=2,
    accelerator=sky.NVIDIA.A100,
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

@sky.integrations.torch(backend="nccl")
@sky.compute
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

@sky.integrations.keras(backend="jax")
@sky.compute
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
    accelerator="A100",
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

@sky.integrations.keras(backend="tensorflow")
@sky.compute
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

@sky.integrations.jax()
@sky.compute
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
    accelerator="H100",
    pip=["jax[cuda12]"],
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

@sky.integrations.transformers(backend="nccl")
@sky.compute
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
    accelerator="A100",
    pip=["transformers", "datasets", "torch", "accelerate"],
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
import skyward as sky
from torch.utils.data import DataLoader

sampler = sky.DistributedSampler(dataset, shuffle=True)
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

## MIG for Multi-Tenant Training

Use MIG (Multi-Instance GPU) to run multiple training jobs on one GPU:

```python
import skyward as sky

with sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.Accelerator.NVIDIA.A100(mig="3g.40gb"),  # 2 workers per GPU
    image=sky.Image(pip=["torch"]),
) as pool:
    # Each worker gets its own MIG partition
    results = train() @ pool
```

MIG profiles for A100/H100:

| Profile | Workers | Memory Each |
|---------|---------|-------------|
| `3g.40gb` | 2 | 40GB |
| `2g.20gb` | 3 | 20GB |
| `1g.10gb` | 7 | 10GB |

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

- [Accelerators](accelerators.md) — GPU selection and MIG partitioning
- [Examples](examples.md) — Working code examples
- [Troubleshooting](troubleshooting.md) — Common issues and solutions
- [API Reference](api-reference.md) — Complete API documentation