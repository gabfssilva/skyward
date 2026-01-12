# Framework Integrations

Skyward integrates with popular ML frameworks to enable distributed training and parallel execution with minimal code changes.

## Available Integrations

### Deep Learning Frameworks

| Framework | Decorator | Use Case |
|-----------|-----------|----------|
| [PyTorch](../distributed-training.md#pytorch) | `@sky.integrations.torch()` | Multi-GPU training with DDP |
| [Keras 3](../distributed-training.md#keras) | `@sky.integrations.keras()` | Backend-agnostic distributed training |
| [JAX](../distributed-training.md#jax) | `@sky.integrations.jax()` | SPMD array distribution |
| [TensorFlow](../distributed-training.md#tensorflow) | `@sky.integrations.tensorflow()` | MultiWorkerMirroredStrategy |
| [HuggingFace](../distributed-training.md#huggingface) | `@sky.integrations.transformers()` | Distributed fine-tuning |

### Parallel Execution

| Framework | Pool Type | Use Case |
|-----------|-----------|----------|
| [Joblib](joblib.md) | `JoblibPool` | Distributed `Parallel` execution |
| [Scikit-learn](joblib.md#scikitlearnpool) | `ScikitLearnPool` | Distributed `GridSearchCV`, cross-validation |

## Quick Start

### PyTorch

```python
import skyward as sky

@sky.integrations.torch(backend="nccl")
@sky.compute
def train(data):
    import torch.distributed as dist
    # dist.is_initialized() is True
    return model

@sky.pool(provider=sky.AWS(), accelerator="A100", nodes=4)
def main():
    result = train(data) @ sky  # Broadcast to all nodes
```

### Keras 3

```python
import skyward as sky

@sky.integrations.keras(backend="jax")
@sky.compute
def train():
    import keras
    model = keras.Sequential([...])
    model.fit(x, y)
    return model

@sky.pool(provider=sky.AWS(), accelerator="A100", nodes=4)
def main():
    result = train() @ sky
```

### JAX

```python
import skyward as sky

@sky.integrations.jax()
@sky.compute
def train():
    import jax
    # jax.distributed already initialized
    return results

@sky.pool(provider=sky.AWS(), accelerator="H100", nodes=4)
def main():
    result = train() @ sky
```

### TensorFlow

```python
import skyward as sky

@sky.integrations.tensorflow()
@sky.compute
def train():
    import tensorflow as tf
    # TF_CONFIG automatically set for MultiWorkerMirroredStrategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = create_model()
        model.fit(dataset)
    return model

@sky.pool(
    provider=sky.AWS(),
    accelerator="A100",
    nodes=4,
    image=sky.Image(pip=["tensorflow"]),
)
def main():
    result = train() @ sky
```

### HuggingFace Transformers

```python
import skyward as sky

@sky.integrations.transformers(backend="nccl")
@sky.compute
def fine_tune():
    from transformers import Trainer, TrainingArguments
    # Trainer auto-detects distributed setup
    trainer = Trainer(...)
    trainer.train()
    return trainer.evaluate()

@sky.pool(provider=sky.AWS(), accelerator="A100", nodes=4)
def main():
    result = fine_tune() @ sky
```

### Joblib/Scikit-learn

```python
from skyward import AWS
from skyward.integrations import JoblibPool
from joblib import Parallel, delayed

with JoblibPool(provider=AWS(), nodes=4, concurrency=4):
    results = Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)
```

## Integration Architecture

Skyward integrations work by:

1. **Environment Setup**: Configure environment variables (MASTER_ADDR, WORLD_SIZE, etc.)
2. **Decorator Composition**: Stack with `@compute` for remote execution
3. **Automatic Discovery**: Detect cluster topology via `instance_info()`
4. **Backend Registration**: Register custom backends (joblib)

### Decorator Stacking Order

**IMPORTANT:** Integration decorators should be placed **above** `@compute`:

```python
@sky.integrations.torch()  # First: sets up environment before function runs
@sky.compute               # Second: handles remote execution
def train(data):
    ...
```

The integration decorator runs **inside** the remote execution context, so it needs to set up the environment before your training code runs.

### Cluster Information

All integrations can access cluster topology:

```python
@sky.integrations.torch()
@sky.compute
def train(data):
    info = sky.instance_info()
    print(f"Node {info.node}/{info.total_nodes}")
    print(f"Is head: {info.is_head}")
    print(f"Instance: {info.instance_type} with {info.gpu_count}x {info.gpu_model}")
```

### Output Control with Integrations

Combine with output control decorators:

```python
from skyward import stdout

@stdout(only="head")
@sky.integrations.torch()
@sky.compute
def train():
    # Only head node prints progress
    print(f"Training step {step}")
```

## Related Topics

- [Distributed Training](../distributed-training.md) - Deep learning distributed training guide
- [Joblib Integration](joblib.md) - Joblib and scikit-learn integration
