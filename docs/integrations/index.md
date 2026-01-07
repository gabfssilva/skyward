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

### Deep Learning

```python
import skyward as sky

@sky.compute
@sky.integrations.torch()
def train(data):
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    # Training code
    return model

@sky.pool(provider=sky.AWS(), accelerator="A100", nodes=4)
def main():
    result = train(data) @ sky  # Broadcast to all nodes
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

### Decorator Stacking

Integration decorators should be placed **below** `@compute`:

```python
@sky.compute          # Outer: handles remote execution
@sky.integrations.torch()  # Inner: sets up environment
def train(data):
    ...
```

### Cluster Information

All integrations can access cluster topology:

```python
@sky.compute
@sky.integrations.torch()
def train(data):
    info = sky.instance_info()
    print(f"Node {info.node}/{info.total_nodes}")
    print(f"Is head: {info.is_head}")
    print(f"Peers: {info.peers}")
```

## Related Topics

- [Distributed Training](../distributed-training.md) - Deep learning distributed training guide
- [Joblib Integration](joblib.md) - Joblib and scikit-learn integration
- [API Reference](../api-reference.md) - Complete API documentation
