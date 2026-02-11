# Framework Integrations

Skyward integrates with popular ML frameworks to enable distributed training and parallel execution with minimal code changes.

## Deep Learning Frameworks

| Framework | Decorator | Use Case |
|-----------|-----------|----------|
| PyTorch | `@sky.integrations.torch()` | Distributed training with DDP |
| Keras 3 | `@sky.integrations.keras()` | Backend-agnostic distributed training |
| JAX | `@sky.integrations.jax()` | SPMD array distribution |
| TensorFlow | `@sky.integrations.tensorflow()` | MultiWorkerMirroredStrategy |
| HuggingFace | `@sky.integrations.transformers()` | Distributed fine-tuning |

### PyTorch

```python
import skyward as sky

@sky.compute
@sky.integrations.torch(backend="nccl")
def train(data):
    import torch.distributed as dist
    # dist.is_initialized() is True
    return model

with sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.A100(), nodes=4) as pool:
    result = train(data) @ pool  # Broadcast to all nodes
```

### Keras 3

```python
import skyward as sky

@sky.compute
@sky.integrations.keras(backend="jax")
def train():
    import keras
    model = keras.Sequential([...])
    model.fit(x, y)
    return model

with sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.A100(), nodes=4) as pool:
    result = train() @ pool
```

### JAX

```python
import skyward as sky

@sky.compute
@sky.integrations.jax()
def train():
    import jax
    # jax.distributed already initialized
    return results

with sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.H100(), nodes=4) as pool:
    result = train() @ pool
```

### TensorFlow

```python
import skyward as sky

@sky.compute
@sky.integrations.tensorflow()
def train():
    import tensorflow as tf
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = create_model()
        model.fit(dataset)
    return model

with sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    nodes=4,
    image=sky.Image(pip=["tensorflow"]),
) as pool:
    result = train() @ pool
```

### HuggingFace Transformers

```python
import skyward as sky

@sky.compute
@sky.integrations.transformers(backend="nccl")
def fine_tune():
    from transformers import Trainer, TrainingArguments
    trainer = Trainer(...)
    trainer.train()
    return trainer.evaluate()

with sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.A100(), nodes=4) as pool:
    result = fine_tune() @ pool
```

## Decorator Stacking Order

!!! warning "Important"
    Integration decorators must be placed **below** `@compute`:

```python
@sky.compute               # Outer: serializes and sends to remote
@sky.integrations.torch()  # Inner: runs on the remote machine, sets up environment
def train(data):
    ...
```

Decorators apply bottom-up. `@sky.integrations.torch()` wraps your function first, then `@sky.compute` wraps that and sends it to the remote node. When it executes remotely, the integration sets up the distributed environment before your training code runs.

### Output Control

Combine with output control decorators:

```python
from skyward import stdout

@sky.compute
@stdout(only="head")
@sky.integrations.torch()
def train():
    print(f"Training step {step}")  # Only head node prints
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
    print(f"Instance: {info.instance_type} with {info.gpu_count}x {info.gpu_model}")
```

## Joblib & Scikit-learn

Skyward provides a custom joblib backend for distributed parallel execution. This enables existing joblib-based code, including scikit-learn, to run on cloud infrastructure.

### JoblibPool

```python
from skyward import AWS
from skyward.integrations import JoblibPool
from joblib import Parallel, delayed

def process(x):
    return x ** 2

with JoblibPool(provider=AWS(), nodes=4, concurrency=4):
    results = Parallel(n_jobs=-1)(
        delayed(process)(x) for x in range(100)
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `ProviderLike` | required | Cloud provider |
| `nodes` | `int` | `1` | Number of compute nodes |
| `concurrency` | `int` | `1` | Concurrent tasks per node |
| `machine` | `str \| None` | `None` | Direct instance type override |
| `image` | `Image \| None` | `None` | Base image (joblib added automatically) |
| `accelerator` | `str \| None` | `None` | GPU specification |
| `cpu` | `int \| None` | `None` | CPU cores per worker |
| `memory` | `str \| None` | `None` | Memory per worker (e.g., "32GB") |
| `allocation` | `str` | `"spot-if-available"` | Instance allocation strategy |

The effective parallelism is `nodes * concurrency`. Use `n_jobs=-1` to use all available slots.

### ScikitLearnPool

`ScikitLearnPool` is specialized for scikit-learn workloads, automatically adding sklearn to dependencies.

```python
from skyward import AWS
from skyward.integrations import ScikitLearnPool
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'poly'],
}

with ScikitLearnPool(provider=AWS(), nodes=4, concurrency=4, memory="8GB"):
    grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid.fit(X, y)
```

Works with `cross_val_score`, `RandomizedSearchCV`, `RFECV`, and any sklearn estimator that accepts `n_jobs`.

### Manual Backend Activation

For existing code or custom pools, use `sklearn_backend`:

```python
import skyward as sky
from skyward.integrations.joblib import sklearn_backend

pool = sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    image=sky.Image(pip=["scikit-learn", "joblib"]),
)

with pool:
    with sklearn_backend(pool):
        grid = GridSearchCV(model, params, n_jobs=-1)
        grid.fit(X, y)
```

---

## Related Topics

- [Distributed Training](distributed-training.md) — Multi-node training guides
- [API Reference](reference/integrations.md) — Integrations API reference
