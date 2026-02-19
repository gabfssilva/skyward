# Framework Integrations

Skyward's integration system connects the compute model — lazy functions, pools, operators — with the distributed training runtimes that ML frameworks provide. Each integration is a decorator that runs on the remote worker, between `@sky.compute` (which handles serialization and dispatch) and your function body (which does the actual work). The decorator configures the distributed environment that the framework expects — environment variables, process groups, device meshes — so that by the time your code runs, the framework is ready for distributed execution.

Integrations are lazy-loaded. `import skyward` doesn't import torch, keras, jax, or any framework SDK. The import happens on the remote worker, when the integration decorator executes. This means you only pay the import cost for the framework you're actually using, and you don't need any framework installed locally — only on the workers, via the Image's `pip` field.

## How Decorators Work

Integration decorators follow a simple pattern: they wrap your function, read the cluster topology from `instance_info()`, configure the framework, and then call your original function. The key detail is **decorator order** — `@sky.compute` must be outermost, and the integration decorator goes below it:

```python
@sky.compute                   # outer: serializes and sends to remote
@sky.integrations.torch        # inner: runs on the remote machine
def train(data):
    ...
```

Decorators apply bottom-up. `@sky.integrations.torch` wraps your function first, producing a new function that initializes the distributed environment before calling your code. Then `@sky.compute` wraps that, creating a `PendingCompute` that, when dispatched, sends the entire bundle — your function plus the integration wrapper — to the remote worker. When it executes on the remote side, the integration runs first, then your code.

You can stack multiple decorators. Output control decorators (`@sky.stdout`, `@sky.stderr`, `@sky.silent`) combine naturally with integration decorators:

```python
@sky.compute
@sky.stdout(only="head")       # suppress stdout on non-head nodes
@sky.integrations.torch        # initialize distributed PyTorch
def train():
    print(f"Training...")      # only head node prints
```

## Deep Learning Frameworks

### PyTorch

`@sky.integrations.torch` initializes PyTorch's distributed process group. It sets `MASTER_ADDR` to the head node's private IP, `MASTER_PORT` to the coordination port, `WORLD_SIZE` to the total number of nodes, `RANK` to this node's index, and calls `torch.distributed.init_process_group()`. The backend defaults to `nccl` for GPU nodes and `gloo` for CPU. After initialization, you wrap your model with `DistributedDataParallel` and PyTorch handles gradient synchronization — each node computes gradients on its own data, and DDP averages them across all nodes before each optimizer step.

```python
@sky.compute
@sky.integrations.torch
def train():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = DDP(MyModel().cuda())
    # gradients are averaged across all nodes during backward()
    ...
```

For a complete training example with DDP, `DistributedSampler`, and metric aggregation, see the [PyTorch Distributed guide](guides/pytorch-distributed.md).

### Keras 3

`@sky.integrations.keras(backend="jax")` sets the `KERAS_BACKEND` environment variable before Keras is imported. This must happen before import because Keras reads the backend at import time — setting it after `import keras` has no effect. The optional `seed` parameter configures random seeds for reproducibility across all backends.

Keras 3 is backend-agnostic — the same model code runs on JAX, TensorFlow, or PyTorch. Skyward's distribution integration (automatic device discovery and `DataParallel`) is currently JAX-only. For the `torch` and `tensorflow` backends, the decorator delegates to those frameworks' native distributed init. For data-parallel training where each node trains independently on its shard (the most common pattern with Skyward), no extra distribution configuration is needed regardless of backend.

```python
@sky.compute
@sky.integrations.keras(backend="jax", seed=42)
def train():
    import keras
    model = keras.Sequential([...])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(x, y)
```

For a complete MNIST training example, see the [Keras Training guide](guides/keras-training.md).

### JAX

`@sky.integrations.jax()` configures JAX's distributed runtime: `JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES`, `JAX_PROCESS_ID`, and `JAX_LOCAL_DEVICE_COUNT`. It then calls `jax.distributed.initialize()`. After initialization, JAX sees all devices across all nodes as a single device mesh, and operations like `pmap` and `pjit` distribute computation automatically.

```python
@sky.compute
@sky.integrations.jax()
def train():
    import jax
    # jax.distributed already initialized
    # all devices across all nodes are visible
    ...
```

### HuggingFace Transformers

`@sky.integrations.transformers(backend="nccl")` sets up the PyTorch distributed environment for the HuggingFace `Trainer`. The `Trainer` auto-detects the distributed setup and handles gradient synchronization, mixed-precision, and distributed evaluation internally.

For single-node fine-tuning, you don't need an integration decorator — the `Trainer` manages device placement on its own. The integration is only needed when training across multiple nodes, where each node needs to know its rank and the master address.

```python
@sky.compute
@sky.integrations.transformers(backend="nccl")
def fine_tune():
    from transformers import Trainer, TrainingArguments
    trainer = Trainer(...)
    trainer.train()
    return trainer.evaluate()
```

For a complete fine-tuning example, see the [HuggingFace Fine-tuning guide](guides/huggingface-finetuning.md).

## Joblib & Scikit-learn

Not all distributed workloads are deep learning. Hyperparameter search, cross-validation, and embarrassingly parallel batch processing are common in ML, and they typically use joblib's `Parallel` for local parallelism. Skyward replaces joblib's backend with a distributed one, so `n_jobs=-1` sends work to cloud instances instead of local cores. No `@sky.compute` decorator is needed — the pool intercepts joblib's task batches, wraps them internally, and dispatches them to the cluster.

### JoblibPool

`JoblibPool` is a context manager that provisions cloud instances and registers a custom joblib backend. Inside the block, every `Parallel(n_jobs=-1)` call distributes tasks across the cluster. When you exit, instances are terminated and the default backend is restored.

```python
import skyward as sky
from joblib import Parallel, delayed

def process(x):
    return x ** 2

with sky.integrations.JoblibPool(provider=sky.AWS(), nodes=4, concurrency=4):
    results = Parallel(n_jobs=-1)(
        delayed(process)(x) for x in range(100)
    )
```

The `concurrency` parameter controls how many tasks each node runs simultaneously. With 4 nodes and `concurrency=4`, you get 16 effective workers. The total parallelism is always `nodes * concurrency`, and `n_jobs=-1` tells joblib to use all available slots.

For a throughput analysis and real-world benchmarks, see the [Joblib Concurrency guide](guides/joblib-concurrency.md).

### ScikitLearnPool

`ScikitLearnPool` is a specialized `JoblibPool` for scikit-learn workloads. It automatically adds sklearn to the worker's dependencies and provides a cleaner interface for common patterns like `GridSearchCV`:

```python
import skyward as sky
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

with sky.integrations.ScikitLearnPool(provider=sky.AWS(), nodes=4, concurrency=4):
    grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(X, y)
```

Everything in scikit-learn that accepts `n_jobs` works unchanged: `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, `RFECV`, and any estimator with built-in parallelism. The scikit-learn API is completely unchanged — `best_params_`, `best_score_`, `cv_results_` are all populated as if the search ran locally. The only difference is that the 105 cross-validation fits ran on a cluster instead of a single machine.

For a complete grid search example with multiple estimator families, see the [Scikit Grid Search guide](guides/scikit-grid-search.md).

### Manual Backend Activation

If you already have a `ComputePool` and want to use it with joblib, use `sklearn_backend` as a context manager:

```python
import skyward as sky

with sky.ComputePool(provider=sky.AWS(), nodes=4) as pool:
    with sky.integrations.sklearn_backend(pool):
        grid = GridSearchCV(model, params, n_jobs=-1)
        grid.fit(X, y)
```

This gives you full control over the pool configuration — accelerators, image, allocation strategy — while still routing joblib tasks through the cluster.

## Next Steps

- [Distributed Training](distributed-training.md) — How multi-node training works in Skyward
- [PyTorch Distributed](guides/pytorch-distributed.md) — Step-by-step DDP training
- [Keras Training](guides/keras-training.md) — Keras with JAX backend on multiple GPUs
- [HuggingFace Fine-tuning](guides/huggingface-finetuning.md) — Transformer fine-tuning on cloud GPUs
- [Joblib Concurrency](guides/joblib-concurrency.md) — Throughput analysis and benchmarks
- [Scikit Grid Search](guides/scikit-grid-search.md) — Distributed hyperparameter search
