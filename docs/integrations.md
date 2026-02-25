# Plugins

Skyward's plugin system connects the compute model — lazy functions, pools, operators — with the distributed training runtimes that ML frameworks provide. Each plugin is a declarative unit that bundles environment setup (pip packages, environment variables), bootstrap operations, and per-task configuration into a single composable object. Plugins are specified on the pool, not on individual functions — they configure the *cluster*, not the *task*.

Plugins are lazy-loaded. `import skyward` doesn't import torch, keras, jax, or any framework SDK. The SDK is installed on the remote worker via the plugin's image transform, and any runtime configuration happens when tasks execute. This means you only pay the import cost for the framework you're actually using, and you don't need any framework installed locally — the plugin handles adding it to the worker's environment.

## How Plugins Work

Plugins are specified as a list on `ComputePool`:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    accelerator="A100",
    plugins=[sky.plugins.torch()],
) as pool:
    results = train() @ pool
```

Each plugin can contribute up to five capabilities:

- **Image transform** — Modifies the `Image` to add pip packages, environment variables, and pip indexes. This runs after `provider.prepare()` and receives the `Cluster` object, so plugins can inspect infrastructure metadata (SSH credentials, provider-specific state, offer details) when configuring the image.
- **Bootstrap operations** — A factory that receives the `Cluster` and returns shell commands to run during instance bootstrap, after the base environment is set up. Because it receives the cluster, plugins can generate bootstrap ops dynamically based on the provisioned infrastructure.
- **Task decorator** — Wraps each `@sky.compute` function at execution time on the worker, configuring the framework's distributed runtime before your code runs.
- **Worker lifecycle (`around_app`)** — A context manager that runs once when the worker starts and tears down when it stops. Used for persistent state like process groups.
- **Client lifecycle (`around_client`)** — A context manager that runs on the client side when the pool is entered. Receives the `ComputePool` and `Cluster` objects. Used for registering custom backends (like joblib).

Plugins compose naturally — you can stack multiple plugins, and their transforms are applied in order:

```python
plugins=[sky.plugins.jax(), sky.plugins.keras(backend="jax")]
```

This first configures the JAX distributed runtime, then sets up Keras with the JAX backend. The JAX plugin adds `jax[cuda12]` to pip and configures distributed initialization; the Keras plugin adds `keras` and sets `KERAS_BACKEND=jax`.

## Deep Learning Frameworks

### PyTorch

`sky.plugins.torch()` configures PyTorch's distributed process group. It adds `torch` to the worker's pip dependencies and sets up `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and calls `torch.distributed.init_process_group()` on each worker. The backend defaults to `nccl` for GPU nodes and `gloo` for CPU. After initialization, you wrap your model with `DistributedDataParallel` and PyTorch handles gradient synchronization — each node computes gradients on its own data, and DDP averages them across all nodes before each optimizer step.

```python
@sky.compute
def train():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = DDP(MyModel().cuda())
    # gradients are averaged across all nodes during backward()
    ...

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    accelerator="A100",
    plugins=[sky.plugins.torch()],
) as pool:
    results = train() @ pool
```

For a complete training example with DDP, `DistributedSampler`, and metric aggregation, see the [PyTorch Distributed guide](guides/pytorch-distributed.md).

### Keras 3

`sky.plugins.keras(backend="jax")` sets the `KERAS_BACKEND` environment variable on the worker before Keras is imported. This must happen before import because Keras reads the backend at import time — setting it after `import keras` has no effect.

Keras 3 is backend-agnostic — the same model code runs on JAX, TensorFlow, or PyTorch. Skyward's automatic distribution (`DataParallel` with device discovery) is currently JAX-only. For the `torch` and `tensorflow` backends, the plugin delegates to those frameworks' native distributed init. For data-parallel training where each node trains independently on its shard (the most common pattern with Skyward), the `keras` plugin alone is sufficient regardless of backend.

When using the JAX backend, combine the Keras and JAX plugins:

```python
@sky.compute
def train():
    import keras
    model = keras.Sequential([...])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(x, y)

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=2,
    accelerator="T4",
    plugins=[sky.plugins.jax(), sky.plugins.keras(backend="jax")],
) as pool:
    results = train() @ pool
```

For a complete MNIST training example, see the [Keras Training guide](guides/keras-training.md).

### JAX

`sky.plugins.jax()` configures JAX's distributed runtime: `JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES`, `JAX_PROCESS_ID`, and `JAX_LOCAL_DEVICE_COUNT`. It then calls `jax.distributed.initialize()`. After initialization, JAX sees all devices across all nodes as a single device mesh, and operations like `pmap` and `pjit` distribute computation automatically.

```python
@sky.compute
def train():
    import jax
    # jax.distributed already initialized
    # all devices across all nodes are visible
    ...

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=2,
    plugins=[sky.plugins.jax()],
) as pool:
    results = train() @ pool
```

### HuggingFace Transformers

`sky.plugins.huggingface(token="...")` adds `transformers`, `datasets`, and `tokenizers` to the worker's pip dependencies, sets the `HF_TOKEN` environment variable, and runs `huggingface-cli login` during bootstrap.

For single-node fine-tuning, the HuggingFace `Trainer` manages device placement on its own — you just need the packages installed. For multi-node distributed training, combine with the `torch` plugin:

```python
@sky.compute
def fine_tune():
    from transformers import Trainer, TrainingArguments
    trainer = Trainer(...)
    trainer.train()
    return trainer.evaluate()

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=2,
    accelerator="A100",
    plugins=[sky.plugins.torch(), sky.plugins.huggingface(token="hf_...")],
) as pool:
    results = fine_tune() @ pool
```

For a complete fine-tuning example, see the [HuggingFace Fine-tuning guide](guides/huggingface-finetuning.md).

## Joblib & Scikit-learn

Not all distributed workloads are deep learning. Hyperparameter search, cross-validation, and embarrassingly parallel batch processing are common in ML, and they typically use joblib's `Parallel` for local parallelism. Skyward's joblib and sklearn plugins replace joblib's backend with a distributed one, so `n_jobs=-1` sends work to cloud instances instead of local cores. The plugins intercept joblib's task batches, wrap them internally, and dispatch them to the cluster.

### Joblib Plugin

`sky.plugins.joblib()` adds `joblib` to the worker's pip dependencies and registers a custom joblib backend on the client side. Inside the pool block, every `Parallel(n_jobs=-1)` call distributes tasks across the cluster.

```python
import skyward as sky
from joblib import Parallel, delayed

def process(x):
    return x ** 2

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    worker=sky.Worker(concurrency=4),
    plugins=[sky.plugins.joblib()],
) as pool:
    results = Parallel(n_jobs=-1)(
        delayed(process)(x) for x in range(100)
    )
```

The `worker` parameter accepts a `Worker` dataclass that controls per-node execution. `Worker(concurrency=4)` means each node runs 4 tasks simultaneously. With 4 nodes and `concurrency=4`, you get 16 effective workers. The total parallelism is always `nodes * concurrency`, and `n_jobs=-1` tells joblib to use all available slots.

For a throughput analysis and real-world benchmarks, see the [Joblib Concurrency guide](guides/joblib-concurrency.md).

### Scikit-learn Plugin

`sky.plugins.sklearn()` extends the joblib plugin by also adding `scikit-learn` to the worker's pip dependencies. It provides a cleaner interface for common patterns like `GridSearchCV`:

```python
import skyward as sky
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    worker=sky.Worker(concurrency=4),
    plugins=[sky.plugins.sklearn()],
):
    grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(X, y)
```

Everything in scikit-learn that accepts `n_jobs` works unchanged: `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, `RFECV`, and any estimator with built-in parallelism. The scikit-learn API is completely unchanged — `best_params_`, `best_score_`, `cv_results_` are all populated as if the search ran locally. The only difference is that the 105 cross-validation fits ran on a cluster instead of a single machine.

For a complete grid search example with multiple estimator families, see the [Scikit Grid Search guide](guides/scikit-grid-search.md).

### cuML Plugin

`sky.plugins.cuml()` adds `cuml-cu12` to pip and configures RAPIDS indexes. It enables GPU-accelerated scikit-learn estimators via NVIDIA's cuML library — same API, but running on GPU instead of CPU:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.accelerators.L4(),
    plugins=[sky.plugins.cuml(), sky.plugins.sklearn()],
) as pool:
    result = train_on_gpu(data) >> pool
```

For a CPU vs GPU comparison, see the [cuML GPU Acceleration guide](guides/cuml-acceleration.md).

## Next Steps

- [Distributed Training](distributed-training.md) — How multi-node training works in Skyward
- [PyTorch Distributed](guides/pytorch-distributed.md) — Step-by-step DDP training
- [Keras Training](guides/keras-training.md) — Keras with JAX backend on multiple GPUs
- [HuggingFace Fine-tuning](guides/huggingface-finetuning.md) — Transformer fine-tuning on cloud GPUs
- [Joblib Concurrency](guides/joblib-concurrency.md) — Throughput analysis and benchmarks
- [Scikit Grid Search](guides/scikit-grid-search.md) — Distributed hyperparameter search
- [cuML GPU Acceleration](guides/cuml-acceleration.md) — GPU-backed scikit-learn with RAPIDS
