# Joblib

joblib's `Parallel` is how Python parallelizes embarrassingly parallel work. scikit-learn uses it for `GridSearchCV`, `cross_val_score`, and any estimator with `n_jobs`. NLTK uses it. Countless data processing pipelines use `Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)` as the standard idiom for local parallelism. The limitation is that "all available workers" means "all cores on this machine." On a laptop, that is 8 or 16. On an expensive workstation, maybe 64. For large hyperparameter searches or batch processing jobs, this is the bottleneck.

Skyward's `joblib` plugin replaces joblib's execution backend with a distributed one. When the plugin is active, `n_jobs=-1` means "all workers in the cluster" -- not local cores. The joblib API is completely unchanged. `Parallel`, `delayed`, `n_jobs` all work as documented. The difference is that each task is serialized with cloudpickle, sent to a remote worker over SSH, executed there, and the result returned. No code changes beyond the pool configuration.

## What It Does

The plugin contributes two hooks:

**Image transform** -- Appends `joblib` (optionally at a pinned version) to the worker's pip dependencies. This ensures the remote workers have joblib installed for deserialization.

**Client lifecycle (`around_client`)** -- This is where the real work happens. When the pool is entered, the plugin does three things:

1. **Registers `SkywardBackend`** as a custom joblib parallel backend. This is a subclass of `ParallelBackendBase` that replaces joblib's default thread/process backends with one that dispatches tasks to the Skyward cluster.

2. **Strips non-stdlib warning filters** from `warnings.filters`. This is a subtle but important fix: sklearn's `Parallel` pickles the current `warnings.filters` list into every task payload via cloudpickle. If your local environment has warning filters from third-party packages (pytest, cloud SDKs, monitoring libraries), those filters reference module classes that may not exist on the worker. Deserialization would fail with `ModuleNotFoundError`. The plugin removes any filter whose category class comes from outside the standard library, keeping only safe builtins like `DeprecationWarning` and `FutureWarning`. Third-party packages installed on the worker will re-inject their own filters at import time.

3. **Enters the `parallel_backend("skyward")` context manager**, which tells joblib to route all `Parallel` calls to the Skyward backend for the duration of the pool block. When the pool exits, the default backend is restored.

## How SkywardBackend Works

`SkywardBackend` is not a monkey-patch. It is a proper joblib backend -- a subclass of `ParallelBackendBase` that implements joblib's backend protocol. joblib's architecture is designed for pluggable backends, and Skyward registers one.

The key method is `submit()`. When joblib dispatches a batch of work, it calls `submit(func)` on the backend. The default backends run `func` in a local thread or process. `SkywardBackend` does something different:

1. **Serialize** the function with `cloudpickle.dumps()`. joblib passes a pre-batched callable that captures the task function and its arguments.
2. **Wrap** the serialized payload in a `@sky.compute` function that deserializes and executes it on the remote worker.
3. **Dispatch** the wrapped function to the pool using the async `>` operator, which returns a `Future[T]`.
4. **Return** the future to joblib. joblib collects futures and resolves them, calling `retrieve_result_callback` to get the result from each future.

This means each joblib task becomes a Skyward compute task, dispatched to a remote worker through the same SSH tunnel and actor system that all Skyward tasks use. The serialization overhead is minimal -- cloudpickle is fast, and the payloads are compressed with lz4 on the wire.

**Effective parallelism** is `nodes * concurrency`. If you have 4 nodes with `Worker(concurrency=10)`, joblib sees 40 available workers. `n_jobs=-1` uses all of them. `n_jobs=20` would use 20. The backend's `effective_n_jobs()` method reports this number to joblib so it can batch work appropriately.

**Nesting** is handled gracefully. If a joblib `Parallel` call happens inside another `Parallel` call (which scikit-learn does internally in some estimators), the inner call falls back to `SequentialBackend` to avoid recursive remote dispatch.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version` | `str \| None` | `None` | Specific joblib version to install (e.g. `"1.4.0"`). `None` installs the latest version. |

Version pinning is useful when you need reproducible environments or when a specific joblib version is required for compatibility with your local scikit-learn version.

## Usage

### Basic Parallel Execution

Any function works with joblib -- the plugin handles serialization and dispatch:

```python
from time import sleep

from joblib import Parallel, delayed

import skyward as sky


def slow_task(x):
    sleep(5)
    return x * 2


with sky.ComputePool(
    provider=sky.AWS(),
    nodes=10,
    worker=sky.Worker(concurrency=10),
    plugins=[sky.plugins.joblib()],
) as pool:
    results = Parallel(n_jobs=-1)(
        delayed(slow_task)(i) for i in range(2000)
    )
```

With 10 nodes and `concurrency=10`, effective parallelism is 100. The 2000 tasks take 5 seconds each. Ideal time: `2000 / 100 * 5 = 100s`. In practice, overhead from serialization and network round-trips adds a few percent -- expect 97-98% efficiency for tasks of this duration.

### Tuning with Worker Concurrency

The `Worker(concurrency=N)` parameter controls how many tasks each node handles simultaneously. This is the multiplier that makes joblib-on-Skyward practical:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    nodes=2,
    vcpus=64,
    worker=sky.Worker(concurrency=120),
    plugins=[sky.plugins.joblib()],
) as pool:
    results = Parallel(n_jobs=-1)(
        delayed(slow_task)(i) for i in range(20000)
    )
```

High concurrency works well for I/O-bound or sleep-heavy tasks (API calls, network requests, waiting on external services). For CPU-bound tasks, match concurrency to the number of available cores. The default executor is threaded, so Python's GIL applies -- for CPU-bound pure-Python work, consider `Worker(executor="process")` to bypass it.

### With scikit-learn (via the sklearn plugin)

If your workload is scikit-learn-based, prefer the `sklearn` plugin instead -- it builds on the same `SkywardBackend` but also installs scikit-learn:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    worker=sky.Worker(concurrency=4),
    plugins=[sky.plugins.sklearn()],
) as pool:
    grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(X, y)
```

See the [sklearn plugin documentation](sklearn.md) for details.

## Warning Filter Sanitization

This deserves a deeper explanation because it solves a problem that would otherwise cause silent failures.

When scikit-learn's `Parallel` dispatches tasks, it captures the current `warnings.filters` list and includes it in the task payload. This is a list of tuples, each containing a warning category class. Those classes are real Python objects that reference their defining module.

In a typical development environment, `warnings.filters` contains entries from pytest (`PytestUnraisableExceptionWarning`), cloud SDKs, monitoring libraries, and other tools. When cloudpickle serializes these, it records the module path. On the worker, cloudpickle tries to import that module to reconstruct the class. If the module does not exist on the worker -- which it almost certainly does not, because workers have a minimal environment -- deserialization fails with `ModuleNotFoundError`.

The plugin solves this by removing any warning filter whose category class comes from outside the standard library before the `parallel_backend("skyward")` context is entered. Only filters from `stdlib_module_names` and `builtins` are kept. This is safe because third-party packages installed on the worker will register their own filters at import time, and the stripped filters were specific to your local environment anyway.

## Next Steps

- [Joblib Concurrency guide](../guides/joblib-concurrency.md) -- Throughput analysis, real-world benchmarks, and cost model
- [sklearn plugin](sklearn.md) -- The scikit-learn plugin that builds on the same backend
- [What are Plugins?](index.md) -- How the plugin system works
