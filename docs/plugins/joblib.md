# Joblib

joblib's `Parallel` is how Python parallelizes embarrassingly parallel work. scikit-learn uses it for `GridSearchCV`, `cross_val_score`, and any estimator with `n_jobs`. NLTK uses it. Countless data processing pipelines use `Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)` as the standard idiom for local parallelism. The limitation is that "all available workers" means "all cores on this machine." On a laptop, that is 8 or 16. On an expensive workstation, maybe 64. For large hyperparameter searches or batch processing jobs, this is the bottleneck.

`sky.plugins.Joblib()` replaces joblib's execution backend with a distributed one. When the plugin is active, `n_jobs=-1` means "every slot in the compute" — not local cores. The joblib API is unchanged: `Parallel`, `delayed` and `n_jobs` work as documented. The difference is that each batch is dispatched to a node, executed there, and the result returned.

## What it does

**`image`** — Appends `joblib` (optionally at a pinned version) to the image's pip list, so the nodes can unpickle the batch.

**`client`** — The hook that does the work, and the only one that does not travel: it runs in the process that opened the `with` block. It registers a `ParallelBackendBase` subclass under the name `skyward`, then enters `parallel_backend("skyward")` for the lifetime of the compute. On exit the previous backend is restored.

Each batch joblib would have handed to a worker process is submitted to the compute as a task instead, and joblib's callback is attached to the resulting future. Nested `Parallel` calls fall back to joblib's sequential backend, so a batch does not try to fan out again from the node.

**Effective parallelism** is the compute's node count times the executor's concurrency. Four nodes at `concurrency=10` gives joblib 40 slots; `n_jobs=-1` uses all of them, `n_jobs=20` uses 20. When the compute is elastic, the node count used is its upper bound.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version` | `str \| None` | `None` | Specific joblib version to install (e.g. `"1.4.0"`). `None` installs the latest version. |

Version pinning is useful when you need reproducible environments or when a specific joblib version is required for compatibility with your local scikit-learn version.

## Usage

### Basic parallel execution

Any function works with joblib — the plugin handles serialization and dispatch:

```python
from time import sleep

from joblib import Parallel, delayed

import skyward as sky


def slow_task(x):
    sleep(5)
    return x * 2


with sky.Compute(
    provider=sky.AWS(),
    nodes=10,
    executor=sky.Executor(concurrency=10),
    plugins=[sky.plugins.Joblib()],
) as compute:
    results = Parallel(n_jobs=-1)(
        delayed(slow_task)(i) for i in range(2000)
    )
```

With 10 nodes and `concurrency=10`, effective parallelism is 100. The 2000 tasks take 5 seconds each. Ideal time: `2000 / 100 * 5 = 100s`. In practice, overhead from serialization and network round-trips adds a few percent — expect 97-98% efficiency for tasks of this duration.

### Tuning concurrency

`Executor(concurrency=N)` controls how many tasks each node runs at once. This is the multiplier that makes joblib-on-Skyward practical:

```python
with sky.Compute(
    provider=sky.AWS(),
    nodes=2,
    cpus=64,
    executor=sky.Executor(concurrency=120),
    plugins=[sky.plugins.Joblib()],
) as compute:
    results = Parallel(n_jobs=-1)(
        delayed(slow_task)(i) for i in range(20000)
    )
```

High concurrency works well for I/O-bound or sleep-heavy tasks (API calls, network requests, waiting on external services). For CPU-bound tasks, match concurrency to the number of available cores. The default executor is threaded, so Python's GIL applies — for CPU-bound pure-Python work, consider `Executor(type="process")` to bypass it.

### With scikit-learn (via the sklearn plugin)

If your workload is scikit-learn-based, prefer the `Sklearn` plugin instead — it enters this plugin's client hook, and additionally installs scikit-learn and scrubs the client's warning filters:

```python
with sky.Compute(
    provider=sky.AWS(),
    nodes=4,
    executor=sky.Executor(concurrency=4),
    plugins=[sky.plugins.Sklearn()],
) as compute:
    grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(X, y)
```

See the [sklearn plugin documentation](sklearn.md) for details.

## Warning filters

Stripping the client's non-stdlib warning filters — the fix for scikit-learn's `Parallel` shipping them inside every batch and breaking the unpickle on a node that lacks the module — belongs to the [`Sklearn`](sklearn.md) plugin, not this one.

## Next steps

- [Joblib Concurrency guide](../guides/joblib-concurrency.md) — Throughput analysis, real-world benchmarks, and cost model
- [Scikit-learn plugin](sklearn.md) — the same backend, plus scikit-learn
- [What are plugins?](index.md) — the hook model
