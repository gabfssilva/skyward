# cuML

scikit-learn is CPU-only. For large datasets — tens of thousands of samples, hundreds of features — algorithms like RandomForest, KNN, DBSCAN, and PCA become bottlenecks. Cross-validation and hyperparameter search multiply the problem: a 5-fold grid search over 20 candidates means 100 fits, each one CPU-bound. On a 64-core machine, this is tolerable. On a laptop, it is hours.

[NVIDIA cuML](https://docs.rapids.ai/api/cuml/stable/) provides GPU-backed implementations of popular scikit-learn estimators. The API is the same — `RandomForestClassifier`, `KMeans`, `PCA`, `cross_val_score` — but the computation runs on GPU with speedups of 10x to 175x depending on the algorithm and dataset size. cuML's zero-code-change acceleration mode goes further: you write standard scikit-learn code with standard scikit-learn imports, and cuML intercepts the calls at runtime, routing them to GPU transparently.

Skyward's `cuml` plugin makes this practical without a local GPU. It installs the cuML package with the correct CUDA variant, configures the NVIDIA pip index, and activates the zero-code-change acceleration on the remote worker. Your function imports `sklearn`, calls `sklearn` APIs, and cuML handles the rest.

## What It Does

The plugin contributes two hooks:

**Image transform** — Appends `cuml-cu12` (or the CUDA variant you specify) to the worker's pip dependencies and adds the NVIDIA pip index (`https://pypi.nvidia.com`) configured for that package. The RAPIDS packages are hosted on NVIDIA's own index, not PyPI, so the plugin handles the index configuration that you would otherwise need to set up manually in the `Image`.

**Worker lifecycle (`around_app`)** — When the worker starts, the plugin calls `cuml.accel.install()`. This is cuML's zero-code-change acceleration entry point. It monkey-patches the scikit-learn namespace so that `from sklearn.ensemble import RandomForestClassifier` returns cuML's GPU implementation instead of scikit-learn's CPU one. The patching happens once, at worker startup, before any task runs. Every task on that worker benefits from it.

The `around_app` hook is different from `around_client` — it runs on the worker process, not on the client. This is important because `cuml.accel.install()` needs to run in the environment where scikit-learn will be imported and used. The client machine does not need cuML installed, and typically does not have a GPU.

## How cuml.accel.install() Works

cuML's acceleration mode is not a wrapper or an adapter. It replaces scikit-learn's estimator classes at the module level. After `cuml.accel.install()` runs:

- `sklearn.ensemble.RandomForestClassifier` is cuML's `RandomForestClassifier`
- `sklearn.cluster.KMeans` is cuML's `KMeans`
- `sklearn.decomposition.PCA` is cuML's `PCA`
- `sklearn.neighbors.KNeighborsClassifier` is cuML's `KNeighborsClassifier`

And so on for all supported estimators. Unsupported estimators (those cuML does not implement) fall through to the original scikit-learn implementation. This means your code does not need to change — not even the imports. `from sklearn.ensemble import RandomForestClassifier` already resolves to the GPU version.

This also means that scikit-learn utilities that operate on estimators — `cross_val_score`, `GridSearchCV`, `Pipeline` — work with cuML estimators automatically. They call `.fit()` and `.predict()` on whatever estimator they receive, and cuML's estimators implement the same interface.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cuda` | `str` | `"cu12"` | CUDA version suffix. Determines which cuML package variant to install (e.g. `cuml-cu12`). Must match the CUDA version on the worker's GPU. |

The default `"cu12"` works with CUDA 12.x, which covers most modern NVIDIA GPUs and cloud instances. If your instances run CUDA 11.x, use `cuda="cu11"`.

## Usage

### Basic GPU-Accelerated Training

The simplest case: a single-node GPU training with standard scikit-learn code:

```python
import skyward as sky


@sky.compute
def train_on_gpu(n_samples: int) -> dict:
    from time import perf_counter

    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = (X[:n_samples] / 255.0).astype(np.float32)
    y = y[:n_samples].astype(np.int32)

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    start = perf_counter()
    scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    elapsed = perf_counter() - start

    return {"accuracy": scores.mean(), "time": elapsed}


with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="L4",
    nodes=1,
    plugins=[
        sky.plugins.cuml(),
        sky.plugins.sklearn(),
    ],
) as pool:
    result = train_on_gpu(50000) >> pool
    print(f"Accuracy: {result['accuracy']:.2%}, Time: {result['time']:.1f}s")
```

The `sklearn` plugin installs scikit-learn and joblib; the `cuml` plugin installs cuML and activates GPU acceleration. The function uses only scikit-learn imports — `RandomForestClassifier`, `cross_val_score` — and cuML transparently routes them to GPU.

The data is loaded on the remote worker (`fetch_openml` downloads from the internet), avoiding the need to serialize and ship large arrays over the SSH tunnel. This is a general best practice for data-heavy workloads.

### Combining with Distributed Joblib

cuML accelerates individual estimator operations on GPU. The sklearn plugin distributes parallel operations across the cluster. Together, each parallel task (e.g., each fold of cross-validation) runs on GPU:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="L4",
    nodes=4,
    worker=sky.Worker(concurrency=2),
    plugins=[
        sky.plugins.cuml(),
        sky.plugins.sklearn(),
    ],
) as pool:
    result = run_grid_search() >> pool
```

Here, `GridSearchCV(n_jobs=-1)` distributes fits across the 4-node cluster, and each fit runs on GPU thanks to cuML. This is particularly effective for large grid searches where both the individual fits and the number of candidates are expensive.

### Without the sklearn Plugin

If your function does not use scikit-learn's `n_jobs` parallelism, you can use the `cuml` plugin alone:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="L4",
    nodes=1,
    plugins=[sky.plugins.cuml()],
    image=sky.Image(pip=["scikit-learn"]),
) as pool:
    result = train_on_gpu(50000) >> pool
```

You need to add `scikit-learn` to the image manually since the `cuml` plugin does not install it. The `cuml` plugin only provides the GPU acceleration layer; the `sklearn` plugin provides the library itself and the distributed joblib backend.

## Requirements

cuML requires an NVIDIA GPU. The plugin is only useful with GPU-equipped instances — `accelerator="L4"`, `accelerator="T4"`, `accelerator="A100"`, etc. Using it on a CPU-only instance will either fail at import time or silently fall through to CPU scikit-learn.

The CUDA version on the worker must be compatible with the `cuda` parameter. The default `"cu12"` requires CUDA 12.x. Most cloud GPU instances ship with CUDA 12 by default.

## Next Steps

- [cuML GPU Acceleration guide](../guides/cuml-acceleration.md) — CPU vs GPU comparison with benchmarks
- [sklearn plugin](sklearn.md) — Distributed scikit-learn with the joblib backend
- [Scikit Grid Search guide](../guides/scikit-grid-search.md) — Distributed hyperparameter search
- [What are Plugins?](index.md) — How the plugin system works
