# cuML

scikit-learn is CPU-only. For large datasets — tens of thousands of samples, hundreds of features — RandomForest, KNN, DBSCAN and PCA become bottlenecks, and cross-validation multiplies the problem: a 5-fold grid search over 20 candidates is 100 fits, each one CPU-bound.

[NVIDIA cuML](https://docs.rapids.ai/api/cuml/stable/) provides GPU implementations of the popular scikit-learn estimators. Its zero-code-change acceleration goes further than a parallel API: you write standard scikit-learn code with standard scikit-learn imports, and cuML rewrites the estimators at runtime so the calls land on the GPU.

`sky.plugins.Cuml()` installs the package from NVIDIA's index and turns that acceleration on, on the node.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cuda` | `str` | `"cu12"` | The CUDA suffix of the RAPIDS wheel. Names the package (`cuml-cu12`) as much as the build |

The default covers CUDA 12.x, which is what most cloud GPU images ship.

## How it works

### `image`

Appends `cuml-{cuda}` to the image's pip list and adds NVIDIA's index (`https://pypi.nvidia.com`) scoped to that package. RAPIDS is not on PyPI, so this is the index configuration you would otherwise write by hand in the `Image`.

The plugin does **not** install scikit-learn. Either add it to the image, or use [`sky.plugins.Sklearn()`](sklearn.md).

### `run`

On the first task in each process, the plugin calls `cuml.accel.install()` — cuML's zero-code-change entry point, which monkeypatches scikit-learn's estimators in the interpreter that imports them.

It happens per task-running process, not at worker startup: under `executor="process"` the interpreter that imports sklearn is the child, and patching the worker would leave the child running plain CPU sklearn. A module-global flag under a lock keeps it to once per process.

## Usage

### GPU-accelerated training

```python
import skyward as sky


@sky.function
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

    return {"accuracy": scores.mean(), "time": perf_counter() - start}


with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.L4(),
    nodes=1,
    plugins=[
        sky.plugins.Cuml(),
        sky.plugins.Sklearn(),
    ],
) as compute:
    result = train_on_gpu(50000) >> compute
    print(f"Accuracy: {result['accuracy']:.2%}, Time: {result['time']:.1f}s")
```

`Sklearn` installs scikit-learn and joblib; `Cuml` installs cuML and turns on the acceleration. The function uses only scikit-learn imports.

The data is loaded on the node — `fetch_openml` downloads there — rather than serialized and shipped. That is the general shape for data-heavy workloads.

### Without the Sklearn plugin

If your function does not lean on scikit-learn's `n_jobs` parallelism, `Cuml` alone will do, as long as scikit-learn is in the image:

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.L4(),
    nodes=1,
    plugins=[sky.plugins.Cuml()],
    image=sky.Image(pip=["scikit-learn"]),
) as compute:
    result = train_on_gpu(50000) >> compute
```

## Requirements

cuML requires an NVIDIA GPU, and the worker's CUDA version must match the `cuda` parameter. On a CPU-only instance the import fails.

## Next steps

- [cuML guide](../guides/cuml-acceleration.md) — CPU vs GPU comparison
- [Scikit-learn plugin](sklearn.md) — distributed scikit-learn over the compute
- [What are plugins?](index.md) — the hook model
