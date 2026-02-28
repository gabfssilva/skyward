# Scikit-learn

scikit-learn is built on joblib for parallelism. Every estimator and utility that accepts `n_jobs` — `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, `RFECV`, `BaggingClassifier`, `VotingClassifier`, and many others — delegates to `joblib.Parallel` internally. This means the parallelism strategy is pluggable: replace the joblib backend, and every scikit-learn operation that uses `n_jobs` distributes automatically.

Skyward's `sklearn` plugin does exactly this. It installs scikit-learn and joblib on the worker, registers the same `SkywardBackend` that the [joblib plugin](joblib.md) uses, and enters the `parallel_backend("skyward")` context for the duration of the pool. Inside the pool block, `n_jobs=-1` means "all workers in the cluster." No code changes are needed beyond the pool configuration — your existing scikit-learn code works as-is.

## What it does

**Image transform** — Appends `scikit-learn` (optionally at a pinned version) and `joblib` to the worker's pip dependencies. Both are needed on the worker because scikit-learn imports joblib internally, and the `SkywardBackend` dispatches tasks that need to be deserialized in an environment where both packages are available.

**Client lifecycle (`around_client`)** — Reuses the joblib plugin's infrastructure: it calls `_setup_backend(pool)` to register `SkywardBackend`, calls `_strip_local_warning_filters()` to sanitize warning filters (see the [joblib plugin documentation](joblib.md) for why this matters), and enters `parallel_backend("skyward")`. This is the same machinery as the joblib plugin — the sklearn plugin is effectively the joblib plugin plus scikit-learn installation.

## Relationship with the Joblib plugin

The `sklearn` plugin and the `joblib` plugin share the same backend. Under the hood, both register `SkywardBackend` as a custom joblib parallel backend, and both enter the `parallel_backend("skyward")` context. The difference is what they install on the worker:

- `sky.plugins.joblib()` installs only `joblib`.
- `sky.plugins.sklearn()` installs `scikit-learn` and `joblib`.

If your workload is scikit-learn-based, use the `sklearn` plugin alone — it includes everything the `joblib` plugin provides. You do not need to stack both plugins. If your workload is pure joblib without scikit-learn, use the `joblib` plugin.

If you happen to specify both, nothing breaks — the backend registration is idempotent, and duplicate pip packages are harmless. But it is unnecessary.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version` | `str \| None` | `None` | Specific scikit-learn version to install (e.g. `"1.4.0"`). `None` installs the latest version. |

Version pinning is important when your local code depends on specific scikit-learn behavior, or when you need reproducibility across runs. The worker's scikit-learn version should match (or be compatible with) the version used to define the estimators and pipelines, because cloudpickle serializes Python objects and deserializes them in the worker's environment.

## What works with `n_jobs`

Everything in scikit-learn that accepts `n_jobs` distributes across the cluster without modification:

- **`GridSearchCV`** — Each combination of hyperparameters and cross-validation fold is a separate task. A grid with 20 candidates and 5-fold CV produces 100 fits, all distributed.
- **`RandomizedSearchCV`** — Same as `GridSearchCV` but with random sampling. `n_iter=50` with 5-fold CV produces 250 fits.
- **`cross_val_score`** / **`cross_validate`** — Each fold is an independent fit+evaluate. 10-fold CV distributes 10 tasks.
- **`RFECV`** (Recursive Feature Elimination with CV) — Each elimination step and fold is distributed.
- **`BaggingClassifier`** / **`BaggingRegressor`** — Each base estimator is fit independently when `n_jobs=-1`.
- **`VotingClassifier`** / **`VotingRegressor`** — Each constituent estimator is fit independently.
- **`MultiOutputClassifier`** / **`MultiOutputRegressor`** — Each target's estimator is fit independently.
- **`Pipeline` with parallel steps** — When combined with `GridSearchCV`, the full pipeline (preprocessing + estimator) is replicated per task.

The pattern is consistent: scikit-learn calls `joblib.Parallel(n_jobs=self.n_jobs)` internally, the Skyward backend intercepts it, and each unit of work is dispatched to a remote worker.

## Usage

### Grid search

The most common use case is distributing hyperparameter search:

```python
import skyward as sky
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@sky.compute
def run_search() -> dict:
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC()),
    ])

    param_grid = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto", 0.01, 0.001],
        "clf__kernel": ["rbf", "poly"],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    return {
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "test_score": grid.score(X_test, y_test),
    }


with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    worker=sky.Worker(concurrency=4),
    plugins=[sky.plugins.sklearn()],
) as pool:
    result = run_search() >> pool
    print(f"Best: {result['best_params']}, CV={result['best_cv_score']:.2%}")
```

This grid has 32 candidates and 5-fold CV, producing 160 fits. With 4 nodes and `concurrency=4`, 16 fits run in parallel. The `n_jobs=-1` inside `GridSearchCV` tells joblib to use all available workers, which the Skyward backend reports as 16.

Note that the `GridSearchCV` call happens inside a `@sky.compute` function. The grid search itself runs on a remote worker — it is the grid search's internal `Parallel` calls that distribute across the cluster. The outer `>> pool` dispatches the function to one node; that node's joblib backend then fans out the 160 individual fits across all nodes.

### Cross-validation

For a quick evaluation without hyperparameter tuning:

```python
@sky.compute
def evaluate_model() -> dict:
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X, y = load_digits(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(clf, X, y, cv=10, n_jobs=-1)

    return {"mean": scores.mean(), "std": scores.std()}


with sky.ComputePool(
    provider=sky.AWS(),
    nodes=3,
    worker=sky.Worker(concurrency=4),
    plugins=[sky.plugins.sklearn()],
) as pool:
    result = evaluate_model() >> pool
```

Ten-fold CV distributes 10 independent fit+evaluate tasks across 12 workers.

### Combining with cuML

For GPU-accelerated scikit-learn, stack the `cuml` plugin with `sklearn`:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="L4",
    nodes=1,
    plugins=[
        sky.plugins.cuml(),
        sky.plugins.sklearn(),
    ],
) as pool:
    result = train_on_gpu() >> pool
```

The `cuml` plugin intercepts sklearn calls and routes them to GPU. The `sklearn` plugin ensures scikit-learn and joblib are installed. See the [cuML plugin documentation](cuml.md) for details.

## Next steps

- [Scikit Grid Search guide](../guides/scikit-grid-search.md) — Complete example with multiple estimator families and pipeline search
- [joblib plugin](joblib.md) — How `SkywardBackend` works, warning filter sanitization, and tuning concurrency
- [cuML plugin](cuml.md) — GPU-accelerated scikit-learn with NVIDIA RAPIDS
- [What are Plugins?](index.md) — How the plugin system works
