# Scikit-learn

scikit-learn is built on joblib for parallelism. Every estimator and utility that accepts `n_jobs` — `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, `RFECV`, `BaggingClassifier`, `VotingClassifier` — delegates to `joblib.Parallel` internally. Replace the joblib backend and every one of them distributes.

`sky.plugins.Sklearn()` does exactly that: it installs scikit-learn and joblib on the nodes, and on the client it points joblib's parallel backend at the compute for the lifetime of the `with` block. Inside that block, `n_jobs=-1` means "every slot in the compute".

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version` | `str \| None` | `None` | Pin, if the code needs one. Otherwise whatever the index has |

Pinning matters when the estimators you build locally have to unpickle on the node: cloudpickle serializes here and deserializes there, and the versions should be compatible.

## How it works

### `image`

Appends `scikit-learn` (optionally at the pinned version) and `joblib` to the image's pip list.

### `client`

Runs in your process, not on a node. It does two things:

1. **Strips non-stdlib warning filters.** scikit-learn's own `Parallel` captures `warnings.filters` and ships them inside every batch. A filter whose category class comes from a third-party package names a module the node may not have, and the batch then fails to unpickle there — a `ModuleNotFoundError` that reads like a missing dependency and is a warning filter. Dropping every filter whose category is not from the stdlib leaves only what every process is guaranteed to carry.
2. **Enters the joblib plugin's client hook**, which registers the Skyward backend and wraps the block in `parallel_backend("skyward")`.

## Relationship with the Joblib plugin

They share one backend. The difference is what lands on the nodes:

- `sky.plugins.Joblib()` installs joblib.
- `sky.plugins.Sklearn()` installs scikit-learn and joblib, and adds the warning-filter scrub.

If your workload is scikit-learn-based, use `Sklearn` alone — it includes everything `Joblib` provides. Stacking both is unnecessary.

## What distributes

Anything in scikit-learn that takes `n_jobs`, without modification: `GridSearchCV` and `RandomizedSearchCV` (one task per candidate/fold), `cross_val_score` and `cross_validate` (one per fold), `RFECV`, the `Bagging*`, `Voting*` and `MultiOutput*` estimators. The pattern is the same throughout — scikit-learn calls `joblib.Parallel(n_jobs=...)`, the Skyward backend intercepts, and each unit of work becomes a task.

The number of parallel slots the backend reports is the compute's node count times the executor's concurrency.

## Usage

### Grid search

```python
import skyward as sky
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC())])
param_grid = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto", 0.01, 0.001],
    "clf__kernel": ["rbf", "poly"],
}

with sky.Compute(
    provider=sky.AWS(),
    nodes=4,
    executor=sky.Executor(concurrency=4),
    plugins=[sky.plugins.Sklearn()],
):
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

print(f"Best: {grid.best_params_}, CV={grid.best_score_:.2%}")
```

32 candidates and 5-fold CV is 160 fits. With 4 nodes at `concurrency=4`, 16 run at a time.

Note the shape: `GridSearchCV` runs **in your process**, and it is its internal `Parallel` calls that fan out. Nothing here is wrapped in `@sky.function` — the client hook is what makes the block distributed.

### Combining with cuML

For GPU-backed estimators, stack [`Cuml`](cuml.md):

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.L4(),
    nodes=1,
    plugins=[
        sky.plugins.Cuml(),
        sky.plugins.Sklearn(),
    ],
) as compute:
    result = train_on_gpu() >> compute
```

## Next steps

- [Scikit Grid Search guide](../guides/scikit-grid-search.md) — a fuller example
- [Joblib plugin](joblib.md) — the backend, and tuning concurrency
- [cuML plugin](cuml.md) — GPU scikit-learn via RAPIDS
- [What are plugins?](index.md) — the hook model
