# Scikit Grid Search

Hyperparameter search is embarrassingly parallel — each candidate configuration can be evaluated independently. scikit-learn's `GridSearchCV` already supports parallelism via `n_jobs`, but it's limited to the cores on a single machine. Skyward's `ScikitLearnPool` extends this to a cluster: it replaces joblib's default backend with a distributed one, so `n_jobs=-1` distributes cross-validation fits across cloud instances instead of local threads.

## The Dataset

Load digits and split into train/test:

```python
--8<-- "examples/guides/09_scikit_grid_search.py:15:16"
```

## Defining the Search Space

Use a Pipeline with a list-of-dicts `param_grid` to search over both estimators and their hyperparameters:

```python
--8<-- "examples/guides/09_scikit_grid_search.py:18:36"
```

Each dict defines a grid for one estimator family. The `"clf"` key swaps the estimator itself (RandomForest, GradientBoosting, SVC), while `"clf__param"` tunes its hyperparameters. scikit-learn expands all combinations — this grid produces 21 candidates, each cross-validated 5 times, for 105 total fits. On a single machine, these run sequentially or across a few cores. On a cluster, they run across dozens of workers simultaneously.

## Distributed Search with `ScikitLearnPool`

`ScikitLearnPool` replaces joblib's default backend so that `Parallel(n_jobs=-1)` — which `GridSearchCV` uses internally — distributes work across cloud instances:

```python
--8<-- "examples/guides/09_scikit_grid_search.py:45:59"
```

Inside the context manager, every joblib `Parallel` call is intercepted and routed to the Skyward cluster. Each fit is serialized with cloudpickle, sent to a worker, executed, and the result returned. The `worker` parameter accepts a `Worker` dataclass that controls per-node execution — `Worker(concurrency=4)` means each node runs 4 fits simultaneously. With 3 nodes and `concurrency=4`, you get 12 parallel fits.

`ScikitLearnPool` is a thin wrapper around `ComputePool` that registers the custom joblib backend on enter and restores the default on exit. The scikit-learn API is completely unchanged — `GridSearchCV`, `Pipeline`, `cross_val_score` all work as documented.

## Results

After the search completes, access results through the standard scikit-learn interface:

```python
best_clf = grid_search.best_params_["clf"]
print(f"Best: {type(best_clf).__name__}, CV={grid_search.best_score_:.2%}")
print(f"Test: {grid_search.score(X_test, y_test):.2%}")
```

The grid search object behaves exactly as it would in a local run — `best_params_`, `best_score_`, `cv_results_` are all populated. The only difference is that the 105 fits ran on a cluster instead of a single machine.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/09_scikit_grid_search.py
```

---

**What you learned:**

- **`ScikitLearnPool`** replaces joblib's backend with a distributed one — `n_jobs=-1` uses all cloud workers.
- **Standard scikit-learn API** — `GridSearchCV`, `Pipeline`, `cross_val_score` work unchanged.
- **`worker=Worker(concurrency=N)`** controls parallelism per node — total parallel fits = nodes x concurrency.
- **Pipeline + param_grid** — search over different estimators and their hyperparameters in one run.
