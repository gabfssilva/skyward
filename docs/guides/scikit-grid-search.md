# Scikit Grid Search

In this guide you'll run a **distributed hyperparameter search** across multiple cloud instances using scikit-learn's `GridSearchCV` and Skyward's joblib backend.

## The Dataset

Load digits and split into train/test:

```python
--8<-- "examples/guides/09_scikit_grid_search.py:15:16"
```

## Defining the Search Space

Use a Pipeline with list-of-dicts `param_grid` to search over both estimators and their hyperparameters:

```python
--8<-- "examples/guides/09_scikit_grid_search.py:18:36"
```

Each dict defines a grid for one estimator family. The `"clf"` key swaps the estimator itself, while `"clf__param"` tunes its hyperparameters.

## Distributed Search with ScikitLearnPool

`ScikitLearnPool` replaces joblib's default backend with a distributed one:

```python
--8<-- "examples/guides/09_scikit_grid_search.py:45:59"
```

Inside the context manager, `n_jobs=-1` distributes fits across all nodes. Each node runs `concurrency` fits in parallel — 3 nodes x 4 concurrent = 12 fits simultaneously.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/09_scikit_grid_search.py
```

---

**What you learned:**

- **`ScikitLearnPool`** distributes joblib work across cloud instances.
- **`n_jobs=-1`** uses all available distributed workers.
- **Standard scikit-learn API** — `GridSearchCV`, `Pipeline`, unchanged.
- **Concurrency per node** multiplies throughput.
