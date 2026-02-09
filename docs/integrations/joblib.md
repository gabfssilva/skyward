# Joblib Integration

Skyward provides seamless integration with [joblib](https://joblib.readthedocs.io/) for distributed parallel execution. This enables existing joblib-based code, including scikit-learn, to run on cloud infrastructure with minimal changes.

## Overview

The integration provides:

- **JoblibPool**: Distributed joblib execution with `Parallel`
- **ScikitLearnPool**: Distributed scikit-learn with `GridSearchCV`, cross-validation
- **SkywardBackend**: Custom joblib backend for cloud execution
- **sklearn_backend**: Context manager for manual backend activation

## Quick Start

### Basic Parallel Execution

```python
from skyward import AWS
from skyward.integrations import JoblibPool
from joblib import Parallel, delayed

def process(x):
    return x ** 2

# Distribute across 4 cloud nodes
with JoblibPool(provider=AWS(), nodes=4, concurrency=4):
    results = Parallel(n_jobs=-1)(
        delayed(process)(x) for x in range(100)
    )

print(results)  # [0, 1, 4, 9, 16, ...]
```

### Distributed GridSearchCV

```python
from skyward import AWS
from skyward.integrations import ScikitLearnPool
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
}

with ScikitLearnPool(provider=AWS(), nodes=4, concurrency=4):
    grid = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=5,
        n_jobs=-1,  # Distributed!
    )
    grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")
```

## JoblibPool

`JoblibPool` creates a compute pool with the joblib backend pre-configured.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `ProviderLike` | required | Cloud provider (AWS, Verda, VastAI) |
| `nodes` | `int` | `1` | Number of compute nodes |
| `concurrency` | `int` | `1` | Concurrent tasks per node |
| `machine` | `str \| None` | `None` | Direct instance type override |
| `image` | `Image \| None` | `None` | Base image (joblib added automatically) |
| `accelerator` | `str \| None` | `None` | GPU specification |
| `cpu` | `int \| None` | `None` | CPU cores per worker |
| `memory` | `str \| None` | `None` | Memory per worker (e.g., "32GB") |
| `allocation` | `str` | `"spot-if-available"` | Instance allocation strategy |
| `timeout` | `int` | `3600` | Task timeout in seconds |
| `joblib_version` | `str \| None` | `None` | Specific joblib version |

### Total Parallelism

The effective parallelism is `nodes * concurrency`:

```python
# 4 nodes x 4 concurrent = 16 parallel tasks
with JoblibPool(provider=AWS(), nodes=4, concurrency=4):
    # n_jobs=-1 uses all 16 slots
    results = Parallel(n_jobs=-1)(...)
```

### Usage

```python
from skyward import AWS
from skyward.integrations import JoblibPool
from joblib import Parallel, delayed

def expensive_computation(params):
    # CPU-intensive work
    return result

with JoblibPool(
    provider=AWS(),
    nodes=4,
    concurrency=4,
    cpu=8,
    memory="16GB",
    allocation="spot-if-available",
) as pool:
    # Access pool if needed
    print(f"Total slots: {pool.total_slots}")

    results = Parallel(n_jobs=-1)(
        delayed(expensive_computation)(p) for p in params
    )
```

## ScikitLearnPool

`ScikitLearnPool` is specialized for scikit-learn workloads, automatically adding sklearn to dependencies.

### Parameters

Same as `JoblibPool`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sklearn_version` | `str \| None` | `None` | Specific scikit-learn version |

### Usage with GridSearchCV

```python
from skyward import AWS
from skyward.integrations import ScikitLearnPool
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'poly'],
}

with ScikitLearnPool(
    provider=AWS(),
    nodes=4,
    concurrency=4,
    memory="8GB",
):
    grid = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X, y)
```

### Usage with cross_val_score

```python
from skyward import AWS
from skyward.integrations import ScikitLearnPool
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

with ScikitLearnPool(provider=AWS(), nodes=4, concurrency=4):
    scores = cross_val_score(
        GradientBoostingClassifier(),
        X, y,
        cv=10,
        n_jobs=-1,
    )

print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Usage with RandomizedSearchCV

```python
from skyward import AWS
from skyward.integrations import ScikitLearnPool
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint, uniform

param_distributions = {
    'hidden_layer_sizes': [(100,), (100, 50), (100, 100)],
    'alpha': uniform(0.0001, 0.01),
    'learning_rate_init': uniform(0.001, 0.01),
    'max_iter': randint(200, 500),
}

with ScikitLearnPool(provider=AWS(), nodes=8, concurrency=2):
    search = RandomizedSearchCV(
        MLPClassifier(),
        param_distributions,
        n_iter=50,
        cv=3,
        n_jobs=-1,
    )
    search.fit(X, y)
```

## Manual Backend Activation

For existing code or custom pools, use `sklearn_backend`:

```python
from skyward import AWS, ComputePool
from skyward.integrations.joblib import sklearn_backend
from sklearn.model_selection import GridSearchCV

pool = ComputePool(
    provider=AWS(),
    nodes=4,
    image=sky.Image(pip=["scikit-learn", "joblib"]),
)

with pool:
    with sklearn_backend(pool):
        grid = GridSearchCV(model, params, n_jobs=-1)
        grid.fit(X, y)
```

## SkywardBackend (Advanced)

The `SkywardBackend` class implements joblib's `ParallelBackendBase` protocol. It's automatically registered when using `JoblibPool` or `ScikitLearnPool`.

### Manual Registration

```python
from skyward.integrations.joblib import SkywardBackend
from joblib.parallel import register_parallel_backend
from joblib import parallel_backend, Parallel, delayed

# Create pool
pool = ComputePool(...)

# Register backend
register_parallel_backend("skyward", lambda: SkywardBackend(pool))

# Use backend
with parallel_backend("skyward"):
    results = Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)
```

## Best Practices

### 1. Match Concurrency to Resources

Set `concurrency` based on CPU cores or GPU count:

```python
# CPU-bound: match to CPU cores
with JoblibPool(provider=AWS(), nodes=4, cpu=8, concurrency=8):
    ...

# Memory-bound: lower concurrency
with JoblibPool(provider=AWS(), nodes=4, memory="32GB", concurrency=2):
    ...
```

### 2. Use Appropriate Node Count

More nodes = more parallelism, but also more overhead:

```python
# Small search space: fewer nodes
with ScikitLearnPool(provider=AWS(), nodes=2, concurrency=4):
    grid = GridSearchCV(model, small_param_grid, n_jobs=-1)

# Large search space: more nodes
with ScikitLearnPool(provider=AWS(), nodes=8, concurrency=4):
    grid = RandomizedSearchCV(model, large_param_dist, n_iter=100, n_jobs=-1)
```

### 3. Handle Large Data

For large datasets, consider data loading strategy:

```python
import numpy as np
from skyward import AWS, S3Volume
from skyward.integrations import ScikitLearnPool

# Mount S3 for data access
with ScikitLearnPool(
    provider=AWS(),
    nodes=4,
    volume=[S3Volume("/data", "my-bucket", "datasets/")],
):
    # Data loaded from S3 on each node
    X = np.load("/data/X.npy")
    y = np.load("/data/y.npy")

    grid = GridSearchCV(model, params, n_jobs=-1)
    grid.fit(X, y)
```

### 4. Use Spot Instances for Cost Savings

Grid search is fault-tolerant (can restart individual fits):

```python
with ScikitLearnPool(
    provider=AWS(),
    nodes=8,
    allocation="always-spot",  # Maximum savings
):
    grid = GridSearchCV(model, params, n_jobs=-1)
    grid.fit(X, y)
```

## Examples

### Ensemble Model Training

```python
from skyward import AWS
from skyward.integrations import ScikitLearnPool
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ensemble = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('lr', LogisticRegression()),
], voting='soft')

with ScikitLearnPool(provider=AWS(), nodes=4, concurrency=4):
    # Each base estimator's cross-validation runs in parallel
    scores = cross_val_score(ensemble, X, y, cv=5, n_jobs=-1)
```

### Feature Selection

```python
from skyward import AWS
from skyward.integrations import ScikitLearnPool
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

with ScikitLearnPool(provider=AWS(), nodes=4, concurrency=4):
    selector = RFECV(
        RandomForestClassifier(n_estimators=50),
        step=1,
        cv=5,
        n_jobs=-1,
    )
    selector.fit(X, y)
    print(f"Optimal features: {selector.n_features_}")
```

### Multi-Metric Evaluation

```python
from skyward import AWS
from skyward.integrations import ScikitLearnPool
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

with ScikitLearnPool(provider=AWS(), nodes=4, concurrency=4):
    scores = cross_validate(
        RandomForestClassifier(),
        X, y,
        cv=5,
        n_jobs=-1,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
    )

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    key = f'test_{metric}'
    print(f"{metric}: {scores[key].mean():.3f} (+/- {scores[key].std() * 2:.3f})")
```

## Troubleshooting

### "Backend not configured"

Ensure you're using the pool as a context manager:

```python
# Wrong
pool = JoblibPool(provider=AWS(), nodes=4)
Parallel(n_jobs=-1)(...)  # Error!

# Correct
with JoblibPool(provider=AWS(), nodes=4):
    Parallel(n_jobs=-1)(...)  # Works
```

### Low Parallelism

Check `n_jobs=-1` is set:

```python
# Check effective jobs
with JoblibPool(provider=AWS(), nodes=4, concurrency=4) as pool:
    print(f"Total slots: {pool.total_slots}")  # Should be 16

    # Must use n_jobs=-1 for full parallelism
    Parallel(n_jobs=-1)(...)  # Uses all 16 slots
    Parallel(n_jobs=1)(...)   # Only 1 slot!
```

### Serialization Errors

Ensure functions and data are picklable:

```python
import cloudpickle

# Test serialization
try:
    cloudpickle.dumps(my_function)
    cloudpickle.dumps(my_data)
except Exception as e:
    print(f"Serialization failed: {e}")
```

### Slow Performance

- Reduce data transfer by loading data remotely
- Increase task granularity (batch more work per task)
- Use appropriate node/concurrency ratio

---

## Related Topics

- [Integrations Overview](index.md) - All framework integrations
- [Distributed Training](../distributed-training.md) - Deep learning distribution
- [API Reference](../api-reference.md) - Complete API documentation
