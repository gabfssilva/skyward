# GPU-accelerated scikit-learn with cuML

scikit-learn runs on CPU. For large datasets, training algorithms like RandomForest or KNN becomes a bottleneck — minutes or hours spent on cross-validation and hyperparameter search. [NVIDIA cuML](https://docs.rapids.ai/api/cuml/stable/) provides GPU-backed implementations of popular sklearn estimators with the same API. Swap the import, and the same code runs on GPU with speedups of 50x to 175x.

Skyward makes this practical even if you don't have a local GPU. Provision a GPU instance on the cloud, send your code there with `@sky.compute`, and cuML handles the rest. The `cuml` plugin installs cuML and configures the RAPIDS package indexes automatically.

## The dataset

A 20,000-sample subset of MNIST, downloaded directly on the remote worker — no need to serialize and ship 784-dimensional arrays over the wire:

```python
--8<-- "examples/guides/16_cuml_acceleration.py:14:22"
```

`load_mnist` is a plain function (not `@sky.compute`) that the GPU task calls. Since it's defined at module level, cloudpickle captures it alongside the decorated functions. The data is fetched from OpenML on the remote machine.

## GPU version with cuML

The GPU version uses standard scikit-learn imports — cuML's zero-code-change acceleration intercepts sklearn calls and routes them to the GPU:

```python
--8<-- "examples/guides/16_cuml_acceleration.py:25:40"
```

All sklearn imports are inside the function because `@sky.compute` serializes the function and sends it to a remote worker. The worker needs to resolve imports in its own environment. The function returns both accuracy and wall-clock time so we can compare against a CPU baseline.

## Running with plugins

The `cuml` plugin handles installing `cuml-cu12` and configuring the RAPIDS pip indexes. Combined with the `sklearn` plugin, which adds `scikit-learn` and `joblib`:

```python
--8<-- "examples/guides/16_cuml_acceleration.py:46:54"
```

The plugins handle all dependency management — no need to manually specify pip packages or index URLs in the Image. The `cuml` plugin knows which RAPIDS indexes to configure for the CUDA 12 packages.

## Results

```python
--8<-- "examples/guides/16_cuml_acceleration.py:58:60"
```

Expect accuracy to be roughly equivalent between CPU and GPU — cuML implements the same algorithms, not approximations. The wall-clock time is where the difference shows: cuML on an L4 can be significantly faster depending on the algorithm and dataset size.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/16_cuml_acceleration.py
```

---

**What you learned:**

- **`plugins=[sky.plugins.cuml(), sky.plugins.sklearn()]`** — cuML plugin installs RAPIDS packages and configures indexes; sklearn plugin adds scikit-learn.
- **cuML estimators work with sklearn utilities** — `cross_val_score`, `GridSearchCV`, and `Pipeline` all accept cuML estimators.
- **Zero-code-change acceleration** — cuML intercepts sklearn calls and routes them to the GPU transparently.
- **Plugins handle dependencies** — no manual pip packages or index URLs needed in the Image.
- **Load data on the worker** — avoid serializing large arrays; download datasets directly on the remote machine.
