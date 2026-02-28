# Keras training

Keras 3 is backend-agnostic — the same model code runs on JAX, TensorFlow, or PyTorch. Skyward's `keras` plugin configures the backend on the remote worker before your function runs, and `shard()` handles data partitioning for multi-node training. This guide walks through training an MLP on MNIST across multiple cloud GPUs using Keras with JAX as the backend.

## The `keras` plugin

Add `sky.plugins.keras(backend="jax")` to your pool's plugins. When using the JAX backend, also include `sky.plugins.jax()` for distributed initialization:

```python
--8<-- "examples/guides/07_keras_training.py:49:54"
```

The `backend` parameter sets `KERAS_BACKEND` on the remote worker before Keras is imported. This is critical — Keras reads the backend at import time, so the environment variable must be set first.

The function itself just uses `@sky.compute` — the backend and distributed setup are handled by the plugins:

```python
--8<-- "examples/guides/07_keras_training.py:11:13"
```

## Loading and sharding data

Load the full dataset inside the function, then use `shard()` to get this node's portion:

```python
--8<-- "examples/guides/07_keras_training.py:17:21"
```

`keras.datasets.mnist.load_data()` downloads the dataset on the remote worker. `shard()` then splits the training data so each node trains on a different subset — with 2 nodes, each gets half. The `shuffle=True` and `seed=42` parameters ensure a deterministic, randomized split so both nodes agree on who gets which samples.

Note that sharding happens *inside* the function, after the data is loaded. The full dataset exists on every node (each one downloads it independently), and sharding selects each node's portion based on `instance_info()`. This is simpler than pre-splitting and distributing data from the client.

## Model definition

Define a standard Keras model — nothing Skyward-specific here:

```python
--8<-- "examples/guides/07_keras_training.py:23:29"
```

This is the same Keras `Sequential` API you'd use locally. The model runs on whatever backend the plugin configured — JAX in this case. If you switch to `backend="torch"`, the same model definition produces a PyTorch-backed model.

## Training

Compile and fit as usual:

```python
--8<-- "examples/guides/07_keras_training.py:31:38"
```

`model.fit()` runs on the remote GPU. Each node trains independently on its shard of the data, so training time scales inversely with the number of nodes (minus overhead). The evaluation runs on the full test set — each node evaluates independently and reports its own accuracy.

For synchronized multi-node training with gradient averaging (similar to PyTorch DDP), Keras provides distribution strategies. The `keras` plugin can configure these automatically when running with JAX on multiple nodes. For data-parallel training where each node trains independently on its shard (as in this example), no extra configuration is needed.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/07_keras_training.py
```

---

**What you learned:**

- **`plugins=[sky.plugins.jax(), sky.plugins.keras(backend="jax")]`** sets the Keras backend and configures JAX distributed on the remote worker.
- **`shard()`** splits training data across nodes — each node trains on its own subset.
- **Standard Keras API** — `Sequential`, `model.compile()`, `model.fit()` work unchanged.
- **Backend-agnostic** — switch between JAX, TensorFlow, and PyTorch with one parameter.
- **Data loads on the worker** — no need to transfer datasets from your local machine.
