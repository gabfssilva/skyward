# Keras Training

Keras 3 is backend-agnostic — the same model code runs on JAX, TensorFlow, or PyTorch. Skyward's `@keras` integration configures the backend on the remote worker before your function runs, and `shard()` handles data partitioning for multi-node training. This guide walks through training an MLP on MNIST across multiple cloud GPUs using Keras with JAX as the backend.

## The `@keras` Integration

Add `@sky.integrations.keras` below `@sky.compute`:

```python
--8<-- "examples/guides/07_keras_training.py:11:13"
```

The `backend` parameter sets `KERAS_BACKEND` on the remote worker before Keras is imported. This is critical — Keras reads the backend at import time, so the environment variable must be set first. The `seed` parameter configures random seeds for reproducibility across all backends.

Decorator order matters: `@sky.compute` must be outermost, `@sky.integrations.keras(...)` below it. The integration runs on the remote worker, not on your local machine.

## Loading and Sharding Data

Load the full dataset inside the function, then use `shard()` to get this node's portion:

```python
--8<-- "examples/guides/07_keras_training.py:18:22"
```

`keras.datasets.mnist.load_data()` downloads the dataset on the remote worker. `shard()` then splits the training data so each node trains on a different subset — with 2 nodes, each gets half. The `shuffle=True` and `seed=42` parameters ensure a deterministic, randomized split so both nodes agree on who gets which samples.

Note that sharding happens *inside* the function, after the data is loaded. The full dataset exists on every node (each one downloads it independently), and sharding selects each node's portion based on `instance_info()`. This is simpler than pre-splitting and distributing data from the client.

## Model Definition

Define a standard Keras model — nothing Skyward-specific here:

```python
--8<-- "examples/guides/07_keras_training.py:24:30"
```

This is the same Keras `Sequential` API you'd use locally. The model runs on whatever backend the integration configured — JAX in this case. If you switch to `backend="torch"`, the same model definition produces a PyTorch-backed model.

## Training

Compile and fit as usual:

```python
--8<-- "examples/guides/07_keras_training.py:32:39"
```

`model.fit()` runs on the remote GPU. Each node trains independently on its shard of the data, so training time scales inversely with the number of nodes (minus overhead). The evaluation runs on the full test set — each node evaluates independently and reports its own accuracy.

For synchronized multi-node training with gradient averaging (similar to PyTorch DDP), Keras provides distribution strategies. The `@keras` integration can configure these automatically when running with JAX on multiple nodes. For data-parallel training where each node trains independently on its shard (as in this example), no extra configuration is needed.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/07_keras_training.py
```

---

**What you learned:**

- **`@sky.integrations.keras(backend=...)`** sets the Keras backend on the remote worker before import.
- **`shard()`** splits training data across nodes — each node trains on its own subset.
- **Standard Keras API** — `Sequential`, `model.compile()`, `model.fit()` work unchanged.
- **Backend-agnostic** — switch between JAX, TensorFlow, and PyTorch with one parameter.
- **Data loads on the worker** — no need to transfer datasets from your local machine.
