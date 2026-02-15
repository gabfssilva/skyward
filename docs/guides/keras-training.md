# Keras Training

In this guide you'll train a neural network on cloud GPUs using **Keras 3**. You'll learn the `@keras` integration, backend selection, and data sharding for multi-node training.

## The @keras Integration

Add `@sky.integrations.keras` to configure the backend:

```python
--8<-- "examples/guides/07_keras_training.py:12:13"
```

The decorator sets the Keras backend (JAX, TensorFlow, or PyTorch) and seeds for reproducibility. The backend runs on whatever hardware is available on the remote instance.

## Loading and Sharding Data

Load the full dataset, then use `shard()` to get this node's portion:

```python
--8<-- "examples/guides/07_keras_training.py:19:23"
```

Each node trains on its own shard — with 2 nodes, each gets half the training data.

## Model Definition

Define a standard Keras model — nothing Skyward-specific here:

```python
--8<-- "examples/guides/07_keras_training.py:25:31"
```

## Training

Call `model.fit()` as usual — it runs on the remote GPU:

```python
--8<-- "examples/guides/07_keras_training.py:33:40"
```

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/07_keras_training.py
```

---

**What you learned:**

- **`@sky.integrations.keras`** configures the backend (JAX, TensorFlow, PyTorch).
- **`shard()`** distributes training data across nodes.
- **Standard Keras API** — `model.compile()`, `model.fit()` work unchanged.
- **Backend-agnostic** — switch between JAX, TF, PyTorch with one parameter.
