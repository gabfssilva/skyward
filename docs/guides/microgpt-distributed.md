# Distributed MicroGPT with JAX

Andrej Karpathy's [MicroGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) trains a character-level GPT in ~200 lines of pure Python — no frameworks, no dependencies. It's the complete algorithm; everything else is just efficiency. This guide takes that same algorithm, rewrites it in idiomatic JAX, and distributes it across multiple GPUs with Skyward. By the end, you'll have a model that trains on sharded data across a cluster, with automatic gradient synchronization, and generates hallucinated names from what it learned.

## The compute function

The entire training pipeline lives inside a single `@sky.compute` function:

```python
--8<-- "examples/30_flax_minigpt.py:33:46"
```

- **`@sky.compute`** wraps the function into a `PendingCompute` — nothing executes until dispatched to a pool via an operator.
- **`@sky.stdout(only="head")`** silences stdout on all nodes except node 0. Without this, every node would print interleaved training logs.

JAX's distributed runtime is initialized by `sky.plugins.jax()` on the pool (shown in the [Running It](#running-it) section below). All hyperparameters are function arguments with defaults — you can override any of them when calling the function without touching the code.

## Data sharding

Each node downloads the dataset independently, but only trains on its own slice:

```python
--8<-- "examples/30_flax_minigpt.py:73:76"
```

`sky.shard(docs)` splits the list of documents across nodes using modulo striding — node 0 gets indices `[0, N, 2N, ...]`, node 1 gets `[1, N+1, 2N+1, ...]`, and so on. Each node then tokenizes only its local shard and concatenates the tokens into a single flat array on GPU:

```python
--8<-- "examples/30_flax_minigpt.py:82:89"
```

This is data parallelism at the dataset level — the model is identical on every node, but each node sees different training examples.

## Mesh sharding

This is where Skyward's JAX plugin and JAX's SPMD model come together. After initializing parameters, the code sets up a sharding mesh:

```python
--8<-- "examples/30_flax_minigpt.py:65:71"
```

`jax.make_mesh` creates a logical device mesh spanning all devices across all nodes. `NamedSharding` with `PartitionSpec("batch")` means data tensors are split along the batch dimension across devices. `PartitionSpec()` (empty) means parameters are replicated on every device.

Because `sky.plugins.jax()` already initialized the distributed runtime, `jax.device_count()` returns the total number of devices across the entire cluster — not just local ones. The mesh spans the full cluster transparently.

## Training

The training step uses Optax for optimization with warmup + cosine decay + gradient clipping:

```python
--8<-- "examples/30_flax_minigpt.py:235:245"
```

The `@nnx.jit` decorator compiles this into an XLA computation. Because the inputs use `NamedSharding`, JAX automatically inserts all-reduce operations for the gradients — each node computes gradients on its local data shard, and XLA averages them across the mesh before the optimizer step. No explicit `all_reduce` call needed.

The training loop samples mini-batches from the local token array and feeds them through the sharded pipeline:

```python
--8<-- "examples/30_flax_minigpt.py:252:266"
```

`jax.make_array_from_process_local_data` takes each node's local batch and constructs a globally-sharded array that JAX's runtime distributes according to the mesh. From JAX's perspective, it sees a single large batch split across devices.

## Running it

The main block provisions a cluster and broadcasts the training function to all nodes:

```python
--8<-- "examples/30_flax_minigpt.py:358:367"
```

`@ pool` (the matmul operator) broadcasts `train_microgpt()` to every node in the pool. Each node executes the same function with the same hyperparameters, but trains on different data shards. The results — a dict per node with loss and generated samples — are collected back to the caller.

The `jax` plugin is specified on the pool via `plugins=[sky.plugins.jax()]` — it installs JAX with CUDA support and initializes the distributed runtime on each worker before the function runs.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/30_flax_minigpt.py
```

---

**What you learned:**

- **`plugins=[sky.plugins.jax()]`** initializes JAX's distributed runtime automatically — coordinator address, process count, and process id are derived from the cluster topology.
- **`sky.shard()`** splits data across nodes — each node tokenizes and trains on its own partition.
- **Mesh sharding** with `NamedSharding` gives you automatic gradient all-reduce — params replicated, data sharded, XLA handles the communication.
- **`@sky.stdout(only="head")`** keeps training logs clean by silencing non-head nodes.
- **`@ pool` (broadcast)** sends the compute function to all nodes — results are collected as a list.
