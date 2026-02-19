# Distributed MicroGPT with JAX

Andrej Karpathy's [MicroGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) trains a character-level GPT in ~200 lines of pure Python — no frameworks, no dependencies. It's the complete algorithm; everything else is just efficiency. This guide takes that same algorithm, rewrites it in idiomatic JAX, and distributes it across multiple GPUs with Skyward. By the end, you'll have a model that trains on sharded data across a cluster, with automatic gradient synchronization, and generates hallucinated names from what it learned.

## The Compute Function

The entire training pipeline lives inside a single `@sky.compute` function:

```python
--8<-- "examples/30_jax_microgpt.py:12:29"
```

Three decorators, each with a distinct role:

- **`@sky.compute`** wraps the function into a `PendingCompute` — nothing executes until dispatched to a pool via an operator.
- **`@sky.integrations.jax()`** initializes JAX's distributed runtime on each worker. It reads the cluster topology from `instance_info()` and calls `jax.distributed.initialize()` with the correct coordinator address, number of processes, and process id.
- **`@sky.stdout(only="head")`** silences stdout on all nodes except node 0. Without this, every node would print interleaved training logs.

All hyperparameters are function arguments with defaults — you can override any of them when calling the function without touching the code.

## Data Sharding

Each node downloads the dataset independently, but only trains on its own slice:

```python
--8<-- "examples/30_jax_microgpt.py:64:69"
```

`sky.shard(docs)` splits the list of documents across nodes using modulo striding — node 0 gets indices `[0, N, 2N, ...]`, node 1 gets `[1, N+1, 2N+1, ...]`, and so on. Each node then tokenizes only its local shard into a matrix of `(num_docs, block_size + 1)` sequences:

```python
--8<-- "examples/30_jax_microgpt.py:72:82"
```

This is data parallelism at the dataset level — the model is identical on every node, but each node sees different training examples.

## The Model

The model is a standard GPT: token embeddings + positional embeddings, N transformer blocks (RMSNorm, multi-head causal attention, MLP with ReLU), and a language model head. All pure functions, no classes:

```python
--8<-- "examples/30_jax_microgpt.py:141:152"
```

Parameters are a nested pytree (dict of `jnp.array`), initialized with small random values:

```python
--8<-- "examples/30_jax_microgpt.py:87:109"
```

Unlike the original gist which processes tokens one at a time with a KV cache, this version processes the full sequence at once with a causal mask — a natural fit for `jax.jit`.

## Mesh Sharding

This is where Skyward's JAX integration and JAX's SPMD model come together. After initializing parameters, the code sets up a sharding mesh:

```python
--8<-- "examples/30_jax_microgpt.py:155:161"
```

`jax.make_mesh` creates a logical device mesh spanning all devices across all nodes. `NamedSharding` with `PartitionSpec("batch")` means data tensors are split along the batch dimension across devices. `PartitionSpec()` (empty) means parameters are replicated on every device.

Because `@sky.integrations.jax()` already initialized the distributed runtime, `jax.device_count()` returns the total number of devices across the entire cluster — not just local ones. The mesh spans the full cluster transparently.

## Training

The loss function computes cross-entropy over next-token predictions, masking out padding tokens:

```python
--8<-- "examples/30_jax_microgpt.py:163:170"
```

The training step uses `value_and_grad` to compute loss and gradients in a single forward-backward pass, then applies Adam with linear learning rate decay:

```python
--8<-- "examples/30_jax_microgpt.py:172:193"
```

The `@jit` decorator compiles this into an XLA computation. Because the inputs use `NamedSharding`, JAX automatically inserts all-reduce operations for the gradients — each node computes gradients on its local data shard, and XLA averages them across the mesh before the optimizer step. No explicit `all_reduce` call needed.

The training loop samples mini-batches from the local token matrix and feeds them through the sharded pipeline:

```python
--8<-- "examples/30_jax_microgpt.py:200:222"
```

`jax.make_array_from_process_local_data` takes each node's local batch and constructs a globally-sharded array that JAX's runtime distributes according to the mesh. From JAX's perspective, it sees a single large batch split across devices.

## Inference

After training, the head node generates sample names using autoregressive decoding:

```python
--8<-- "examples/30_jax_microgpt.py:225:254"
```

The loop starts with a `[BOS]` token and feeds the full sequence buffer through the model at each step. Temperature controls randomness: lower values produce more conservative names, higher values produce more creative ones. Generation stops when the model emits a `BOS` token (end of name) or hits the block size limit.

Only node 0 runs inference — the other nodes have already contributed their share during training via gradient synchronization.

## Running It

The main block provisions a 2-node cluster and broadcasts the training function to all nodes:

```python
--8<-- "examples/30_jax_microgpt.py:279:288"
```

`@ pool` (the matmul operator) broadcasts `train_microgpt()` to every node in the pool. Each node executes the same function with the same hyperparameters, but trains on different data shards. The results — a dict per node with loss and generated samples — are collected back to the caller.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/30_jax_microgpt.py
```

---

**What you learned:**

- **`@sky.integrations.jax()`** initializes JAX's distributed runtime automatically — coordinator address, process count, and process id are derived from the cluster topology.
- **`sky.shard()`** splits data across nodes — each node tokenizes and trains on its own partition.
- **Mesh sharding** with `NamedSharding` gives you automatic gradient all-reduce — params replicated, data sharded, XLA handles the communication.
- **`@sky.stdout(only="head")`** keeps training logs clean by silencing non-head nodes.
- **`@ pool` (broadcast)** sends the compute function to all nodes — results are collected as a list.
