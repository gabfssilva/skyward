# FSDP fine-tuning with HuggingFace

Some models don't fit on a single GPU. GPT-2 XL has 1.5 billion parameters — loading the model weights, gradients, and optimizer states requires far more than the 16 GB a T4 offers. DistributedDataParallel (DDP) doesn't help here because it *replicates* the full model on every node. Fully Sharded Data Parallelism (FSDP) solves this by *sharding* parameters, gradients, and optimizer states across nodes. Each node holds only a fraction of the model, and parameters are gathered on-the-fly during forward and backward passes.

Skyward's `accelerate` plugin configures the entire FSDP environment — topology, sharding strategy, wrapping policy — through a single config dict. The training function itself is standard HuggingFace Trainer code with zero FSDP-specific logic.

## The `accelerate` plugin

Add `sky.plugins.accelerate()` to your pool's plugins with an FSDP config:

```python
--8<-- "examples/guides/19_fsdp_huggingface.py:102:114"
```

The plugin does two things before any task runs. First, it sets the distributed topology env vars (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) and calls `torch.distributed.init_process_group()`. Second, it sets `ACCELERATE_USE_FSDP=true` and translates the `fsdp` dict into the `FSDP_*` environment variables that HuggingFace Accelerate reads.

This ordering matters. `TrainingArguments.__post_init__` triggers `PartialState()` creation, which reads `ACCELERATE_USE_FSDP` to decide whether to use FSDP or plain multi-GPU. If the env var is missing at that point, the singleton locks to `MULTI_GPU` and FSDP never activates. The plugin ensures everything is set before any training code runs.

## FSDP config explained

The `fsdp` dict maps directly to Accelerate's FSDP environment variables:

| Key | What it does |
|-----|-------------|
| `sharding_strategy` | `FULL_SHARD` shards params, gradients, and optimizer states across all nodes. Most memory-efficient. |
| `auto_wrap_policy` | `TRANSFORMER_BASED_WRAP` wraps each transformer block as an FSDP unit. |
| `transformer_layer_cls_to_wrap` | The class name to wrap — `GPT2Block` for GPT-2 models. |
| `backward_prefetch` | `BACKWARD_PRE` prefetches the next layer's params during backward, overlapping communication with compute. |
| `sync_module_states` | Broadcasts module states from rank 0 so all nodes start with identical weights. |
| `use_orig_params` | Keeps original parameter references, required for optimizer compatibility. |
| `cpu_ram_efficient_loading` | Loads model weights on CPU first, then moves to GPU — avoids doubling memory usage during init. |

## The training function

The function is a standard `@sky.function` — no FSDP imports, no process group setup, no sharding logic:

```python
--8<-- "examples/guides/19_fsdp_huggingface.py:6:13"
```

All imports happen inside the function body:

```python
--8<-- "examples/guides/19_fsdp_huggingface.py:15:18"
```

This is intentional — `torch`, `transformers`, and `datasets` are only installed on the remote workers (via the Image's `pip` field), not on your local machine. Skyward serializes the function with cloudpickle and ships it over SSH, so remote imports keep your local environment clean.

## Loading the tokenizer

```python
--8<-- "examples/guides/19_fsdp_huggingface.py:23:25"
```

GPT-2 doesn't have a pad token by default — it was trained as a pure autoregressive model with no padding. Setting `pad_token = eos_token` is the standard workaround for fine-tuning, where batches need uniform sequence lengths.

## Preparing the dataset

```python
--8<-- "examples/guides/19_fsdp_huggingface.py:27:40"
```

The dataset is loaded and tokenized on each node independently. The filter removes short texts (under 50 characters) that would produce mostly padding. `max_length=256` with `padding="max_length"` produces fixed-length sequences — required for efficient batching. The final `map` copies `input_ids` to `labels`, which is how causal language models learn: predict the next token from the previous ones.

Note that every node loads and tokenizes the full dataset. FSDP shards the *model*, not the data — the Trainer's internal `DistributedSampler` ensures each node trains on a different subset of samples.

## Loading the model

```python
--8<-- "examples/guides/19_fsdp_huggingface.py:42:43"
```

`low_cpu_mem_usage=True` loads weights progressively instead of allocating the full model in CPU memory first. For GPT-2 XL (1.5B parameters, ~6 GB in fp32), this avoids a memory spike during initialization. Combined with `cpu_ram_efficient_loading` in the FSDP config, the model goes from disk to CPU to GPU shard without ever fully materializing on a single device.

## Configuring the Trainer

```python
--8<-- "examples/guides/19_fsdp_huggingface.py:45:64"
```

The `TrainingArguments` are standard HuggingFace — nothing FSDP-specific. A few settings worth noting:

- **`gradient_accumulation_steps=4`** — Each node accumulates gradients over 4 micro-batches before synchronizing. This effectively multiplies the batch size without increasing memory.
- **`gradient_checkpointing=True`** — Trades compute for memory by recomputing activations during backward instead of storing them. Essential for fitting large models — it roughly halves the activation memory at the cost of ~30% more compute.
- **`gradient_checkpointing_kwargs={"use_reentrant": False}`** — Uses the newer, more reliable checkpointing implementation. The reentrant version has known issues with FSDP.
- **`save_strategy="no"`** — Disables checkpointing. The instances are ephemeral — saved checkpoints would be lost on teardown.

The Trainer detects FSDP from the `ACCELERATE_USE_FSDP` environment variable the plugin set. When it creates `PartialState()`, the singleton sees the FSDP env vars and activates sharded training. From this point on, the Trainer handles parameter sharding, gradient synchronization, and cross-node communication automatically.

## Training and results

```python
--8<-- "examples/guides/19_fsdp_huggingface.py:66:76"
```

`trainer.train()` runs the full training loop — forward, backward, gradient sync, optimizer step — across all FSDP-sharded nodes. Each node reports back its peak GPU memory and the final training loss. `trainer.is_fsdp_enabled` confirms that FSDP actually activated — useful for debugging configuration issues.

## `accelerate` vs `torch` plugin

Both plugins set up distributed PyTorch, but they target different use cases:

- **`sky.plugins.torch()`** — Sets topology env vars and calls `init_process_group()`. Use for DDP, manual `all_reduce`, or any code that manages distribution itself.
- **`sky.plugins.accelerate()`** — Does everything `torch()` does, plus configures FSDP/DeepSpeed via Accelerate env vars. Use when training with HuggingFace Trainer or Accelerate and you need FSDP or DeepSpeed.

If your model fits on a single GPU and you just want data parallelism, `torch()` with DDP is simpler. If your model doesn't fit — or you want features like mixed-precision offloading — use `accelerate()` with FSDP.

## Running it

The main block provisions 3 spot T4 instances with 32 GB RAM each, broadcasts `train_fsdp()` to all nodes via `@ compute`, and collects the results:

```python
--8<-- "examples/guides/19_fsdp_huggingface.py:82:133"
```

`@ compute` sends the function to every node in the pool. Each node runs the same Trainer code, FSDP coordinates the parameter sharding across them, and results come back as a list — one dict per node with loss, memory usage, and whether FSDP was active.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/19_fsdp_huggingface.py
```

---

**What you learned:**

- **`sky.plugins.accelerate(config={...})`** configures FSDP environment variables and initializes the process group — the training function has zero FSDP code.
- **FSDP shards everything** — parameters, gradients, and optimizer states are distributed across nodes, fitting models that would OOM on a single GPU.
- **Config dict maps to Accelerate env vars** — `sharding_strategy`, `auto_wrap_policy`, `transformer_layer_cls_to_wrap`, and other keys translate directly to `FSDP_*` variables.
- **`accelerate` vs `torch`** — use `accelerate()` for FSDP/DeepSpeed with HuggingFace Trainer; use `torch()` for DDP or manual distributed code.
- **Standard HuggingFace code** — the training function has zero FSDP-specific logic; the plugin handles everything before the function runs.
