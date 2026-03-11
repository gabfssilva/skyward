# Accelerate

Hugging Face Accelerate is an abstraction layer for distributed training. It wraps PyTorch's distributed primitives — FSDP, DeepSpeed, DDP — behind a unified API so that training code written for a single GPU works across multiple nodes without changes. The typical workflow is: run `accelerate config` to generate a YAML file, then launch your script with `accelerate launch` instead of `python`. Accelerate reads the YAML, sets dozens of environment variables (`ACCELERATE_USE_FSDP`, `FSDP_SHARDING_STRATEGY`, `RANK`, `WORLD_SIZE`, etc.), and initializes the distributed backend before your code runs.

The challenge with Skyward is that there is no `accelerate launch`. Skyward runs a long-lived worker process on each node, and tasks are dispatched to it over SSH. The worker is already running when your `@sky.function` executes — there is no opportunity to wrap the process startup with a CLI launcher. The `accelerate` plugin solves this by setting the same environment variables that `accelerate launch` would have set, at the right time: inside the `around_process` hook, before `PartialState()` is created.

This timing matters. When you instantiate `TrainingArguments`, its `__post_init__` accesses `self.device`, which triggers `PartialState()` creation. `PartialState` is a singleton — once created, it locks the distributed type for the entire process. If `ACCELERATE_USE_FSDP` is not in the environment at that moment, the singleton detects `MULTI_GPU` instead of `FSDP`, and FSDP never activates. The plugin ensures that every FSDP and DeepSpeed variable is present before any task runs.

## What it does

The plugin installs `accelerate` on the remote worker and configures the distributed environment for accelerate-based training (FSDP, DeepSpeed, mixed precision) by setting environment variables and initializing the torch process group.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AccelerateConfig \| None` | `None` | Accelerate settings. Mirrors the YAML structure produced by `accelerate config`. Topology fields (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) are injected from the cluster automatically. |

The `config` dictionary supports three top-level sections:

### `fsdp` (FsdpConfig)

| Key | Type | Mapped env var | Description |
|-----|------|---------------|-------------|
| `sharding_strategy` | `str` | `FSDP_SHARDING_STRATEGY` | `"FULL_SHARD"`, `"SHARD_GRAD_OP"`, `"NO_SHARD"`, `"HYBRID_SHARD"`, `"HYBRID_SHARD_ZERO2"` |
| `auto_wrap_policy` | `str` | `FSDP_AUTO_WRAP_POLICY` | `"TRANSFORMER_BASED_WRAP"` or `"SIZE_BASED_WRAP"` |
| `transformer_layer_cls_to_wrap` | `str` | `FSDP_TRANSFORMER_CLS_TO_WRAP` | Module class name for transformer wrapping (e.g. `"GPT2Block"`, `"LlamaDecoderLayer"`) |
| `backward_prefetch` | `str` | `FSDP_BACKWARD_PREFETCH` | `"BACKWARD_PRE"` or `"BACKWARD_POST"` |
| `state_dict_type` | `str` | `FSDP_STATE_DICT_TYPE` | `"FULL_STATE_DICT"`, `"SHARDED_STATE_DICT"`, or `"LOCAL_STATE_DICT"` |
| `forward_prefetch` | `bool` | `FSDP_FORWARD_PREFETCH` | Enable forward prefetch for overlapping communication |
| `use_orig_params` | `bool` | `FSDP_USE_ORIG_PARAMS` | Use original parameter names (required for some optimizers) |
| `cpu_ram_efficient_loading` | `bool` | `FSDP_CPU_RAM_EFFICIENT_LOADING` | Load model weights on CPU to reduce GPU memory spike |
| `sync_module_states` | `bool` | `FSDP_SYNC_MODULE_STATES` | Sync module states from rank 0 to all ranks at init |
| `offload_params` | `bool` | `FSDP_OFFLOAD_PARAMS` | Offload parameters to CPU when not in use |
| `min_num_params` | `int` | `FSDP_MIN_NUM_PARAMS` | Minimum number of parameters for size-based wrapping |
| `activation_checkpointing` | `bool` | `FSDP_ACTIVATION_CHECKPOINTING` | Trade compute for memory by recomputing activations |

### `deepspeed` (DeepSpeedConfig)

| Key | Type | Mapped env var | Description |
|-----|------|---------------|-------------|
| `zero_stage` | `int` | `ACCELERATE_DEEPSPEED_ZERO_STAGE` | ZeRO optimization stage (0, 1, 2, or 3) |
| `offload_optimizer_device` | `str` | `ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE` | `"none"`, `"cpu"`, or `"nvme"` |
| `offload_param_device` | `str` | `ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE` | `"none"`, `"cpu"`, or `"nvme"` |
| `offload_optimizer_nvme_path` | `str` | `ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_NVME_PATH` | Path for NVMe optimizer offload |
| `offload_param_nvme_path` | `str` | `ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_NVME_PATH` | Path for NVMe parameter offload |
| `gradient_accumulation_steps` | `int` | `ACCELERATE_GRADIENT_ACCUMULATION_STEPS` | Number of gradient accumulation steps |
| `gradient_clipping` | `float` | `ACCELERATE_GRADIENT_CLIPPING` | Maximum gradient norm |
| `zero3_save_16bit_model` | `bool` | `ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL` | Save 16-bit model for ZeRO stage 3 |
| `config_file` | `str` | `ACCELERATE_DEEPSPEED_CONFIG_FILE` | Path to a full DeepSpeed JSON config |

### Top-level keys

| Key | Type | Mapped env var | Description |
|-----|------|---------------|-------------|
| `mixed_precision` | `str` | `ACCELERATE_MIXED_PRECISION` | `"no"`, `"fp16"`, `"bf16"`, or `"fp8"` |

## How it works

### Image transform

The `transform` hook appends `"accelerate"` to the image's pip packages. This installs the `accelerate` library on the remote worker alongside whatever other packages the image already includes. Since the plugin does not install PyTorch itself, you typically combine it with an explicit `torch` in your `Image(pip=[...])` or with `sky.plugins.torch()`.

### Worker lifecycle (`around_process`)

The `around_process` hook configures the distributed environment once per executor subprocess. When the first task arrives:

1. Imports `torch` and `torch.distributed` (remote-only imports).
2. Reads `instance_info()` from the hook's parameter to get the cluster topology.
3. If the cluster has fewer than 2 nodes, yields immediately — no distributed setup needed.
4. Sets the base topology environment variables: `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK` (always `"0"`), `LOCAL_WORLD_SIZE` (always `"1"`), and `NODE_RANK`.
5. If the config contains an `fsdp` section, sets `ACCELERATE_USE_FSDP=true` and maps each FSDP key to its corresponding environment variable.
6. If the config contains a `deepspeed` section instead, sets `ACCELERATE_USE_DEEPSPEED=true` and maps each DeepSpeed key to its corresponding environment variable.
7. If `mixed_precision` is set, writes `ACCELERATE_MIXED_PRECISION`.
8. Selects the backend: `"nccl"` when `torch.cuda.is_available()`, `"gloo"` otherwise.
9. Calls `dist.init_process_group(backend=..., init_method="env://")`.
10. Yields to the worker lifecycle — subsequent tasks run with the process group and all accelerate variables already active.
11. On worker shutdown, calls `dist.destroy_process_group()` in the `finally` block.

The environment variables are set before `init_process_group`, which means they are present when `TrainingArguments` later creates the `PartialState` singleton. This is the critical ordering that makes FSDP and DeepSpeed work without `accelerate launch`.

## Usage

### FSDP fine-tuning

```python
import skyward as sky


@sky.function(timeout=1800)
@sky.stdout(only="head")
def finetune(model_name: str) -> dict:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    info = sky.instance_info()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.filter(lambda x: len(x["text"]) > 50).select(range(500))
    ds = ds.map(
        lambda x: {**tokenizer(x["text"], truncation=True, max_length=256, padding="max_length"),
                    "labels": tokenizer(x["text"], truncation=True, max_length=256, padding="max_length")["input_ids"]},
        batched=True,
        remove_columns=ds.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"/tmp/fsdp-{info.node}",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=2e-5,
            fp16=torch.cuda.is_available(),
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=ds,
        processing_class=tokenizer,
    )

    result = trainer.train()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    return {
        "node": info.node,
        "fsdp_enabled": trainer.is_fsdp_enabled,
        "loss": round(result.training_loss, 4),
        "peak_gpu_gb": round(peak_gb, 2),
    }


with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.T4(),
    memory_gb=32,
    nodes=3,
    image=sky.Image(
        pip=["torch", "transformers", "datasets"],
        pip_indexes=[
            sky.PipIndex(url="https://download.pytorch.org/whl/cu128", packages=["torch"]),
        ],
    ),
    plugins=[
        sky.plugins.accelerate(config={
            "mixed_precision": "fp16",
            "fsdp": {
                "sharding_strategy": "FULL_SHARD",
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "transformer_layer_cls_to_wrap": "GPT2Block",
                "backward_prefetch": "BACKWARD_PRE",
                "state_dict_type": "SHARDED_STATE_DICT",
                "sync_module_states": True,
                "use_orig_params": True,
                "cpu_ram_efficient_loading": True,
            },
        }),
    ],
) as compute:
    results = finetune("gpt2-xl") @ compute
    for r in results:
        print(f"Node {r['node']}: fsdp={r['fsdp_enabled']}, loss={r['loss']}, peak={r['peak_gpu_gb']}GB")
```

The `@` operator broadcasts `finetune()` to all 3 nodes. FSDP shards the GPT-2 XL parameters (1.5B) across the nodes, so each T4 (16GB) only holds a fraction of the model. Without FSDP, this model would OOM on a single T4. The `sync_module_states=True` ensures that rank 0's weights are broadcast to all ranks at initialization, and `cpu_ram_efficient_loading=True` keeps the initial load on CPU to avoid a GPU memory spike.

Notice that the task function has zero FSDP-specific code. The `Trainer` auto-detects FSDP from the environment variables set by the plugin and wraps the model internally.

### DeepSpeed ZeRO

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    nodes=4,
    image=sky.Image(
        pip=["torch", "transformers", "datasets", "deepspeed"],
        pip_indexes=[
            sky.PipIndex(url="https://download.pytorch.org/whl/cu128", packages=["torch"]),
        ],
    ),
    plugins=[
        sky.plugins.accelerate(config={
            "mixed_precision": "bf16",
            "deepspeed": {
                "zero_stage": 3,
                "offload_optimizer_device": "cpu",
                "offload_param_device": "cpu",
                "gradient_accumulation_steps": 4,
                "gradient_clipping": 1.0,
            },
        }),
    ],
) as compute:
    results = finetune("meta-llama/Llama-2-7b-hf") @ compute
```

DeepSpeed ZeRO stage 3 partitions parameters, gradients, and optimizer states across all nodes. With CPU offloading enabled, optimizer states and parameters spill to host RAM when not needed on the GPU, allowing you to train models much larger than GPU memory. The plugin sets `ACCELERATE_USE_DEEPSPEED=true` and maps each key to the corresponding `ACCELERATE_DEEPSPEED_*` variable.

### Mixed precision only

```python
plugins=[
    sky.plugins.accelerate(config={"mixed_precision": "bf16"}),
]
```

When neither `fsdp` nor `deepspeed` is present in the config, the plugin sets up the basic distributed topology (`RANK`, `WORLD_SIZE`, etc.) and mixed precision. This is equivalent to multi-GPU DDP with automatic mixed precision — the simplest accelerate configuration.

### Why not `sky.plugins.torch()`?

The accelerate plugin is self-contained: it sets all topology environment variables and calls `dist.init_process_group()` itself. The `torch` plugin does the same thing. Using both would attempt to initialize the process group twice, which fails. Choose one or the other — never both.

The accelerate plugin does not install PyTorch for you (it only installs `accelerate`). Add PyTorch and your other dependencies via `Image(pip=[...])` with the appropriate CUDA wheel index, as shown in the examples above. If you need torchvision or torchaudio, add them to the image the same way.

## Next steps

- [FSDP with HuggingFace guide](../guides/fsdp-huggingface.md) — Step-by-step FSDP fine-tuning walkthrough
- [PyTorch plugin](torch.md) — Lower-level DDP setup without accelerate
- [What are Plugins?](index.md) — How the plugin system works
- [Distributed Training](../distributed-training.md) — How plugins fit into multi-node training
