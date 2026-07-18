# Accelerate

Hugging Face Accelerate wraps PyTorch's distributed primitives — FSDP, DeepSpeed, DDP — behind one API, so training code written for a single GPU runs across many. The usual workflow is `accelerate config` to write a YAML file, then `accelerate launch` instead of `python`: the launcher reads the YAML, spawns one process per rank, sets several dozen environment variables, and initializes the backend before your code runs.

**Skyward has no `accelerate launch`.** The launcher spawns the training process itself and exits when it returns, which is the opposite of a worker that is started once and lives to take many tasks. So `sky.plugins.Accelerate()` does what the launcher would have done and no more: it sets the environment `accelerate` reads and forms the process group, in the process that runs the task, before that task runs.

## Why the timing matters

`TrainingArguments.__post_init__` touches `self.device` before it looks at `ACCELERATE_USE_FSDP`. That first touch constructs the `PartialState` singleton and freezes the distributed type for the process. If the FSDP flag is not already in the environment by then, the singleton locks to plain multi-GPU and FSDP never turns on.

Setting the whole environment before the first task is what keeps that from happening.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `dict[str, Any]` | `{}` | The accelerate settings, in the same shape as the YAML `accelerate config` writes |

`fsdp` turns on FSDP, `deepspeed` turns on DeepSpeed, `mixed_precision` sets the dtype, and `backend` picks the process-group backend (default `"nccl"`). Topology — rank, world size, address — is injected from the node and is never read from here.

`Accelerate` is a **collective** plugin: a compute running one cannot be resized.

### `fsdp`

| Key | Mapped env var | Values |
|-----|---------------|--------|
| `sharding_strategy` | `FSDP_SHARDING_STRATEGY` | `"FULL_SHARD"`, `"SHARD_GRAD_OP"`, `"NO_SHARD"`, `"HYBRID_SHARD"`, `"HYBRID_SHARD_ZERO2"` |
| `auto_wrap_policy` | `FSDP_AUTO_WRAP_POLICY` | `"TRANSFORMER_BASED_WRAP"`, `"SIZE_BASED_WRAP"` |
| `transformer_layer_cls_to_wrap` | `FSDP_TRANSFORMER_CLS_TO_WRAP` | Module class name, e.g. `"GPT2Block"`, `"LlamaDecoderLayer"` |
| `backward_prefetch` | `FSDP_BACKWARD_PREFETCH` | `"BACKWARD_PRE"`, `"BACKWARD_POST"` |
| `state_dict_type` | `FSDP_STATE_DICT_TYPE` | `"FULL_STATE_DICT"`, `"SHARDED_STATE_DICT"`, `"LOCAL_STATE_DICT"` |
| `forward_prefetch` | `FSDP_FORWARD_PREFETCH` | bool |
| `use_orig_params` | `FSDP_USE_ORIG_PARAMS` | bool |
| `cpu_ram_efficient_loading` | `FSDP_CPU_RAM_EFFICIENT_LOADING` | bool |
| `sync_module_states` | `FSDP_SYNC_MODULE_STATES` | bool |
| `offload_params` | `FSDP_OFFLOAD_PARAMS` | bool |
| `activation_checkpointing` | `FSDP_ACTIVATION_CHECKPOINTING` | bool |
| `min_num_params` | `FSDP_MIN_NUM_PARAMS` | int |

### `deepspeed`

| Key | Mapped env var | Values |
|-----|---------------|--------|
| `zero_stage` | `ACCELERATE_DEEPSPEED_ZERO_STAGE` | 0, 1, 2, 3 |
| `offload_optimizer_device` | `ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE` | `"none"`, `"cpu"`, `"nvme"` |
| `offload_param_device` | `ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE` | `"none"`, `"cpu"`, `"nvme"` |
| `offload_optimizer_nvme_path` | `ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_NVME_PATH` | path |
| `offload_param_nvme_path` | `ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_NVME_PATH` | path |
| `gradient_accumulation_steps` | `ACCELERATE_GRADIENT_ACCUMULATION_STEPS` | int |
| `gradient_clipping` | `ACCELERATE_GRADIENT_CLIPPING` | float |
| `zero3_save_16bit_model` | `ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL` | bool |
| `config_file` | `ACCELERATE_DEEPSPEED_CONFIG_FILE` | path to a full DeepSpeed JSON config |

Keys not in these tables are passed through under their own uppercased name.

### Top level

| Key | Mapped env var | Values |
|-----|---------------|--------|
| `mixed_precision` | `ACCELERATE_MIXED_PRECISION` | `"no"`, `"fp16"`, `"bf16"`, `"fp8"` |
| `backend` | — (used directly) | `"nccl"` (default), `"gloo"` |

## How it works

### `image`

Appends `accelerate` to the image's pip list. It does **not** install PyTorch — add it via `Image(pip=[...])` with the right CUDA wheel index, or pair with [`sky.plugins.Torch()`](torch.md).

### `run`

On the first task in each process, and only when the compute has more than one node:

1. Sets the topology: `MASTER_ADDR` (rank zero's address), `MASTER_PORT` (29500), `WORLD_SIZE`, `RANK`, `NODE_RANK`, and `LOCAL_RANK`/`LOCAL_WORLD_SIZE` fixed at `0`/`1`.
2. Sets the `ACCELERATE_*`, `FSDP_*` and `ACCELERATE_DEEPSPEED_*` variables derived from `config`.
3. Calls `dist.init_process_group(backend=..., init_method="env://", ...)`, unless a group already exists.

A module-global flag under a lock keeps this to once per process. On a single node there is nobody to rendezvous with, so nothing happens at all.

It is a `run` hook rather than `setup` because `init_process_group` is a collective: the process that blocks in it must be the one that runs the collective code afterwards. Under `executor="process"` that is the child, not the worker.

### Composing with the Torch plugin

Unlike the torch plugin, `Accelerate` checks `dist.is_initialized()` before forming the group. So the two can be listed together, and the group is paid for once — but **`Torch` must come first**, since it initializes unconditionally:

```python
plugins=[sky.plugins.Torch(cuda="cu128"), sky.plugins.Accelerate(config={...})]
```

That combination is worth it when you want the torch plugin's wheel-index handling rather than writing it into the `Image` yourself. `Accelerate` on its own forms the group perfectly well; it just does not install torch.

## Usage

### FSDP fine-tuning

```python
import skyward as sky


@sky.function(timeout=1800)
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
        lambda x: {
            **tokenizer(x["text"], truncation=True, max_length=256, padding="max_length"),
            "labels": tokenizer(x["text"], truncation=True, max_length=256, padding="max_length")["input_ids"],
        },
        batched=True,
        remove_columns=ds.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"/tmp/fsdp-{info.rank}",
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
        "rank": info.rank,
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
        sky.plugins.Accelerate(config={
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
        print(f"Rank {r['rank']}: fsdp={r['fsdp_enabled']}, loss={r['loss']}, peak={r['peak_gpu_gb']}GB")
```

`@` broadcasts to all three nodes. FSDP shards GPT-2 XL's 1.5B parameters across them, so each T4 holds a fraction of the model — without FSDP it would not fit. `sync_module_states=True` broadcasts rank zero's weights at init; `cpu_ram_efficient_loading=True` keeps the initial load on CPU to avoid a GPU memory spike.

The task function contains no FSDP-specific code. `Trainer` detects FSDP from the environment the plugin set and wraps the model itself.

### DeepSpeed ZeRO

```python
plugins=[
    sky.plugins.Accelerate(config={
        "mixed_precision": "bf16",
        "deepspeed": {
            "zero_stage": 3,
            "offload_optimizer_device": "cpu",
            "offload_param_device": "cpu",
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
        },
    }),
]
```

ZeRO stage 3 partitions parameters, gradients and optimizer states across nodes; with CPU offloading they spill to host RAM when not needed on the GPU. `deepspeed` itself has to be in the image.

`fsdp` takes precedence: if both sections are present, only FSDP is configured.

### Mixed precision only

```python
plugins=[sky.plugins.Accelerate(config={"mixed_precision": "bf16"})]
```

With neither `fsdp` nor `deepspeed`, the plugin sets the topology and the precision — plain multi-GPU DDP with AMP.

## Next steps

- [FSDP with HuggingFace guide](../guides/fsdp-huggingface.md) — a full walkthrough
- [PyTorch plugin](torch.md) — DDP without accelerate
- [What are plugins?](index.md) — the hook model
- [Distributed Training](../distributed-training.md) — where plugins fit
