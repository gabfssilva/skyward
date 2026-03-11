"""Distributed FSDP Fine-tuning with HuggingFace Transformers on Skyward.

Demonstrates training GPT-2 XL (1.5B) with FSDP sharded across 3 nodes.
FSDP shards parameters, gradients, and optimizer states — fitting a model
that would OOM on a single T4 (16GB) or even two.

Uses sky.plugins.accelerate() which sets ``ACCELERATE_USE_FSDP=true``
and distributed env vars (RANK, WORLD_SIZE, MASTER_ADDR, etc.) before
any task runs.  This ensures ``PartialState()`` — created lazily by
``TrainingArguments`` — detects FSDP mode instead of plain MULTI_GPU.

All FSDP configuration is passed via the plugin config — the task
function itself has no FSDP-specific code.
"""

import skyward as sky


@sky.function(timeout=1800)
def train_fsdp(
    model_name: str,
    dataset_name: str,
    max_samples: int = 2000,
    num_epochs: int = 1,
    batch_size: int = 2,
) -> dict:
    """Fine-tune a model with FSDP across multiple nodes."""
    import os
    import traceback

    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    info = sky.instance_info()

    env_snapshot = {
        k: os.environ.get(k, "<unset>")
        for k in [
            "RANK", "WORLD_SIZE", "LOCAL_RANK",
            "MASTER_ADDR", "MASTER_PORT",
            "ACCELERATE_USE_FSDP",
        ]
    }
    pg_initialized = dist.is_initialized()
    world_size = dist.get_world_size() if pg_initialized else -1

    training_args = TrainingArguments(
        output_dir=f"/tmp/fsdp-{info.node}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-5,
        warmup_steps=10,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        average_tokens_across_devices=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, "wikitext-2-raw-v1", split="train")
    ds = ds.filter(lambda x: len(x["text"]) > 50)
    ds = ds.select(range(min(max_samples, len(ds))))

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds = ds.map(lambda x: {"labels": x["input_ids"]})

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
    )

    param_count = sum(p.numel() for p in model.parameters())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    train_loss = None
    train_error = None
    try:
        train_result = trainer.train()
        train_loss = round(train_result.training_loss, 4)
    except Exception as e:
        train_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    return {
        "node": info.node,
        "env_vars": env_snapshot,
        "pg_initialized": pg_initialized,
        "world_size": world_size,
        "fsdp_enabled": trainer.is_fsdp_enabled,
        "accelerator_type": str(trainer.accelerator.distributed_type),
        "model_params_b": round(param_count / 1e9, 1),
        "peak_gpu_gb": round(peak_gb, 2),
        "final_loss": train_loss,
        "error": train_error,
    }


if __name__ == "__main__":
    MODEL = "gpt2-xl"

    import os

    with sky.Compute(
        provider=sky.AWS(),
        accelerator=sky.accelerators.T4(),
        memory_gb=32,
        nodes=3,
        allocation="spot",
        image=sky.Image(
            pip=["torch", "transformers", "datasets"],
            pip_indexes=[
                sky.PipIndex(
                    url="https://download.pytorch.org/whl/cu128",
                    packages=["torch"],
                ),
            ],
            env={
                "HF_TOKEN": os.environ["HF_TOKEN"],
                "PYTHONUNBUFFERED": "1",
            },
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
        ttl=2400,
    ) as compute:
        import json

        print("=" * 60)
        print(f"FSDP fine-tuning {MODEL}")
        print("=" * 60)

        results = train_fsdp(
            model_name=MODEL,
            dataset_name="wikitext",
            max_samples=500,
            num_epochs=1,
            batch_size=1,
        ) @ compute

        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)

        for r in results:
            print(json.dumps(r, indent=2))
