"""FSDP with HuggingFace Trainer + sky.plugins.torch() — no accelerate plugin.

Tests whether the Trainer's FSDP integration works when the process group
is already initialized by sky.plugins.torch().
"""

import skyward as sky


@sky.function(timeout=600)
def test_trainer_fsdp(model_name: str = "gpt2") -> dict:
    """Fine-tune GPT-2 (small, 124M params) with Trainer + FSDP."""
    import os

    import torch
    import torch.distributed as dist

    info = sky.instance_info()

    # ── Diagnostics ──────────────────────────────────────────────
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
    rank = dist.get_rank() if pg_initialized else -1

    # ── Trainer with FSDP ────────────────────────────────────────
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    distributed = pg_initialized and world_size > 1

    # Tell accelerate's PartialState that we want FSDP, not plain MULTI_GPU.
    # Without this, PartialState sees an initialized PG but defaults to
    # DistributedType.MULTI_GPU, and the Trainer silently ignores fsdp=.
    if distributed:
        os.environ["ACCELERATE_USE_FSDP"] = "true"

    fsdp_kwargs = (
        {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "fsdp_transformer_layer_cls_to_wrap": "GPT2Block",
                "fsdp_cpu_ram_efficient_loading": True,
                "fsdp_sync_module_states": True,
                "fsdp_use_orig_params": True,
            },
        }
        if distributed
        else {}
    )

    training_args = TrainingArguments(
        output_dir=f"/tmp/fsdp-test-{info.node}",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-5,
        warmup_steps=5,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        logging_steps=5,
        max_steps=20,
        save_strategy="no",
        report_to="none",
        average_tokens_across_devices=False,
        **fsdp_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.filter(lambda x: len(x["text"]) > 50)
    ds = ds.select(range(min(200, len(ds))))

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds = ds.map(lambda x: {"labels": x["input_ids"]})

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    param_count = sum(p.numel() for p in model.parameters())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    # Workaround for transformers bug: GPT2 accepts **kwargs, so the Trainer
    # passes num_items_in_batch into the model's loss_function. Inside
    # fixed_cross_entropy, `loss (scalar []) / num_items_in_batch (tensor [1])`
    # fails with a broadcast shape mismatch. Disabling this prevents the
    # Trainer from passing num_items_in_batch to the model.
    # See: https://github.com/huggingface/transformers/issues/35086
    trainer.model_accepts_loss_kwargs = False

    import traceback

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
        "rank": rank,
        "fsdp_enabled": trainer.is_fsdp_enabled,
        "accelerator_type": str(trainer.accelerator.distributed_type),
        "model_params_m": round(param_count / 1e6, 1),
        "peak_gpu_gb": round(peak_gb, 2),
        "final_loss": train_loss,
        "error": train_error,
    }


if __name__ == "__main__":
    import json

    with sky.Compute(
        provider=sky.AWS(),
        accelerator=sky.accelerators.T4(),
        nodes=2,
        plugins=[sky.plugins.torch()],
        image=sky.Image(pip=["transformers", "datasets", "accelerate"]),
    ) as compute:
        print("=" * 60)
        print("FSDP Test — HuggingFace Trainer + sky.plugins.torch()")
        print("=" * 60)

        results = test_trainer_fsdp(model_name="gpt2") @ compute

        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)

        for r in results:
            print(json.dumps(r, indent=2))
