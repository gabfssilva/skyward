"""FSDP Fine-tuning with HuggingFace — shard a large model across multiple GPUs."""

import skyward as sky


@sky.function(timeout=1800)
def train_fsdp(
    model_name: str,
    dataset_name: str,
    max_samples: int = 2000,
    num_epochs: int = 1,
    batch_size: int = 2,
) -> dict:
    """Fine-tune a causal LM with FSDP across multiple nodes."""
    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    info = sky.instance_info()
    assert info is not None

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

    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    param_count = sum(p.numel() for p in model.parameters())

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
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
        ),
        train_dataset=ds,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    return {
        "node": info.node,
        "fsdp_enabled": trainer.is_fsdp_enabled,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "model_params_b": round(param_count / 1e9, 1),
        "peak_gpu_gb": round(peak_gb, 2),
        "final_loss": round(train_result.training_loss, 4),
    }


if __name__ == "__main__":
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
        print("FSDP fine-tuning gpt2-xl across 3 × T4")
        print("=" * 60)

        results = train_fsdp(
            model_name="gpt2-xl",
            dataset_name="wikitext",
            max_samples=500,
            num_epochs=1,
            batch_size=1,
        ) @ compute

        for r in results:
            print(json.dumps(r, indent=2))
