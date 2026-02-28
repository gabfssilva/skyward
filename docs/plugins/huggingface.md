# HuggingFace

The HuggingFace ecosystem — transformers, datasets, tokenizers — is the dominant interface for working with pre-trained language models. Fine-tuning a model from the Hub is a well-understood workflow: download a checkpoint, tokenize a dataset, configure the Trainer, and run. The friction is not the code but the environment. You need a GPU, you need the right CUDA version, you need the HuggingFace libraries installed, and if the model is gated, you need to authenticate before anything downloads.

Skyward's `huggingface` plugin handles all of this at the pool level. It installs the core HuggingFace libraries on the worker, sets the `HF_TOKEN` environment variable, and runs `huggingface-cli login` during bootstrap so that gated model downloads work without any manual intervention inside your compute function. Your function just calls `from_pretrained()` and it works — even for gated models like Llama or Mistral.

## What it does

**Image transform** — Appends `transformers`, `datasets`, and `tokenizers` to the worker's pip dependencies and sets `HF_TOKEN` in the environment. This means the libraries are installed during bootstrap and the token is available to every process on the worker. You do not need to specify these packages in the `Image` yourself, and you do not need to call `huggingface_hub.login()` inside your function.

**Bootstrap** — After the base environment is set up, the plugin runs `huggingface-cli login --token $HF_TOKEN`. This writes the token to the Hub's local credential store (`~/.cache/huggingface/token`), which is where `from_pretrained()` looks when downloading gated models. The login happens once during instance bootstrap, not on every task.

Together, these two hooks mean that by the time your `@sky.compute` function runs, the worker has the HuggingFace libraries installed, the CLI authenticated, and the token in the environment. Your function can focus on the actual ML work.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `token` | `str` | *(required)* | HuggingFace API token. Used for `HF_TOKEN` env var and CLI login. Required for gated models; recommended for all usage to avoid rate limits. |

The token is passed as a plain string. It ends up in the worker's environment as `HF_TOKEN` and in the Hub's credential store via `huggingface-cli login`. If you are working with gated models (Llama, Mistral, Gemma), you must accept the model's license on the Hub before the token will grant access.

## How it works

When `ComputePool.__enter__` runs, the plugin's image transform modifies the `Image` before bootstrap script generation:

```python
Image(
    pip=(*existing_pip, "transformers", "datasets", "tokenizers"),
    env={**existing_env, "HF_TOKEN": token},
)
```

The bootstrap script generator then picks up these additions: `uv` installs the three packages, and the environment variable is exported in the shell profile. After the base bootstrap completes, the plugin's bootstrap hook runs a single command:

```
huggingface-cli login --token $HF_TOKEN
```

This writes the token to disk. From that point on, any HuggingFace library call that needs authentication — `AutoModel.from_pretrained("meta-llama/...")`, `load_dataset("private/dataset")` — can find the credentials without any explicit login call in your code.

## Usage

### Single-node fine-tuning

The most common pattern is single-node fine-tuning with the Trainer API. The plugin handles dependencies and authentication; HuggingFace's Trainer handles device placement:

```python
import skyward as sky


@sky.compute
def finetune(model_name: str, epochs: int) -> dict:
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    )

    dataset = load_dataset("imdb")
    train = dataset["train"].map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=256),
        batched=True,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/finetune",
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            fp16=True,
        ),
        train_dataset=train,
    )

    trainer.train()
    return trainer.evaluate()


with sky.ComputePool(
    provider=sky.AWS(),
    nodes=1,
    accelerator="A100",
    plugins=[sky.plugins.huggingface(token="hf_...")],
) as pool:
    result = finetune("distilbert-base-uncased", epochs=3) >> pool
    print(result)
```

Notice that the function imports `transformers` and `datasets` inside the function body. This is deliberate: the function is serialized with cloudpickle and sent to the remote worker, where the imports resolve against the worker's environment. You do not need `transformers` installed locally.

The Trainer handles device placement internally — it detects the GPU and moves the model there, enables fp16 when appropriate, and manages the training loop. For a single node, no distributed coordination is needed.

### Multi-node distributed training

For larger models or faster iteration, distribute training across multiple nodes. The HuggingFace Trainer integrates with PyTorch's DistributedDataParallel when the process group is initialized. Combine the `huggingface` plugin with the `torch` plugin:

```python
@sky.compute
def distributed_finetune(model_name: str) -> dict:
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    # Trainer handles data sharding across DDP ranks

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/distributed-finetune",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            fp16=True,
            ddp_find_unused_parameters=False,
        ),
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer.evaluate()


with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    accelerator="A100",
    plugins=[
        sky.plugins.torch(backend="nccl"),
        sky.plugins.huggingface(token="hf_..."),
    ],
) as pool:
    results = distributed_finetune("gpt2") @ pool
```

The `torch` plugin initializes the process group before your function runs — it sets `MASTER_ADDR`, `WORLD_SIZE`, `RANK`, and calls `init_process_group()`. The Trainer detects the initialized process group and automatically wraps the model with `DistributedDataParallel`, shards the data across ranks, and synchronizes gradients. The `huggingface` plugin contributes the libraries and authentication. Plugin order in the list does not affect behavior — each plugin's hooks run at their respective lifecycle points.

### Inference with pipelines

For lighter workloads like batch inference, the plugin is equally useful — it ensures the libraries and authentication are in place:

```python
@sky.compute
def classify(texts: list[str]) -> list[dict]:
    from transformers import pipeline

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0,
    )
    return classifier(texts)


with sky.ComputePool(
    provider=sky.AWS(),
    nodes=1,
    accelerator="T4",
    plugins=[sky.plugins.huggingface(token="hf_...")],
) as pool:
    predictions = classify(["Great movie!", "Terrible film."]) >> pool
```

## Next steps

- [HuggingFace Fine-tuning guide](../guides/huggingface-finetuning.md) — Complete fine-tuning walkthrough with dataset preparation, training, and evaluation
- [PyTorch Distributed](../guides/pytorch-distributed.md) — How DDP works under the hood with the `torch` plugin
- [What are Plugins?](index.md) — How the plugin system works
