# HuggingFace Fine-tuning

Fine-tuning a pre-trained transformer is one of the most common ML workflows: take a model from the HuggingFace Hub, adapt it to your task with a small labeled dataset, and evaluate the results. The bottleneck is usually hardware — fine-tuning even a small model like DistilBERT benefits significantly from a GPU, and larger models require one. Skyward lets you wrap the entire pipeline in a single `@sky.compute` function, provision a GPU instance, and run it remotely. Everything — model download, tokenization, training, evaluation — happens on the cloud instance.

## Loading Model and Tokenizer

Load a pre-trained model inside the compute function:

```python
--8<-- "examples/guides/08_huggingface_finetuning.py:23:29"
```

`AutoModelForSequenceClassification.from_pretrained()` downloads the base model and adds a classification head. The download happens on the remote instance, which typically has faster internet than a laptop and avoids transferring multi-GB model weights over the SSH tunnel. The `id2label` and `label2id` mappings configure the model for binary sentiment classification.

## Preparing the Dataset

Load IMDB, tokenize, and prepare for training — all on the remote instance:

```python
--8<-- "examples/guides/08_huggingface_finetuning.py:31:39"
```

`load_dataset("imdb")` downloads the dataset on the worker. The `select(range(max_samples))` call limits the dataset size for faster iteration during development — remove it for a full fine-tuning run. Tokenization runs remotely too, so you don't need `transformers` or `datasets` installed locally.

This is one of the key advantages of remote execution: heavy data processing and model operations happen on a machine with the right hardware and fast network, while your local machine just dispatches the work and collects results.

## Training with the Trainer API

Configure training arguments and launch the Trainer:

```python
--8<-- "examples/guides/08_huggingface_finetuning.py:45:63"
```

The `Trainer` manages the training loop, evaluation, gradient accumulation, and mixed-precision (fp16) when a GPU is available. `eval_strategy="epoch"` runs evaluation after each epoch. `save_strategy="no"` disables checkpointing — since the instance is ephemeral, saved checkpoints would be lost on teardown. For production fine-tuning, you'd save checkpoints to a persistent location (S3, HuggingFace Hub, or a mounted volume).

The function returns a summary dict with training loss, evaluation accuracy, and runtime. This is the result that comes back through the SSH tunnel to your local process.

## Dispatching to the Cloud

The full example dispatches the fine-tuning job to an A100 instance:

```python
result = finetune(
    model_name="distilbert-base-uncased",
    epochs=2,
    batch_size=16,
) >> pool
```

No `@sky.integrations.torch` decorator is needed — the HuggingFace Trainer handles device placement and mixed-precision internally. Skyward's job is to provision the GPU instance, run the function, and return the result. The Trainer API does the rest.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/08_huggingface_finetuning.py
```

---

**What you learned:**

- **Everything runs remotely** — model download, tokenization, training, evaluation all happen on the cloud GPU.
- **No Skyward-specific APIs inside the function** — standard HuggingFace `Trainer`, `AutoModel`, `load_dataset`.
- **Remote imports** — `transformers` and `datasets` only need to be installed on the worker (via the Image's `pip` field), not locally.
- **Ephemeral instances** — checkpoints are lost on teardown; save to persistent storage for production runs.
- **No integration decorator needed** — the HuggingFace Trainer manages device placement and distributed setup internally.
