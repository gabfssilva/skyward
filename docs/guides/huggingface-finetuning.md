# HuggingFace Fine-tuning

In this guide you'll **fine-tune a pre-trained transformer** for sentiment analysis using the HuggingFace Trainer API on a cloud GPU.

## Loading Model and Tokenizer

Load a pre-trained model from the HuggingFace Hub:

```python
--8<-- "examples/guides/08_huggingface_finetuning.py:23:29"
```

`AutoModelForSequenceClassification` adds a classification head on top of the transformer.

## Preparing the Dataset

Load IMDB, tokenize, and prepare for training:

```python
--8<-- "examples/guides/08_huggingface_finetuning.py:31:39"
```

Tokenization runs on the remote instance — no need to transfer tokenized data.

## Training with the Trainer API

Configure training and launch:

```python
--8<-- "examples/guides/08_huggingface_finetuning.py:45:63"
```

The Trainer handles the training loop, evaluation, and mixed-precision (fp16) automatically.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/08_huggingface_finetuning.py
```

---

**What you learned:**

- **HuggingFace models** run on cloud GPUs with a single `@sky.compute` function.
- **Trainer API** handles training loops, evaluation, and fp16.
- **Remote execution** means tokenization and training happen on the GPU instance.
- **No distributed setup needed** — just send the function to the pool.
