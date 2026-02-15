"""HuggingFace Fine-tuning — fine-tune a transformer on cloud GPUs."""

import skyward as sky


@sky.compute
def finetune(model_name: str, epochs: int, batch_size: int, max_samples: int = 500) -> dict:
    """Fine-tune a transformer for sentiment analysis."""
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    info = sky.instance_info()
    assert info is not None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    dataset = load_dataset("imdb")
    train_ds = dataset["train"].select(range(max_samples))
    test_ds = dataset["test"].select(range(max_samples // 4))

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return {"accuracy": (preds == eval_pred.label_ids).mean()}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/finetuned",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            report_to="none",
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    return {
        "node": info.node,
        "model": model_name,
        "train_loss": train_result.training_loss,
        "eval_accuracy": eval_result["eval_accuracy"],
        "runtime": train_result.metrics["train_runtime"],
    }


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.AWS(),
        accelerator="A100",
        image=sky.Image(pip=["transformers", "datasets", "accelerate", "torch", "scikit-learn"]),
    ) as pool:
        result = finetune(
            model_name="distilbert-base-uncased",
            epochs=2,
            batch_size=16,
        ) >> pool

        print(f"Model: {result['model']}")
        print(f"Train loss: {result['train_loss']:.4f}")
        print(f"Eval accuracy: {result['eval_accuracy']:.2%}")
        print(f"Runtime: {result['runtime']:.1f}s")
