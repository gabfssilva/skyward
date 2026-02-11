"""HuggingFace Fine-tuning Example.

Demonstrates fine-tuning a pre-trained transformer model:
- Loading models from HuggingFace Hub
- Using the Trainer API
- Distributed training with Accelerate
- Text classification task (sentiment analysis)
"""

import skyward as sky


@sky.compute
def check_environment() -> dict:
    """Verify the training environment."""
    import torch
    import transformers

    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@sky.compute
def finetune_classifier(
    model_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_samples: int = 1000,
) -> dict:
    """Fine-tune a transformer for text classification."""
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    pool = sky.instance_info()

    # =================================================================
    # Load Model and Tokenizer
    # =================================================================
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    # =================================================================
    # Load and Prepare Dataset
    # =================================================================
    # Load a subset of IMDB for demonstration
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"].select(range(max_samples))
    test_dataset = dataset["test"].select(range(max_samples // 4))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding=False,  # Dynamic padding with data collator
        )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # =================================================================
    # Training Configuration
    # =================================================================
    training_args = TrainingArguments(
        output_dir=f"/tmp/finetuned-{model_name.split('/')[-1]}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",  # Don't save checkpoints for demo
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",  # Disable wandb/tensorboard
    )

    # =================================================================
    # Metrics
    # =================================================================
    def compute_metrics(eval_pred):
        import numpy as np

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # =================================================================
    # Train
    # =================================================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    # =================================================================
    # Results
    # =================================================================
    return {
        "node": pool.node,
        "is_head": pool.is_head,
        "model": model_name,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "eval_accuracy": eval_result["eval_accuracy"],
        "train_runtime_seconds": train_result.metrics["train_runtime"],
    }


@sky.compute
def inference_demo(model_name: str, texts: list[str]) -> dict:
    """Run inference with a pre-trained model."""
    from transformers import pipeline

    pool = sky.instance_info()

    # Create sentiment analysis pipeline
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=0 if __import__("torch").cuda.is_available() else -1,
    )

    # Run inference
    results = classifier(texts)

    return {
        "node": pool.node,
        "model": model_name,
        "predictions": [
            {"text": t[:50] + "...", "label": r["label"], "score": round(r["score"], 3)}
            for t, r in zip(texts, results, strict=False)
        ],
    }


if __name__ == "__main__":
    # =================================================================
    # Fine-tuning Setup
    # =================================================================
    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=2,
        accelerator=sky.NVIDIA.A100,
        pip=[
            "transformers>=4.35.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "torch>=2.0.0",
            "scikit-learn",
        ],
        allocation="spot-if-available",
    ) as pool:
        print("=" * 60)
        print("Environment Check")
        print("=" * 60)

        env_info = check_environment() >> pool
        print(f"PyTorch: {env_info['torch_version']}")
        print(f"Transformers: {env_info['transformers_version']}")
        print(f"GPU: {env_info['gpu_name']}")

        # =================================================================
        # Fine-tune DistilBERT
        # =================================================================
        print("\n" + "=" * 60)
        print("Fine-tuning DistilBERT for Sentiment Analysis")
        print("=" * 60)

        results = finetune_classifier(
            model_name="distilbert-base-uncased",
            num_epochs=2,
            batch_size=16,
            learning_rate=2e-5,
            max_samples=500,  # Small for demo
        ) @ pool

        print("\nTraining Results:")
        for r in results:
            role = "HEAD" if r["is_head"] else "WORKER"
            print(f"\nNode {r['node']} ({role}):")
            print(f"  Model: {r['model']}")
            print(f"  Train samples: {r['train_samples']}")
            print(f"  Train loss: {r['train_loss']:.4f}")
            print(f"  Eval accuracy: {r['eval_accuracy']:.2%}")
            print(f"  Runtime: {r['train_runtime_seconds']:.1f}s")

        # =================================================================
        # Inference Demo
        # =================================================================
        print("\n" + "=" * 60)
        print("Inference Demo")
        print("=" * 60)

        sample_texts = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Terrible experience. Waste of time and money.",
            "It was okay, nothing special but not bad either.",
        ]

        inference_result = inference_demo(
            model_name="distilbert-base-uncased-finetuned-sst-2-english",
            texts=sample_texts,
        ) >> pool

        print("\nPredictions:")
        for pred in inference_result["predictions"]:
            print(f"  [{pred['label']}] ({pred['score']:.1%}) {pred['text']}")
