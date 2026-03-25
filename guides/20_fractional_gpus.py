"""Fractional GPUs — run inference on a slice of a GPU for lower cost."""

import skyward as sky


@sky.function
def classify(texts: list[str]) -> dict:
    """Classify sentiment on a fractional GPU."""
    import torch
    from transformers import pipeline

    info = sky.instance_info()
    device = torch.device("cuda")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )
    results = classifier(texts)

    return {
        "predictions": results,
        "device": str(device),
        "vram_gb": round(vram_gb, 1),
        "accelerators": info.accelerators,
        "node": info.node,
    }


TEXTS = [
    "Fractional GPUs are a great way to save money on inference workloads.",
    "Paying for a full A100 to run DistilBERT would be absurdly wasteful.",
    "Right-sizing your GPU allocation is the easiest cloud cost optimization.",
]


if __name__ == "__main__":
    with sky.Compute(
        provider=sky.AWS(),
        accelerator=sky.accelerators.L4(count=0.5),
        image=sky.Image(pip=["torch", "transformers"]),
        options=sky.Options(provision_timeout=600),
    ) as compute:
        result = classify(TEXTS) >> compute

        print(f"Device: {result['device']} ({result['vram_gb']} GB VRAM)")
        print(f"Accelerator count: {result['accelerators']}")
        for text, pred in zip(TEXTS, result["predictions"]):
            print(f"  {pred['label']} ({pred['score']:.2f}): {text[:60]}")
