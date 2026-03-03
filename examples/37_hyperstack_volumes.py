"""Hyperstack Object Storage Volumes.

End-to-end workflow: upload a dataset locally, train a model on the cluster,
and download the trained model — all via S3-compatible object storage.

    ┌──────────┐  Storage.upload()        ┌──────────────────┐
    │  Local   │ ──────────────────────→  │  S3 Bucket       │
    │  Machine │ ←────────────────────── │  (Hyperstack)    │
    └──────────┘  Storage.download()      └──────────────────┘
                                                 ↕  s3fs-fuse
                                         ┌──────────────────┐
                                         │  Hyperstack VM   │
                                         │  /data  /model   │
                                         └──────────────────┘

Credentials are resolved lazily from environment variables:
    HYPERSTACK_ACCESS_KEY, HYPERSTACK_SECRET_KEY
"""

import os
import time
from pathlib import Path

import skyward as sky


@sky.function
def train(data_path: str, model_dir: str) -> dict:
    """Load dataset from volume, train a model, save to output volume."""
    import pickle

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Load dataset from mounted volume
    raw = np.loadtxt(data_path, delimiter=",", skiprows=1)
    features, labels = raw[:, :-1], raw[:, -1].astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)

    acc = accuracy_score(labels, model.predict(features))

    # Save trained model to the output volume
    out = Path(model_dir) / "model.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(model, f)

    # Wait for s3fs to sync the write
    time.sleep(5)

    return {
        "samples": len(features),
        "accuracy": round(acc, 4),
        "model_bytes": out.stat().st_size,
    }


if __name__ == "__main__":
    DATA_BUCKET = "my-dataset-bucket"
    MODEL_BUCKET = "my-model-bucket"

    # Storage with lazy credential resolution from environment variables
    storage = sky.Storage(
        endpoint="https://objects.ord1.hyperstack.cloud",
        access_key=lambda: os.environ["HYPERSTACK_ACCESS_KEY"],
        secret_key=lambda: os.environ["HYPERSTACK_SECRET_KEY"],
        path_style=True,
    )

    # Volumes with explicit storage — no provider fallback needed
    data_volume = sky.Volume(
        bucket=DATA_BUCKET,
        mount="/data",
        prefix="iris/",
        read_only=True,
        storage=storage,
    )
    model_volume = sky.Volume(
        bucket=MODEL_BUCKET,
        mount="/model",
        prefix="experiment-001/",
        read_only=False,
        storage=storage,
    )

    # ── 1. Generate dataset locally and upload to volume ─────────────
    print("Preparing dataset...")
    import numpy as np
    from sklearn.datasets import load_iris

    iris = load_iris()

    dataset = np.column_stack([iris.data, iris.target])
    csv_path = Path("/tmp/iris.csv")
    header = ",".join(iris.feature_names + ["target"])
    np.savetxt(csv_path, dataset, delimiter=",", header=header, comments="")

    print(f"  {len(dataset)} samples, {iris.data.shape[1]} features")

    print("Uploading to volume...")
    with storage:
        storage.upload(DATA_BUCKET, csv_path, key="iris.csv")
        print(f"  Uploaded: {storage.ls(DATA_BUCKET)}")

    # ── 2. Train on the cluster ──────────────────────────────────────
    with sky.ComputePool(
        provider=sky.Hyperstack(),
        accelerator=sky.accelerators.L4(),
        image=sky.Image(pip=["scikit-learn", "numpy"]),
        volumes=[data_volume, model_volume],
    ) as pool:
        print("\nTraining...")
        result = train("/data/iris.csv", "/model") >> pool
        print(f"  {result['samples']} samples, acc={result['accuracy']}, model={result['model_bytes']} bytes")

    # ── 3. Download trained model locally ────────────────────────────
    print("\nDownloading model...")
    model_path = Path("/tmp/trained_model.pkl")

    with storage:
        storage.download(MODEL_BUCKET, "model.pkl", model_path)

    import pickle

    with open(model_path, "rb") as f:
        model = pickle.load(f)  # noqa: S301

    preds = model.predict(iris.data)
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(iris.target, preds)
    print(f"  Loaded model: {type(model).__name__}, full-dataset accuracy={acc:.4f}")
