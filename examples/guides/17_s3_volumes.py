"""S3 Volumes — end-to-end: upload data, train on cluster, download results.

    ┌──────────┐  Storage.upload()        ┌──────────────────┐
    │  Local   │ ──────────────────────→  │  S3 Bucket       │
    │  Machine │ ←────────────────────── │  (provider)      │
    └──────────┘  Storage.download()      └──────────────────┘
                                                 ↕  s3fs-fuse
                                         ┌──────────────────┐
                                         │  Remote Worker   │
                                         │  /data  /output  │
                                         └──────────────────┘
"""

from pathlib import Path

import skyward as sky


@sky.function
def train(data_path: str, output_dir: str) -> dict:
    """Load dataset from volume, train a model, save to output volume."""
    import pickle

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    raw = np.loadtxt(data_path, delimiter=",", skiprows=1)
    features, labels = raw[:, :-1], raw[:, -1].astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)

    acc = accuracy_score(labels, model.predict(features))

    out = Path(output_dir) / "model.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(model, f)

    return {
        "samples": len(features),
        "accuracy": round(acc, 4),
        "model_bytes": out.stat().st_size,
    }


if __name__ == "__main__":
    DATA_BUCKET = "my-dataset-bucket"
    MODEL_BUCKET = "my-model-bucket"

    # Explicit storage for standalone CRUD (upload/download outside a pool)
    storage = sky.Storage(
        endpoint="https://objects.ord1.hyperstack.cloud",
        access_key="YOUR_ACCESS_KEY",
        secret_key="YOUR_SECRET_KEY",
        path_style=True,
    )

    data_volume = sky.Volume(
        bucket=DATA_BUCKET,
        mount="/data",
        prefix="iris/",
        read_only=True,
        storage=storage,
    )
    model_volume = sky.Volume(
        bucket=MODEL_BUCKET,
        mount="/output",
        prefix="experiment-001/",
        read_only=False,
        storage=storage,
    )

    import numpy as np
    from sklearn.datasets import load_iris

    iris = load_iris()
    dataset = np.column_stack([iris.data, iris.target])
    csv_path = Path("/tmp/iris.csv")
    header = ",".join(iris.feature_names + ["target"])
    np.savetxt(csv_path, dataset, delimiter=",", header=header, comments="")

    with storage:
        storage.upload(DATA_BUCKET, csv_path, key="iris.csv")
        print(f"Uploaded: {storage.ls(DATA_BUCKET)}")

    with sky.ComputePool(
        provider=sky.Hyperstack(),
        accelerator=sky.accelerators.L4(),
        image=sky.Image(pip=["scikit-learn", "numpy"]),
        volumes=[data_volume, model_volume],
    ) as pool:
        result = train("/data/iris.csv", "/output") >> pool
        print(f"{result['samples']} samples, acc={result['accuracy']}, model={result['model_bytes']} bytes")

    model_path = Path("/tmp/trained_model.pkl")

    with storage:
        storage.download(MODEL_BUCKET, "model.pkl", model_path)

    import pickle

    with open(model_path, "rb") as f:
        model = pickle.load(f)  # noqa: S301

    from sklearn.metrics import accuracy_score

    preds = model.predict(iris.data)
    acc = accuracy_score(iris.target, preds)
    print(f"Loaded model: {type(model).__name__}, accuracy={acc:.4f}")
