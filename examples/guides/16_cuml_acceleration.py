"""GPU-Accelerated Scikit-Learn with cuML.

Demonstrates how NVIDIA cuML can accelerate scikit-learn on GPUs.
The CPU version uses standard scikit-learn; the GPU version activates cuML's
zero-code-change acceleration. Same algorithm, same data, same hyperparameters
— only the backend changes.
"""

import skyward as sky

N_SAMPLES = 52667


def load_mnist(n_samples: int):  # noqa: N806
    """Load a subset of MNIST on the remote worker."""
    import numpy as np
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)  # noqa: N806
    X = (X[:n_samples] / 255.0).astype(np.float32)  # noqa: N806
    y = y[:n_samples].astype(np.int32)
    return X, y


@sky.compute
def train_on_gpu(n_samples: int):
    """Train the same RandomForest, but with cuML GPU acceleration."""
    from time import perf_counter

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X, y = load_mnist(n_samples)  # noqa: N806

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    start = perf_counter()
    scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    elapsed = perf_counter() - start

    return {"accuracy": scores.mean(), "time": elapsed}


if __name__ == "__main__":
    print(f"MNIST subset: {N_SAMPLES:,} samples, 784 features\n")

    with sky.ComputePool(
        provider=sky.AWS(),
        accelerator=sky.accelerators.L4(),
        nodes=1,
        plugins=[
            sky.plugins.cuml(),
            sky.plugins.sklearn()
        ],
    ) as pool:
        # warm-up
        train_on_gpu(N_SAMPLES) >> pool

        result = train_on_gpu(N_SAMPLES) >> pool

    print(f"accuracy: {result['accuracy']:.2%}, time: {result['time']:.1f}s")
