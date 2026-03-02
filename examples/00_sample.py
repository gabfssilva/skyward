import skyward as sky


def load_mnist(n_samples: int):
    import numpy as np
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = (X[:n_samples] / 255.0).astype(np.float32)
    y = y[:n_samples].astype(np.int32)
    return X, y


@sky.function
def do_train(n_samples: int) -> dict:
    from time import perf_counter

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X, y = load_mnist(n_samples)

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    start = perf_counter()
    scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    elapsed = perf_counter() - start

    return {"accuracy": scores.mean(), "time": elapsed}


if __name__ == '__main__':
    with sky.ComputePool(
        provider=sky.AWS(),
        accelerator=sky.accelerators.T4(),
        image=sky.Image(
            pip=['scikit-learn'],
        ),
        plugins=[
            sky.plugins.cuml(),
        ]
    ) as pool:
        result = do_train(5000) >> pool

        print(result)
