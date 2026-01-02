"""Test if sklearn functions can be executed remotely."""

import cloudpickle
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from skyward import AWS, ComputePool, Image, compute


@compute
def run_sklearn_directly() -> dict:
    """Run sklearn directly on remote (no serialization of sklearn objects)."""
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X, y = load_digits(return_X_y=True)
    X, y = X[:300], y[:300]

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3)

    return {"mean": float(scores.mean()), "std": float(scores.std())}


@compute
def run_sklearn_from_payload(payload: bytes) -> list:
    """Run a serialized sklearn function on remote."""
    import cloudpickle

    func = cloudpickle.loads(payload)
    return func()


if __name__ == "__main__":
    print("Test 1: Run sklearn directly on remote...")

    with ComputePool(
        provider=AWS(),
        nodes=1,
        cpu=2,
        image=Image(pip=["scikit-learn"]),
        spot="always",
    ) as pool:
        result = run_sklearn_directly() >> pool
        print(f"  Result: {result}")
        print("  ✅ Direct execution works!")

        print("\nTest 2: Run serialized sklearn function...")

        # Create a function like joblib would
        X, y = load_digits(return_X_y=True)
        X, y = X[:300], y[:300]

        def fit_and_score():
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            scores = cross_val_score(clf, X, y, cv=3)
            return [float(scores.mean())]

        payload = cloudpickle.dumps(fit_and_score)
        print(f"  Payload size: {len(payload)} bytes")

        result = run_sklearn_from_payload(payload) >> pool
        print(f"  Result: {result}")
        print("  ✅ Serialized function works!")

    print("\n✅ All tests passed!")
