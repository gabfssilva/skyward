"""Distributed Grid Search with scikit-learn.

Demonstrates how to distribute sklearn's GridSearchCV using Skyward's
joblib backend integration. Zero changes to sklearn code required!

Features:
- Uses sklearn's native GridSearchCV (no custom wrapper)
- Distributes cross-validation folds across cloud workers
- Works with any sklearn estimator that supports n_jobs
- Automatic environment replication to workers
"""

from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from skyward import AWS, ComputePool, Image
from skyward.integrations import sklearn_backend


def run_distributed_grid_search():
    """Run a distributed grid search comparing multiple models."""
    # Load the digits dataset
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}, Classes: {len(set(y))}")

    # Define models and their parameter grids
    models = [
        (
            "RandomForest",
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5],
            },
        ),
        (
            "SVM",
            SVC(),
            {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "poly"],
                "gamma": ["scale", "auto"],
            },
        ),
    ]

    results = []

    with ComputePool(
        provider=AWS(),
        nodes=4,
        cpu=4,
        image=Image(pip=["scikit-learn"]),
        spot="always",
    ) as pool:
        # Enable Skyward as the joblib backend
        with sklearn_backend(pool):
            for name, estimator, param_grid in models:
                n_candidates = 1
                for v in param_grid.values():
                    n_candidates *= len(v)

                print(f"\n{'=' * 60}")
                print(f"Running GridSearchCV for {name}")
                print(f"  Candidates: {n_candidates}")
                print(f"  CV folds: 5")
                print(f"  Total fits: {n_candidates * 5}")
                print("=" * 60)

                # Standard sklearn GridSearchCV - automatically distributed!
                grid_search = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,  # This triggers distributed execution
                    verbose=1,
                )

                grid_search.fit(X_train, y_train)

                # Evaluate on test set
                test_score = grid_search.score(X_test, y_test)

                results.append({
                    "name": name,
                    "best_params": grid_search.best_params_,
                    "best_cv_score": grid_search.best_score_,
                    "test_score": test_score,
                    "n_candidates": n_candidates,
                })

                print(f"\n{name} Results:")
                print(f"  Best CV score: {grid_search.best_score_:.2%}")
                print(f"  Test score: {test_score:.2%}")
                print(f"  Best params: {grid_search.best_params_}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results.sort(key=lambda r: r["test_score"], reverse=True)

    for r in results:
        print(f"\n{r['name']}:")
        print(f"  CV: {r['best_cv_score']:.2%} | Test: {r['test_score']:.2%}")
        params = ", ".join(f"{k}={v}" for k, v in r["best_params"].items())
        print(f"  Params: {params}")

    winner = results[0]
    print(f"\n{'=' * 60}")
    print(f"WINNER: {winner['name']} with {winner['test_score']:.2%} test accuracy")
    print("=" * 60)


if __name__ == "__main__":
    run_distributed_grid_search()
