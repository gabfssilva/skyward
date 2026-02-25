"""Distributed Grid Search with scikit-learn.

Demonstrates a single GridSearchCV that searches over both estimators
AND their hyperparameters using Skyward's distributed joblib backend.

Uses sklearn's native Pipeline with list-of-dicts param_grid to treat
the estimator itself as a searchable hyperparameter.
"""

from functools import reduce
from operator import mul

from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import skyward as sky


def main():
    """Run a unified grid search over estimators and their hyperparameters."""
    X, y = load_digits(return_X_y=True)  # noqa: N806
    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}, Classes: {len(set(y))}")

    # Pipeline where the classifier step can be swapped via param_grid
    pipe = Pipeline([("clf", SVC())])

    # List of dicts: each dict defines a separate grid for one estimator family
    # The "clf" parameter swaps the estimator; "clf__param" tunes its hyperparams
    param_grid = [
        {
            "clf": [RandomForestClassifier(random_state=42)],
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
        },
        {
            "clf": [GradientBoostingClassifier(random_state=42)],
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth": [3, 5],
        },
        {
            "clf": [SVC()],
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["rbf", "poly"],
            "clf__gamma": ["scale", "auto"],
        },
    ]

    n_candidates = sum(
        reduce(mul, (len(v) for v in grid.values()), 1)
        for grid in param_grid
    )
    cv_folds = 5

    print(f"\n{'=' * 60}")
    print("Running unified GridSearchCV across all estimators")
    print(f"  Estimator families: {len(param_grid)}")
    print(f"  Total candidates: {n_candidates}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Total fits: {n_candidates * cv_folds}")
    print("=" * 60)

    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=3,
        worker=sky.Worker(concurrency=4),
        allocation="spot",
        plugins=[
            sky.plugins.sklearn()
        ]
    ):
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=3,
        )
        grid_search.fit(X_train, y_train)

    test_score = grid_search.score(X_test, y_test)
    best_clf = grid_search.best_params_["clf"]
    best_params = {
        k.removeprefix("clf__"): v
        for k, v in grid_search.best_params_.items()
        if k != "clf"
    }

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"Best estimator: {type(best_clf).__name__}")
    print(f"Best params: {best_params}")
    print(f"Best CV score: {grid_search.best_score_:.2%}")
    print(f"Test score: {test_score:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
