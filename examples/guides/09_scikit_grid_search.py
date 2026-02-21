"""Scikit Grid Search — distributed hyperparameter search."""

from functools import reduce
from operator import mul

from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import skyward as sky

if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([("clf", SVC())])

    param_grid = [
        {
            "clf": [RandomForestClassifier(random_state=42)],
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [None, 10, 20],
        },
        {
            "clf": [GradientBoostingClassifier(random_state=42)],
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.01, 0.1, 0.2],
        },
        {
            "clf": [SVC()],
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["rbf", "poly"],
        },
    ]

    n_candidates = sum(
        reduce(mul, (len(v) for v in grid.values()), 1)
        for grid in param_grid
    )
    print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"Total candidates: {n_candidates}, fits: {n_candidates * 5}")

    with sky.integrations.ScikitLearnPool(
        provider=sky.AWS(),
        nodes=3,
        worker=sky.Worker(concurrency=4),
        image=sky.Image(pip=["scikit-learn"]),
    ):
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=3,
        )
        grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_params_["clf"]
    cv_score = grid_search.best_score_
    test_score = grid_search.score(X_test, y_test)
    print(f"Best: {type(best_clf).__name__}, CV={cv_score:.2%}, Test={test_score:.2%}")
